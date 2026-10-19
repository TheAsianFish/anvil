"""`make eval-export`: does the exported student agree with the trained one, in fp32 and int8?

For each dataset the `eval.classify` student is exported with `anvil.export.export_student`,
exactly as `anvil export --int8` does, with every held-out input as a parity probe and its
teacher label for the int8 kappa gate; the bundle carries the calibration `make
eval-calibration` wrote (computed here if it is missing). Then a quantization probe answers
the question the M1 spike left open on a random head: which layers cost int8 its accuracy on
a trained one? Each linear-layer group of the fp32 graph is quantized alone (every other
MatMul and Gemm stays fp32), plus all of them together and the shipped default (all but the
MLP down-projections), and each variant is scored against the fp32 ONNX logits and the
teacher labels.

Raw output: `eval/results/export-<dataset>-<teacher>/` (`report.json`, `meta.json`). The
bundles stay in the eval workspace (`.anvil/eval/<task>-<sample>/exports/<student>/`), where
`make bench` finds them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anvil import __version__, log
from anvil.errors import AnvilError
from anvil.settings import Settings
from eval.classify import TeacherKind, git_provenance
from eval.render import SMOKE_SUFFIX
from eval.student import TrainedStudent, trained_student

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
DATASETS = ("banking77", "ag_news")

logger = log.get_logger("anvil.eval.export")


def bundle_dir(ts: TrainedStudent) -> Path:
    """Where `make eval-export` puts the bundle of the eval student (and `make bench` reads it)."""
    return ts.workspace.root / "exports" / ts.path.name


def linear_groups(model: object, config: object) -> dict[str, list[str]]:
    """The fp32 graph's MatMul, Gemm and embedding Gather nodes by the layer they implement.

    The dynamo exporter names weights `val_<n>`, so groups are told apart by weight shape and
    op (ModernBERT: Wqkv is [hidden, 3 * hidden], attention Wo [hidden, hidden], MLP Wi
    [hidden, 2 * intermediate], MLP Wo [intermediate, hidden]; the pooling head and the
    classifier are Gemms), plus the token embedding Gather, which `quantize_dynamic` also
    quantizes by default. MatMuls between two activations (attention scores and the weighted
    sum) are left out: dynamic quantization does not touch them.
    """
    import onnx

    assert isinstance(model, onnx.ModelProto)
    h = int(config.hidden_size)  # type: ignore[attr-defined]
    i = int(config.intermediate_size)  # type: ignore[attr-defined]
    shapes = {i_.name: list(i_.dims) for i_ in model.graph.initializer}
    by_shape = {
        (h, 3 * h): "attention Wqkv",
        (h, h): "attention Wo",
        (h, 2 * i): "MLP Wi",
        (i, h): "MLP Wo",
    }
    groups: dict[str, list[str]] = {}
    seen_square = 0  # [hidden, 3 * hidden] weights met so far (see below)
    for node in model.graph.node:
        if node.op_type == "Gemm":
            name = "head and classifier"
        elif node.op_type == "MatMul":
            weights = [shapes[x] for x in node.input if x in shapes]
            if not weights:
                continue  # between two activations: dynamic quantization leaves it in fp32
            else:
                shape = (int(weights[0][0]), int(weights[0][1]))
                if shape == (h, 3 * h) and 3 * h == 2 * i:
                    # ModernBERT-base: Wqkv and MLP Wi are both [768, 2304]. Graph order is
                    # topological and each layer's attention comes before its MLP, so the
                    # two alternate.
                    name = "attention Wqkv" if seen_square % 2 == 0 else "MLP Wi"
                    seen_square += 1
                else:
                    name = by_shape.get(shape, "other MatMul")
        elif node.op_type == "Gather" and node.input and node.input[0] in shapes:
            name = "embeddings"
        else:
            continue
        groups.setdefault(name, []).append(node.name)
    return groups


def quantization_probe(
    ts: TrainedStudent, bundle: Path, texts: list[str], teacher: list[str]
) -> list[dict[str, object]]:
    """Score int8 variants that quantize one linear group at a time against fp32 ONNX."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    from anvil.calibrate import THRESHOLD_TOLERANCE
    from anvil.evaluate.metrics import cohen_kappa
    from anvil.export import (
        embedding_nodes,
        mlp_down_projection_nodes,
        strip_initializer_value_info,
    )
    from anvil.runtime import BUNDLE_META, VARIANT_FILES, OnnxStudent
    from anvil.students.encoder import EncoderClassifier

    config = EncoderClassifier.load(ts.path).model.config
    meta = json.loads((bundle / BUNDLE_META).read_text(encoding="utf-8"))
    fp32 = OnnxStudent.load(bundle, "fp32")
    ref = fp32.logits(texts)
    labels = fp32.labels
    # The inputs the cascade answers at the bundle's threshold: an int8 flip on an input that
    # escalates anyway never reaches a caller.
    answered = (
        fp32.predict_proba(texts).max(axis=1) >= fp32.threshold - THRESHOLD_TOLERANCE
        if fp32.threshold is not None
        else np.ones(len(texts), dtype=bool)
    )
    ref_kappa = cohen_kappa([labels[k] for k in ref.argmax(axis=1)], teacher)
    model = onnx.load(str(bundle / VARIANT_FILES["fp32"]))
    strip_initializer_value_info(model)
    groups = linear_groups(model, config)
    every = [n for names in groups.values() for n in names]
    variants: list[tuple[str, list[str]]] = [
        ("everything", []),
        ("default: all but MLP Wo", mlp_down_projection_nodes(model, config)),
        (
            "all but MLP Wo and embeddings",
            mlp_down_projection_nodes(model, config) + embedding_nodes(model),
        ),
    ]
    # The quantizer rewrites each Gemm before it reads the exclusion list, so the head and the
    # classifier are quantized in every variant (measured on Banking77: excluding them by name
    # changed no output and no byte). "only head and classifier" is therefore the floor every
    # other row sits on.
    variants += [
        (f"only {g}", [n for n in every if n not in set(names)]) for g, names in groups.items()
    ]
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(dir=bundle.parent) as tmp:
        for name, exclude in variants:
            path = Path(tmp) / f"{len(rows)}.onnx"
            started = time.perf_counter()
            fresh = onnx.ModelProto()
            fresh.CopyFrom(model)  # the quantizer may rewrite the graph it is given
            quantize_dynamic(
                fresh,
                path,
                weight_type=QuantType.QInt8,
                per_channel=False,
                nodes_to_exclude=exclude,
            )
            got = OnnxStudent(bundle, path, meta, "int8").logits(texts)
            diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
            same = got.argmax(axis=1) == ref.argmax(axis=1)
            kappa = cohen_kappa([labels[k] for k in got.argmax(axis=1)], teacher)
            rows.append(
                {
                    "variant": name,
                    "quantized_nodes": len(every) - len(exclude),
                    "fp32_nodes": len(exclude),
                    "size_mb": round(path.stat().st_size / 1e6, 1),
                    "max_abs_diff": float(diff.max()),
                    "mean_abs_diff": float(diff.mean()),
                    "agreement": float(np.mean(same)),
                    "agreement_answered": float(np.mean(same[answered]))
                    if answered.any()
                    else None,
                    "kappa": kappa,
                    "kappa_drop": ref_kappa - kappa,
                    "seconds": round(time.perf_counter() - started, 1),
                }
            )
            logger.info("quantization probe", extra=rows[-1])
            path.unlink()
    head = {
        "groups": {g: len(n) for g, n in groups.items()},
        "fp32_kappa": ref_kappa,
        "answered": int(answered.sum()),
    }
    return [head, *rows]


def run(name: str, teacher: TeacherKind, settings: Settings, probe: bool) -> dict[str, object]:
    from anvil.export import export_student
    from anvil.pipeline import calibrate_student, read_calibration

    ts = trained_student(name, teacher, settings)
    calibration = read_calibration(ts.path)
    if calibration is None or calibration.get("teacher_model") != ts.teacher_model:
        calibration = calibrate_student(ts.spec, ts.workspace, ts.path).calibration.as_dict()
    texts = [it.text for it, _ in ts.heldout]
    teacher_labels = [lab for _, lab in ts.heldout]
    bundle = bundle_dir(ts)
    started = time.perf_counter()
    exported = export_student(
        ts.path,
        bundle,
        texts,
        int8=True,
        teacher_labels=teacher_labels,
        calibration=calibration,
        spec=ts.spec.model_dump(mode="json"),
    )
    export_seconds = time.perf_counter() - started
    report: dict[str, object] = {
        "source": {
            "dataset": name,
            "sample_sha256": ts.data.sha256,
            "teacher_kind": ts.kind,
            "teacher_model": ts.teacher_model,
            "student_model": ts.spec.student.model,
            "student_run": ts.path.name,
            "trained_here": ts.trained_here,
        },
        "spent_usd": ts.spent_usd,
        "probes": len(texts),
        "export_seconds": round(export_seconds, 1),
        "fp32": asdict(exported.fp32),
        "int8": asdict(exported.int8) if exported.int8 is not None else None,
        "int8_fp32_nodes": exported.excluded_nodes,
        "calibration": {k: calibration.get(k) for k in ("temperature", "threshold", "coverage")},
    }
    if probe:
        groups, *rows = quantization_probe(ts, bundle, texts, teacher_labels)
        report["quantization_probe"] = {**groups, "variants": rows}
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.export", description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--teacher", choices=["auto", "claude", "replay", "gold"], default="auto")
    parser.add_argument("--no-probe", action="store_true", help="skip the quantization probe")
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.export " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    try:
        for name in args.datasets:
            started = time.perf_counter()
            report = run(name, args.teacher, settings, probe=not args.no_probe)
            source = report["source"]
            assert isinstance(source, dict)
            out = RESULTS_DIR / (
                f"export-{name}-{source['teacher_model']}"
                + (SMOKE_SUFFIX if args.no_probe else "")
            )
            out.mkdir(parents=True, exist_ok=True)
            meta = {
                "command": command,
                **provenance,
                "anvil": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpus": os.cpu_count(),
                "versions": _versions(),
                "teacher_model": source["teacher_model"],
                "wall_seconds": round(time.perf_counter() - started, 1),
                "spent_usd": report["spent_usd"],
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
            fp32, int8 = report["fp32"], report["int8"]
            assert isinstance(fp32, dict)
            verdict = {True: "pass", False: "FAIL"}
            line = f"{name}: fp32 max diff {fp32['max_abs_diff']:.2e} ({verdict[fp32['passed']]})"
            if isinstance(int8, dict):
                line += (
                    f", int8 agreement {int8['agreement']:.3f}, kappa drop"
                    f" {int8['kappa_drop']:+.3f} ({verdict[int8['passed']]})"
                )
            print(f"{line}; raw output {out}")  # noqa: T201 - the final summary
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    return 0


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in ("torch", "transformers", "onnx", "onnxruntime", "onnxscript", "tokenizers"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


if __name__ == "__main__":
    raise SystemExit(main())
