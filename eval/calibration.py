"""`make eval-calibration`: is the student's confidence usable for a cascade, and what coverage
does a threshold buy at a target kappa with the teacher?

For each dataset, the `eval.classify` student (`eval/student.py`) scores its held-out inputs
once; everything after that runs on those logits, which are committed (`logits.npz`), so the
study re-runs from the raw output in seconds with `--logits`:

1. On the whole held-out split: the temperature, NLL and ECE before and after, the reliability
   bins, the coverage-vs-parity curve, and the threshold `anvil calibrate` would serve (target
   kappa 0.9 and 0.95, rule `lower`). These are in-sample: the threshold is picked and scored
   on the same items, which is what the product does with the labels it has.
2. The out-of-sample check (`splits`): the held-out split is cut in half at random many times;
   the temperature and the threshold are fitted on one half and the other half says whether
   the cascade really keeps the target kappa on the inputs it answers, and at what coverage.
   Four ways to pick the threshold are compared: `point` (the kappa estimate reaches the
   target), `lower` (its one-sided 95% percentile-bootstrap lower bound does), `bca` (its
   one-sided 95% BCa lower bound does) and `point-raw` (point rule on uncalibrated
   confidences), so the effect of the bound, of its method and of the temperature are each
   visible.
3. The coverage cost of each rule on the whole held-out split (`by_rule`): the threshold,
   coverage and lower bound every rule would serve at each target, from the same logits.

Raw output: `eval/results/calibration-<dataset>-<teacher>/` (`report.json`, `meta.json`,
`logits.npz`). It also writes `calibration.json` into the student's directory (target 0.9,
rule `lower`), which `make eval-export` puts into the ONNX bundle.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from anvil import __version__, log
from anvil.calibrate import (
    Rule,
    calibrate,
    choose_threshold,
    coverage_curve,
    ece,
    nll,
    reliability,
    serving_temperature,
    softmax,
)
from anvil.errors import AnvilError, ConfigError
from anvil.evaluate.bootstrap import kappa_replicates, resample_indices
from anvil.pipeline import write_calibration
from anvil.settings import Settings
from anvil.teacher.labelfile import read_label_file
from eval.classify import TeacherKind, file_safe, git_provenance, label_file
from eval.render import SMOKE_SUFFIX
from eval.student import TrainedStudent, trained_student

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
DATASETS = ("banking77", "ag_news")
TARGETS = (0.9, 0.95)
SERVED_TARGET = 0.9
SERVED_RULE: Rule = "lower"
RULES: tuple[Rule, ...] = ("point", "lower", "bca")
COVERAGE_GRID = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5)
N_BINS = 15
MIN_ACCEPTED = 30
DEFAULT_SPLITS = 200
DEFAULT_RESAMPLES = 1000
METHODS: tuple[tuple[str, Rule, bool], ...] = (
    ("point", "point", True),
    ("lower", "lower", True),
    ("bca", "bca", True),
    ("point-raw", "point", False),
)
"""(name, threshold rule, temperature-scaled confidences) for the split study."""

logger = log.get_logger("anvil.eval.calibration")

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _kappa(teacher: IntArray, student: IntArray, n_labels: int) -> float:
    idx = np.arange(teacher.size, dtype=np.int64)[None, :]
    return float(kappa_replicates(teacher, student, n_labels, idx)[0])


def _quantiles(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    p5, p50, p95 = np.percentile(a, [5, 50, 95])
    return {"mean": float(a.mean()), "p5": float(p5), "median": float(p50), "p95": float(p95)}


def curve_at(z: FloatArray, y: IntArray, temperature: float) -> list[dict[str, object]]:
    """The coverage-vs-parity curve at the `COVERAGE_GRID` levels: for each, the curve point
    with the largest coverage not above it (ties keep inputs together, so a level can fall
    between two points)."""
    conf = softmax(z, temperature).max(axis=1)
    points = coverage_curve(conf, y, z.argmax(axis=1).astype(np.int64), z.shape[1])
    rows: list[dict[str, object]] = []
    for level in COVERAGE_GRID:
        below = [p for p in points if p.coverage <= level + 1e-12]
        if below:
            rows.append({"level": level, **asdict(below[-1])})
    return rows


def split_study(
    z: FloatArray, y: IntArray, n_splits: int, n_resamples: int, seed: int
) -> dict[str, object]:
    """Fit on a random half, score on the other, `n_splits` times."""
    n, k = z.shape
    student = z.argmax(axis=1).astype(np.int64)
    rng = np.random.default_rng(seed)
    temps: list[float] = []
    ece_raw: list[float] = []
    ece_cal: list[float] = []
    nll_raw: list[float] = []
    nll_cal: list[float] = []
    outcomes: dict[str, dict[str, list[dict[str, float | bool]]]] = {
        str(t): {name: [] for name, _, _ in METHODS} for t in TARGETS
    }
    for split in range(n_splits):
        order = rng.permutation(n)
        fit, test = order[: n // 2], order[n // 2 :]
        temperature, _ = serving_temperature(z[fit], y[fit])
        temps.append(temperature)
        ece_raw.append(ece(softmax(z[test]), y[test], N_BINS))
        ece_cal.append(ece(softmax(z[test], temperature), y[test], N_BINS))
        nll_raw.append(nll(z[test], y[test]))
        nll_cal.append(nll(z[test], y[test], temperature))
        for name, rule, scaled in METHODS:
            t_used = temperature if scaled else 1.0
            conf_fit = softmax(z[fit], t_used).max(axis=1)
            conf_test = softmax(z[test], t_used).max(axis=1)
            for target in TARGETS:
                choice = choose_threshold(
                    conf_fit,
                    y[fit],
                    student[fit],
                    k,
                    target,
                    rule=rule,
                    min_accepted=MIN_ACCEPTED,
                    n_resamples=n_resamples,
                    seed=seed + split,
                )
                row: dict[str, float | bool] = {"found": choice is not None}
                if choice is not None:
                    accepted = conf_test >= choice.threshold
                    row["fit_coverage"] = choice.coverage
                    row["coverage"] = float(accepted.mean())
                    if accepted.any():
                        t_acc, s_acc = y[test][accepted], student[test][accepted]
                        row["kappa"] = _kappa(t_acc, s_acc, k)
                        row["met"] = row["kappa"] >= target
                    cascade = np.where(accepted, student[test], y[test])
                    row["system_kappa"] = _kappa(y[test], cascade, k)
                outcomes[str(target)][name].append(row)
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for key, by_method in outcomes.items():
        summary[key] = {}
        for name, rows in by_method.items():
            found = [r for r in rows if r["found"]]
            scored = [r for r in found if "kappa" in r]
            entry: dict[str, object] = {
                "found": len(found) / len(rows),
                # Over every split where a threshold was found: one that accepts nothing on the
                # unseen half has not kept the target on any answered input, so it is not met.
                "met": (sum(bool(r.get("met")) for r in found) / len(found)) if found else None,
            }
            if scored:
                entry["coverage"] = _quantiles([float(r["coverage"]) for r in found])
                entry["fit_coverage"] = _quantiles([float(r["fit_coverage"]) for r in found])
                entry["kappa"] = _quantiles([float(r["kappa"]) for r in scored])
                entry["system_kappa"] = _quantiles([float(r["system_kappa"]) for r in found])
            summary[key][name] = entry
    return {
        "splits": n_splits,
        "fit_size": n // 2,
        "test_size": n - n // 2,
        "seed": seed,
        "n_resamples": n_resamples,
        "min_accepted": MIN_ACCEPTED,
        "temperature": _quantiles(temps),
        "ece_before": _quantiles(ece_raw),
        "ece_after": _quantiles(ece_cal),
        "nll_before": _quantiles(nll_raw),
        "nll_after": _quantiles(nll_cal),
        "targets": summary,
    }


MIN_CEILING_ITEMS = 20
"""Ceiling items a coverage level needs before its self-agreement is reported."""


def _kappa_ci(
    a: IntArray, b: IntArray, n_labels: int, n_resamples: int, seed: int
) -> dict[str, float]:
    idx = resample_indices(a.size, n_resamples, seed)
    reps = kappa_replicates(a, b, n_labels, idx)
    low, high = np.percentile(reps, [2.5, 97.5])
    return {"kappa": _kappa(a, b, n_labels), "low": float(low), "high": float(high)}


def ceiling_on_accepted(
    z: FloatArray,
    y: IntArray,
    repeat: IntArray,
    temperature: float,
    thresholds: dict[str, float | None],
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """The teacher's self-agreement (the parity ceiling) on the inputs a threshold accepts.

    `repeat` holds the teacher's label for each held-out input under the paraphrased prompt
    (-1 where it was not asked twice). The ceiling is measured on every input, the cascade
    serves only the confident ones, and the teacher may well be more consistent on those: a
    target the whole-split ceiling rules out can still be reachable there. For each coverage
    level of the curve and each served threshold, the ceiling items above it give the
    teacher's self-agreement and the student's agreement with the teacher on the same items.
    """
    k = z.shape[1]
    student = z.argmax(axis=1).astype(np.int64)
    conf = softmax(z, temperature).max(axis=1)
    asked = repeat >= 0
    ordered = np.sort(conf)[::-1]

    def row(accepted: npt.NDArray[np.bool_]) -> dict[str, object]:
        both = accepted & asked
        out: dict[str, object] = {"coverage": float(accepted.mean()), "n": int(both.sum())}
        if both.sum() >= MIN_CEILING_ITEMS:
            out["ceiling"] = _kappa_ci(y[both], repeat[both], k, n_resamples, seed)
            out["student_vs_teacher"] = _kappa_ci(y[both], student[both], k, n_resamples, seed)
        return out

    by_level = []
    for level in COVERAGE_GRID:
        m = max(1, round(level * conf.size))
        by_level.append({"level": level, **row(conf >= ordered[m - 1])})
    served: dict[str, object] = {}
    for target, threshold in thresholds.items():
        served[target] = row(conf >= threshold) if threshold is not None else None
    return {"asked": int(asked.sum()), "by_level": by_level, "served": served}


def analyse(
    z: FloatArray,
    y: IntArray,
    n_splits: int,
    n_resamples: int,
    seed: int,
    repeat: IntArray | None = None,
) -> dict[str, object]:
    """Every number of the report from held-out logits and teacher label codes (and, when
    the teacher labelled some inputs twice, the second labels as `repeat`)."""
    full = {
        str(t): calibrate(
            z,
            y,
            target_kappa=t,
            rule=SERVED_RULE,
            n_bins=N_BINS,
            min_accepted=MIN_ACCEPTED,
            n_resamples=n_resamples,
            seed=seed,
        ).as_dict()
        for t in TARGETS
    }
    by_rule: dict[str, dict[str, dict[str, object]]] = {}
    for rule in RULES:
        by_rule[rule] = {}
        for target in TARGETS:
            chosen = calibrate(
                z,
                y,
                target_kappa=target,
                rule=rule,
                n_bins=N_BINS,
                min_accepted=MIN_ACCEPTED,
                n_resamples=n_resamples,
                seed=seed,
            ).as_dict()
            by_rule[rule][str(target)] = {
                key: chosen[key] for key in ("threshold", "coverage", "kappa", "kappa_low")
            }
    temperature = calibrate(z, y, target_kappa=None).temperature
    extra: dict[str, object] = {}
    if repeat is not None and (repeat >= 0).any():
        thresholds: dict[str, float | None] = {}
        for t, cal in full.items():
            value = cal.get("threshold")
            thresholds[t] = float(value) if isinstance(value, int | float) else None
        extra["ceiling"] = ceiling_on_accepted(
            z, y, repeat, temperature, thresholds, n_resamples, seed
        )
    return {
        "n": int(z.shape[0]),
        "labels": int(z.shape[1]),
        "kappa": _kappa(y, z.argmax(axis=1).astype(np.int64), z.shape[1]),
        "full": full,
        "by_rule": by_rule,
        "reliability": {
            "before": [asdict(b) for b in reliability(softmax(z), y, N_BINS)],
            "after": [asdict(b) for b in reliability(softmax(z, temperature), y, N_BINS)],
        },
        "curve": curve_at(z, y, temperature),
        "split_study": split_study(z, y, n_splits, n_resamples, seed),
        **extra,
    }


@dataclass(frozen=True)
class Scored:
    """The eval student's logits on its held-out inputs."""

    trained: TrainedStudent
    logits: npt.NDArray[np.float32]
    teacher: IntArray
    """Teacher label codes, indices into `labels`."""
    input_ids: list[str]
    labels: list[str]
    seconds: float
    repeat: IntArray | None = None
    """The teacher's paraphrase-prompt label code per held-out input (-1 if not asked)."""


def paraphrase_codes(ts: TrainedStudent, index: dict[str, int]) -> IntArray | None:
    """The teacher's labels under the paraphrased prompt for the held-out inputs, as codes:
    from the committed label file when there is one, else from the workspace store. None for
    the gold teacher, which is never asked twice."""
    if ts.kind == "gold":
        return None
    para = ts.config.paraphrase_spec()
    path = label_file(ts.config.dataset.name, para.teacher.model, para.prompt_hash)
    if path.exists():
        second = {r.input_id: r.label for r in read_label_file(path) if r.label is not None}
    else:
        with ts.workspace.open_store(para) as store:
            second = store.labels(ts.teacher_model, para.prompt_hash)
    codes = [index.get(second.get(it.id, ""), -1) for it, _ in ts.heldout]
    return np.array(codes, dtype=np.int64) if any(c >= 0 for c in codes) else None


def held_out_logits(name: str, teacher: TeacherKind, settings: Settings) -> Scored:
    """Score the eval student (trained now if the workspace lacks it) on its held-out inputs."""
    from anvil.students.encoder import EncoderClassifier

    ts = trained_student(name, teacher, settings)
    student = EncoderClassifier.load(ts.path)
    index = {label: i for i, label in enumerate(student.labels)}
    started = time.perf_counter()
    logits = student.predict_logits([it.text for it, _ in ts.heldout])
    return Scored(
        trained=ts,
        logits=logits.astype(np.float32),
        teacher=np.array([index[lab] for _, lab in ts.heldout], dtype=np.int64),
        input_ids=[it.id for it, _ in ts.heldout],
        labels=student.labels,
        seconds=time.perf_counter() - started,
        repeat=paraphrase_codes(ts, index),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.calibration", description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument(
        "--teacher", choices=["auto", "claude", "ollama", "replay", "gold"], default="auto"
    )
    parser.add_argument("--splits", type=int, default=DEFAULT_SPLITS)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--logits",
        type=Path,
        help="re-run the analysis from a committed logits.npz (one dataset; no student needed)",
    )
    args = parser.parse_args(argv)
    if args.logits is not None and len(args.datasets) != 1:
        parser.error("--logits takes exactly one dataset")
    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.calibration " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    smoke = args.splits != DEFAULT_SPLITS or args.resamples != DEFAULT_RESAMPLES or args.seed
    try:
        for name in args.datasets:
            started = time.perf_counter()
            meta_extra: dict[str, object] = {}
            scored: Scored | None = None
            if args.logits is not None:
                saved = np.load(args.logits)
                z = saved["logits"].astype(np.float64)
                y = saved["teacher"].astype(np.int64)
                repeat = saved["repeat"].astype(np.int64) if "repeat" in saved else None
                teacher_model = str(saved["teacher_model"])
                previous = args.logits.parent / "report.json"
                if not previous.exists():
                    raise ConfigError(f"{previous} not found next to {args.logits}")
                source = json.loads(previous.read_text(encoding="utf-8"))["source"]
                if source.get("dataset") != name:
                    raise ConfigError(
                        f"{args.logits} holds {source.get('dataset')!r} logits, not {name!r}"
                    )
                meta_extra["from_logits"] = str(args.logits)
            else:
                scored = held_out_logits(name, args.teacher, settings)
                ts = scored.trained
                z = scored.logits.astype(np.float64)
                y = scored.teacher
                repeat = scored.repeat
                teacher_model = ts.teacher_model
                source = {
                    "dataset": name,
                    "sample_sha256": ts.data.sha256,
                    "teacher_kind": ts.kind,
                    "teacher_model": teacher_model,
                    "student_model": ts.spec.student.model,
                    "student_run": ts.path.name,
                    "trained_here": ts.trained_here,
                    "scoring_seconds": round(scored.seconds, 1),
                }
            analysis = analyse(z, y, args.splits, args.resamples, args.seed, repeat)
            report = {"source": source, **analysis}
            full = analysis["full"]
            assert isinstance(full, dict)
            served: dict[str, object] = full[str(SERVED_TARGET)]
            out = RESULTS_DIR / (
                f"calibration-{name}-{file_safe(teacher_model)}" + (SMOKE_SUFFIX if smoke else "")
            )
            out.mkdir(parents=True, exist_ok=True)
            if scored is not None:
                arrays = {
                    "logits": scored.logits,
                    "teacher": y,
                    "input_ids": np.array(scored.input_ids),
                    "labels": np.array(scored.labels),
                    "teacher_model": np.array(teacher_model),
                }
                if scored.repeat is not None:
                    arrays["repeat"] = scored.repeat
                np.savez_compressed(out / "logits.npz", **arrays)
                if not smoke:  # a smoke run must not change what export puts in the bundle
                    write_calibration(scored.trained.path, served, teacher_model)
            wall = time.perf_counter() - started
            meta = {
                "command": command,
                **provenance,
                "anvil": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpus": os.cpu_count(),
                "seed": args.seed,
                "splits": args.splits,
                "n_resamples": args.resamples,
                "teacher_model": teacher_model,
                "wall_seconds": round(wall, 1),
                "spent_usd": scored.trained.spent_usd if scored is not None else 0.0,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                **meta_extra,
            }
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
            print(  # noqa: T201 - the final summary
                f"{name}: T {served['temperature']:.3f}, ECE {served['ece_before']:.3f}"
                f" -> {served['ece_after']:.3f}; at kappa {SERVED_TARGET} ({SERVED_RULE})"
                f" coverage {served['coverage']}; raw output {out}"
            )
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
