"""Export a trained classify student to an ONNX bundle and gate it on measured parity.

Decisions from docs/notes/onnx-export-spike.md: `torch.onnx.export(dynamo=True)` with dynamic
batch and sequence axes; fp32 must match PyTorch to 1e-4 with identical argmax on the probe
texts or nothing is written; int8 is opt-in, per-tensor dynamic quantization with the MLP
down-projection MatMuls kept in fp32, gated on argmax agreement with the fp32 graph on the
inputs the cascade answers and on the kappa drop (a logit tolerance does not work for int8).
The bundle layout is described in `anvil.runtime`.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
import warnings
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import torch

from anvil.calibrate import THRESHOLD_TOLERANCE, softmax
from anvil.errors import ExportError, StudentError
from anvil.evaluate.metrics import cohen_kappa
from anvil.log import get_logger
from anvil.runtime import (
    BUNDLE_FORMAT,
    BUNDLE_META,
    TOKENIZER_FILE,
    VARIANT_FILES,
    OnnxStudent,
)
from anvil.students.encoder import META_FILE, EncoderClassifier
from anvil.students.tagger import META_FILE as TAGGER_META_FILE

FP32_MAX_ABS_DIFF = 1e-4  # spike measured 8e-06 on ModernBERT-base
INT8_MIN_AGREEMENT = 0.98  # argmax agreement with the fp32 ONNX model
INT8_MAX_KAPPA_DROP = 0.01  # kappa vs teacher, fp32 ONNX minus int8, when teacher labels are given
# Probe texts also run one at a time (up to this many), so a graph that only works at the
# traced batch size, or only padded, fails the gate.
SINGLE_PROBES = 16
BATCH_SIZE = 32

LOG = get_logger(__name__)
FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class Parity:
    """How one exported variant compares with its reference on the probe texts."""

    variant: str
    n: int  # probe texts
    reference: str  # "pytorch" for fp32, "onnx-fp32" for int8
    max_abs_diff: float
    mean_abs_diff: float
    agreement: float  # argmax agreement with the reference
    kappa: float | None  # this variant vs the teacher labels, when given
    reference_kappa: float | None  # the reference vs the teacher labels
    kappa_drop: float | None  # reference_kappa - kappa
    passed: bool
    gate: str
    size_mb: float
    seconds: float
    answered: int | None = None  # int8: probes fp32 or int8 answers at the bundle threshold
    agreement_answered: float | None = None  # int8: argmax agreement on those probes


@dataclass(frozen=True)
class ExportReport:
    bundle: Path
    fp32: Parity
    int8: Parity | None
    excluded_nodes: int  # MatMuls kept fp32 in the int8 graph


class _LogitsOnly(torch.nn.Module):
    """The HF model with two inputs and one output, so the graph has nothing else."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits
        return logits


def strip_initializer_value_info(model: onnx.ModelProto) -> int:
    """Drop value_info entries that duplicate initializers; returns how many were removed.

    The dynamo exporter writes them. The quantizer's Gemm -> MatMul rewrite transposes the
    classifier weight but leaves its value_info at the old shape, so its next shape inference
    fails ("Inferred shape and existing shape differ").
    """
    inits = {i.name for i in model.graph.initializer}
    keep = [vi for vi in model.graph.value_info if vi.name not in inits]
    removed = len(model.graph.value_info) - len(keep)
    del model.graph.value_info[:]
    model.graph.value_info.extend(keep)
    return removed


def mlp_down_projection_nodes(
    model: onnx.ModelProto,
    config: Any,  # Any: a transformers PretrainedConfig (untyped)
) -> list[str]:
    """MatMuls whose weight initializer is [intermediate_size, hidden_size]: ModernBERT's
    `mlp.Wo`, which reads the GeGLU output and loses most accuracy under int8 (spike).

    `config` is the student's HF config. Matched by weight shape, not node name, so it does
    not depend on the exporter's naming scheme.
    """
    shape = [int(config.intermediate_size), int(config.hidden_size)]
    down = {i.name for i in model.graph.initializer if list(i.dims) == shape}
    return [n.name for n in model.graph.node if n.op_type == "MatMul" and down & set(n.input)]


def embedding_nodes(model: onnx.ModelProto) -> list[str]:
    """Gathers that read a weight initializer: the token embedding lookup. `quantize_dynamic`
    quantizes Gather by default, so excluding the linear layers alone does not keep a model
    fp32."""
    inits = {i.name for i in model.graph.initializer}
    return [n.name for n in model.graph.node if n.op_type == "Gather" and n.input[0] in inits]


def answered_by_either(
    fp32_logits: FloatArray, int8_logits: FloatArray, temperature: float, cut: float
) -> npt.NDArray[np.bool_]:
    """Inputs whose calibrated confidence reaches `cut` under the fp32 or the int8 graph."""
    fp32_conf = softmax(fp32_logits, temperature).max(axis=1)
    int8_conf = softmax(int8_logits, temperature).max(axis=1)
    answered: npt.NDArray[np.bool_] = (fp32_conf >= cut) | (int8_conf >= cut)
    return answered


def _fp32_gate(max_abs_diff: float, agreement: float) -> tuple[bool, str]:
    rule = f"max |logit diff| <= {FP32_MAX_ABS_DIFF:.0e} and agreement == 1.0"
    return max_abs_diff <= FP32_MAX_ABS_DIFF and agreement == 1.0, rule


def _int8_gate(
    agreement: float, kappa_drop: float | None, scope: str = "all probes"
) -> tuple[bool, str]:
    """`agreement` is measured on the probes the cascade answers under fp32 or int8 (all of
    them without a threshold): an int8 flip on an input that both graphs escalate to the
    teacher never reaches a caller.
    On the Banking77 student every int8 flip was on an escalated input (RESULTS.md)."""
    rule = f"agreement with fp32 onnx on {scope} >= {INT8_MIN_AGREEMENT}"
    passed = agreement >= INT8_MIN_AGREEMENT
    if kappa_drop is not None:
        rule += f" and kappa drop vs teacher <= {INT8_MAX_KAPPA_DROP}"
        passed = passed and kappa_drop <= INT8_MAX_KAPPA_DROP
    return passed, rule


def _load_classifier(student_dir: Path) -> EncoderClassifier:
    meta_path = student_dir / META_FILE
    if (student_dir / TAGGER_META_FILE).exists() or (
        meta_path.exists() and "labels" not in json.loads(meta_path.read_text(encoding="utf-8"))
    ):
        raise ExportError(
            "export supports classify students; extraction serving is not in this release"
        )
    try:
        student = EncoderClassifier.load(student_dir)
    except StudentError as exc:
        raise ExportError(f"cannot export {student_dir}: {exc}") from exc
    if not (student_dir / TOKENIZER_FILE).is_file():
        raise ExportError(
            f"{student_dir} has no {TOKENIZER_FILE}; export needs a fast (Rust) tokenizer."
            " Retrain the student with `anvil train` using a model that ships one"
        )
    student.model.to("cpu").eval()
    return student


@torch.no_grad()
def _torch_logits(student: EncoderClassifier, texts: Sequence[str]) -> FloatArray:
    out = [
        student.model(**student.encode(texts[i : i + BATCH_SIZE])).logits.float().numpy()
        for i in range(0, len(texts), BATCH_SIZE)
    ]
    return np.concatenate(out).astype(np.float32)


def _probe_logits(texts: Sequence[str], run: Callable[[Sequence[str]], FloatArray]) -> FloatArray:
    """Batched logits for every probe, then each of the first SINGLE_PROBES alone."""
    singles = [run([t]) for t in texts[:SINGLE_PROBES]]
    return np.concatenate([run(texts), *singles]).astype(np.float32)


def _export_graph(student: EncoderClassifier, path: Path, opset: int | None) -> None:
    # Trace with batch 2 and a length above 2 so neither axis is specialised to a constant.
    enc = student.tokenizer(
        ["anvil export trace", "x"], padding="max_length", max_length=8, return_tensors="pt"
    )
    example = (enc["input_ids"].to(torch.int64), enc["attention_mask"].to(torch.int64))
    batch = torch.export.Dim("batch", min=1, max=1024)
    seq = torch.export.Dim("sequence", min=2, max=8192)
    kwargs: dict[str, Any] = {}  # Any: torch.onnx.export keyword arguments are heterogeneous
    if opset is not None:
        kwargs["opset_version"] = opset
    # The exporter logs plain-text warnings (optional torchvision ops it skips) through the
    # `torch.onnx` logger; they say nothing about this model, so they are muted for the call.
    onnx_log = logging.getLogger("torch.onnx")
    level = onnx_log.level
    onnx_log.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            torch.onnx.export(
                _LogitsOnly(student.model).eval(),
                example,
                str(path),
                dynamo=True,
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_shapes={
                    "input_ids": {0: batch, 1: seq},
                    "attention_mask": {0: batch, 1: seq},
                },
                verbose=False,
                **kwargs,
            )
    finally:
        onnx_log.setLevel(level)


def _calib_number(calib: dict[str, object], key: str) -> float | None:
    value = calib.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExportError(f"calibration {key} is {value!r}, expected a number; recalibrate")
    return float(value)


def _size_mb(*paths: Path) -> float:
    return round(sum(p.stat().st_size for p in paths if p.exists()) / 1e6, 2)


def _kappas(
    labels: list[str], pred: FloatArray, ref: FloatArray, teacher: Sequence[str] | None
) -> tuple[float | None, float | None, float | None]:
    if teacher is None:
        return None, None, None
    n = len(teacher)
    kappa = cohen_kappa([labels[i] for i in pred[:n].argmax(axis=1)], teacher)
    ref_kappa = cohen_kappa([labels[i] for i in ref[:n].argmax(axis=1)], teacher)
    return round(kappa, 6), round(ref_kappa, 6), round(ref_kappa - kappa, 6)


def _compare(got: FloatArray, ref: FloatArray) -> tuple[float, float, float]:
    diff = np.abs(got.astype(np.float64) - ref.astype(np.float64))
    agreement = float(np.mean(got.argmax(axis=1) == ref.argmax(axis=1)))
    return float(diff.max()), float(diff.mean()), agreement


def _log(p: Parity) -> None:
    LOG.info("export variant", extra={"parity": asdict(p)})


def export_student(
    student_dir: Path,
    out: Path,
    probe_texts: Sequence[str],
    *,
    int8: bool = False,
    teacher_labels: Sequence[str] | None = None,
    calibration: dict[str, object] | None = None,
    spec: dict[str, object] | None = None,
    opset: int | None = None,
) -> ExportReport:
    """Export the classifier saved in `student_dir` to the bundle directory `out`.

    fp32 parity is measured on `probe_texts` (PyTorch through the HF tokenizer vs ONNX through
    the runtime tokenizer, so tokenization differences count too). If it fails, ExportError is
    raised and `out` is left untouched. An int8 variant that fails its gate is recorded in the
    meta with `passed: false` and its file is removed; the caller decides what that means.
    """
    texts = list(probe_texts)
    if not texts:
        raise ExportError("export needs probe texts to measure parity; pass at least one input")
    if teacher_labels is not None and len(teacher_labels) != len(texts):
        raise ExportError(
            f"{len(teacher_labels)} teacher labels for {len(texts)} probe texts; pass one per text"
        )
    if out.exists() and (
        not out.is_dir() or (any(out.iterdir()) and not (out / BUNDLE_META).exists())
    ):
        raise ExportError(f"{out} exists and is not an Anvil bundle; choose another --out")
    student = _load_classifier(student_dir)
    calib = calibration or {}
    temperature = _calib_number(calib, "temperature")
    threshold = _calib_number(calib, "threshold")
    if temperature is not None and temperature <= 0:
        raise ExportError(f"calibration temperature must be > 0, got {temperature}; recalibrate")

    out.parent.mkdir(parents=True, exist_ok=True)
    # A plain mkdir, not tempfile.mkdtemp: mkdtemp makes the directory private (0700), and the
    # bundle is an artifact other users and containers read (`docker run -v bundle:/model`).
    tmp = out.parent / f".{out.name}.export-{uuid.uuid4().hex[:12]}"
    tmp.mkdir()
    try:
        meta: dict[str, object] = {
            "format": BUNDLE_FORMAT,
            "labels": student.labels,
            "max_length": student.max_length,
            "pad_token_id": int(student.tokenizer.pad_token_id),
            "padding_side": str(student.tokenizer.padding_side),
            "truncation_side": str(student.tokenizer.truncation_side),
            "temperature": 1.0 if temperature is None else temperature,
            "threshold": threshold,
            "calibration": calibration,
            # train_labels is one label per training example: too big for serving metadata.
            "student": {k: v for k, v in student.meta.items() if k != "train_labels"},
            "spec": spec,
        }
        shutil.copy2(student_dir / TOKENIZER_FILE, tmp / TOKENIZER_FILE)

        t0 = time.perf_counter()
        fp32_path = tmp / VARIANT_FILES["fp32"]
        try:
            _export_graph(student, fp32_path, opset)
        except Exception as exc:  # the exporter raises many unrelated exception types
            raise ExportError(
                f"torch.onnx.export failed for {student_dir}: {type(exc).__name__}: {exc}"
            ) from exc
        onnx_fp32 = OnnxStudent(tmp, fp32_path, meta, "fp32")
        ref = _probe_logits(texts, lambda b: _torch_logits(student, b))
        got = _probe_logits(texts, onnx_fp32.logits)
        max_diff, mean_diff, agreement = _compare(got, ref)
        passed, gate = _fp32_gate(max_diff, agreement)
        kappa, ref_kappa, drop = _kappas(student.labels, got, ref, teacher_labels)
        fp32 = Parity(
            variant="fp32", n=len(texts), reference="pytorch",
            max_abs_diff=max_diff, mean_abs_diff=mean_diff, agreement=agreement,
            kappa=kappa, reference_kappa=ref_kappa, kappa_drop=drop, passed=passed, gate=gate,
            size_mb=_size_mb(fp32_path, tmp / f"{fp32_path.name}.data"),
            seconds=round(time.perf_counter() - t0, 2),
        )  # fmt: skip
        _log(fp32)
        if not passed:
            raise ExportError(
                f"fp32 ONNX export of {student_dir} failed parity: max |logit diff| "
                f"{max_diff:.3g}, argmax agreement {agreement:.4f} (gate: {gate}); nothing was"
                " written. Try `--opset` or report the model"
            )
        variants: dict[str, object] = {
            "fp32": {"file": fp32_path.name, "parity": asdict(fp32), "passed": True}
        }

        q8: Parity | None = None
        excluded = 0
        if int8:
            q8, excluded = _export_int8(
                student, tmp, texts, got[: len(texts)], teacher_labels, meta
            )
            variants["int8"] = {
                "file": VARIANT_FILES["int8"],
                "parity": asdict(q8),
                "passed": q8.passed,
                "excluded_nodes": excluded,
            }
        meta["variants"] = variants
        meta["exported_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        (tmp / BUNDLE_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        # Swap by renames, so a crash never leaves `out` without a bundle in it for long and
        # never loses the old one: it is only deleted once the new one is in place.
        old = out.with_name(f".{out.name}.old")
        if old.exists():
            shutil.rmtree(old)
        if out.exists():
            out.rename(old)
        tmp.rename(out)
        if old.exists():
            shutil.rmtree(old, ignore_errors=True)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
    return ExportReport(bundle=out, fp32=fp32, int8=q8, excluded_nodes=excluded)


def _export_int8(
    student: EncoderClassifier,
    tmp: Path,
    texts: list[str],
    fp32_logits: FloatArray,
    teacher_labels: Sequence[str] | None,
    meta: dict[str, object],
) -> tuple[Parity, int]:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    t0 = time.perf_counter()
    model = onnx.load(str(tmp / VARIANT_FILES["fp32"]))
    strip_initializer_value_info(model)
    exclude = mlp_down_projection_nodes(model, student.model.config)
    q_path = tmp / VARIANT_FILES["int8"]
    try:
        quantize_dynamic(
            model,
            q_path,
            weight_type=QuantType.QInt8,
            per_channel=False,
            nodes_to_exclude=exclude,
        )
    except Exception as exc:  # the quantizer raises ORT and onnx exception types
        raise ExportError(f"int8 quantization failed: {type(exc).__name__}: {exc}") from exc
    got = OnnxStudent(tmp, q_path, meta, "int8").logits(texts)
    max_diff, mean_diff, agreement = _compare(got, fp32_logits)
    kappa, ref_kappa, drop = _kappas(student.labels, got, fp32_logits, teacher_labels)
    answered = np.ones(len(texts), dtype=bool)
    scope = "all probes"
    threshold = meta.get("threshold")
    if isinstance(threshold, int | float):
        temperature = float(str(meta.get("temperature", 1.0)))
        cut = threshold - THRESHOLD_TOLERANCE
        # A flip reaches a caller when either graph would answer the input itself: the
        # threshold was calibrated on fp32, but int8's own confidence decides when it serves.
        answered = answered_by_either(fp32_logits[: len(texts)], got, temperature, cut)
        scope = f"the probes fp32 or int8 answers at threshold {threshold:.4f}"
    n = len(texts)
    same = got[:n].argmax(axis=1) == fp32_logits[:n].argmax(axis=1)
    # With no answered probe there is nothing a caller could see differ.
    on_answered = float(same[answered].mean()) if answered.any() else 1.0
    passed, gate = _int8_gate(on_answered, drop, scope)
    size = _size_mb(q_path, tmp / f"{q_path.name}.data")
    if not passed:
        q_path.unlink()  # an int8 graph that failed its gate is never shipped
    parity = Parity(
        variant="int8", n=len(texts), reference="onnx-fp32",
        max_abs_diff=max_diff, mean_abs_diff=mean_diff, agreement=agreement,
        kappa=kappa, reference_kappa=ref_kappa, kappa_drop=drop, passed=passed, gate=gate,
        size_mb=size, seconds=round(time.perf_counter() - t0, 2),
        answered=int(answered.sum()), agreement_answered=on_answered,
    )  # fmt: skip
    _log(parity)
    return parity, len(exclude)
