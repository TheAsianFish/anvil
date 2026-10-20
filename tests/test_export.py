from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnx
import pytest
import torch
from onnx import TensorProto, helper

import anvil.export as export_mod
from anvil.errors import ExportError
from anvil.export import (
    ExportReport,
    _fp32_gate,
    _int8_gate,
    answered_by_either,
    export_student,
    mlp_down_projection_nodes,
    strip_initializer_value_info,
)
from anvil.runtime import BUNDLE_META, OnnxStudent
from anvil.spec import StudentSpec
from anvil.students.encoder import EncoderClassifier
from anvil.students.tagger import EncoderTagger

LABELS = ["billing", "bug", "feature"]
CORPUS = [
    "refund my invoice please",
    "the app crashed again with an error",
    "please add dark mode",
    "my login does not work today",
    "café crème brûlée",
    "ok",
]
# Mixed lengths, an empty string, an unknown word and non-ASCII text, so padding,
# truncation and normalisation are all on the parity path.
PROBES = [
    *CORPUS,
    "",
    "zebra",
    " ".join(["please help"] * 30),  # longer than max_length 32: truncated
    "ÉRROR!!! the APP crashed… 日本語",
]
TEACHER = [
    "billing",
    "bug",
    "feature",
    "bug",
    "feature",
    "billing",
    "bug",
    "bug",
    "billing",
    "bug",
]


@dataclass(frozen=True)
class Exported:
    student_dir: Path
    bundle: Path
    report: ExportReport


_CACHE: dict[int, Exported] = {}


def exported(factory: pytest.TempPathFactory, max_length: int = 32) -> Exported:
    """One tiny random-init student per max_length, exported once per session (with int8)
    and shared with tests/test_runtime.py; export takes a few seconds."""
    if max_length not in _CACHE:
        root = factory.mktemp(f"export-{max_length}")
        spec = StudentSpec(model="tiny", max_length=max_length, seed=0)
        student = EncoderClassifier.build(spec, LABELS, CORPUS)
        student.meta.update({"task": "toy", "train_labels": ["bug"] * 3})
        student.save(root / "student")
        long_text = " ".join(CORPUS * 40)  # ~700 tokens: beyond the 8-token trace and 512
        probes = [*PROBES, long_text] if max_length > 512 else PROBES
        report = export_student(
            root / "student",
            root / "bundle",
            probes,
            int8=max_length <= 512,
            teacher_labels=TEACHER if max_length <= 512 else None,
            calibration={"temperature": 1.5, "threshold": 0.4},
            spec={"name": "toy", "kind": "classify"},
        )
        _CACHE[max_length] = Exported(root / "student", root / "bundle", report)
    return _CACHE[max_length]


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Exported:
    return exported(tmp_path_factory)


@torch.no_grad()
def torch_logits(student: EncoderClassifier, texts: Sequence[str]) -> npt.NDArray[np.float32]:
    out: npt.NDArray[np.float32] = student.model(**student.encode(texts)).logits.numpy()
    return out


def test_fp32_bundle_layout_and_meta(bundle: Exported) -> None:
    files = {p.name for p in bundle.bundle.iterdir()}
    assert {"model.onnx", "tokenizer.json", BUNDLE_META} <= files
    meta = json.loads((bundle.bundle / BUNDLE_META).read_text(encoding="utf-8"))
    assert meta["format"] == 1
    assert meta["labels"] == LABELS
    assert meta["max_length"] == 32
    assert meta["temperature"] == 1.5
    assert meta["threshold"] == 0.4
    assert meta["spec"] == {"name": "toy", "kind": "classify"}
    assert meta["student"] == {"task": "toy"}  # train_labels is dropped
    assert meta["variants"]["fp32"]["passed"] is True
    assert meta["variants"]["fp32"]["parity"]["max_abs_diff"] == bundle.report.fp32.max_abs_diff
    # No temporary export directory is left next to the bundle.
    assert {p.name for p in bundle.bundle.parent.iterdir()} == {"student", "bundle"}


def test_fp32_parity_on_padded_batch_and_single_inputs(bundle: Exported) -> None:
    fp32 = bundle.report.fp32
    assert fp32.passed
    assert fp32.reference == "pytorch"
    assert fp32.n == len(PROBES)
    assert fp32.max_abs_diff <= 1e-4
    assert fp32.agreement == 1.0
    assert fp32.kappa is not None
    assert fp32.kappa_drop == 0.0

    student = EncoderClassifier.load(bundle.student_dir)
    onnx_student = OnnxStudent.load(bundle.bundle)
    batch = onnx_student.logits(PROBES)
    np.testing.assert_allclose(batch, torch_logits(student, PROBES), atol=1e-4, rtol=0)
    for text in PROBES:
        np.testing.assert_allclose(
            onnx_student.logits([text]), torch_logits(student, [text]), atol=1e-4, rtol=0
        )
    # Batches of 3 cut the probes differently from the padded batch above.
    np.testing.assert_allclose(onnx_student.logits(PROBES, batch_size=3), batch, atol=1e-5)


def test_sequences_longer_than_512_tokens_keep_parity(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    long = exported(tmp_path_factory, max_length=1024)
    assert long.report.fp32.passed
    assert long.report.int8 is None
    assert long.report.fp32.max_abs_diff <= 1e-4
    student = EncoderClassifier.load(long.student_dir)
    onnx_student = OnnxStudent.load(long.bundle)
    text = " ".join(CORPUS * 40)
    ids, _ = onnx_student.encode([text])
    assert 512 < ids.shape[1] <= 1024
    np.testing.assert_allclose(
        onnx_student.logits([text, "ok"]), torch_logits(student, [text, "ok"]), atol=1e-4, rtol=0
    )


def test_int8_export_reports_its_gate(bundle: Exported) -> None:
    q8 = bundle.report.int8
    assert q8 is not None
    assert q8.reference == "onnx-fp32"
    assert 0.0 <= q8.agreement <= 1.0
    assert q8.kappa is not None
    assert q8.reference_kappa is not None
    assert q8.kappa_drop is not None
    assert bundle.report.excluded_nodes == 2  # one mlp.Wo per layer
    meta = json.loads((bundle.bundle / BUNDLE_META).read_text(encoding="utf-8"))
    info = meta["variants"]["int8"]
    assert info["passed"] is q8.passed
    assert info["parity"]["agreement"] == q8.agreement
    assert (bundle.bundle / "model.int8.onnx").exists() is q8.passed
    # The bundle has a threshold (0.4), so agreement is also measured on the answered probes.
    assert q8.answered is not None
    assert 0 <= q8.answered <= q8.n
    assert q8.agreement_answered is not None
    assert "answers at threshold 0.4000" in q8.gate


def test_int8_that_fails_its_gate_is_recorded_and_not_shipped(
    tmp_path: Path, bundle: Exported, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(export_mod, "INT8_MIN_AGREEMENT", 1.01)
    report = export_student(bundle.student_dir, tmp_path / "b", PROBES[:4], int8=True)
    assert report.fp32.passed
    assert report.int8 is not None
    assert not report.int8.passed
    assert not (tmp_path / "b" / "model.int8.onnx").exists()
    meta = json.loads((tmp_path / "b" / BUNDLE_META).read_text(encoding="utf-8"))
    assert meta["variants"]["int8"]["passed"] is False
    assert meta["variants"]["int8"]["parity"]["agreement"] == report.int8.agreement
    assert meta["temperature"] == 1.0
    assert meta["threshold"] is None


def test_gates() -> None:
    assert _fp32_gate(1e-4, 1.0)[0]
    assert not _fp32_gate(1.1e-4, 1.0)[0]
    assert not _fp32_gate(0.0, 0.999)[0]
    assert "1e-04" in _fp32_gate(0.0, 1.0)[1]
    assert _int8_gate(0.98, None) == (True, "agreement with fp32 onnx on all probes >= 0.98")
    assert not _int8_gate(0.979, None)[0]
    assert _int8_gate(0.99, 0.01)[0]
    assert not _int8_gate(0.99, 0.011)[0]
    assert not _int8_gate(0.97, -0.05)[0]
    assert "kappa drop" in _int8_gate(1.0, 0.0)[1]


def test_answered_counts_inputs_either_graph_would_answer() -> None:
    # Row 0: both confident. Row 1: only fp32 confident (int8 would escalate it). Row 2: only
    # int8 confident, on a flipped argmax: served int8 answers it, so the gate must see it.
    # Row 3: both escalate, so a flip there never reaches a caller.
    fp32 = np.array([[4.0, 0.0], [4.0, 0.0], [0.2, 0.0], [0.1, 0.0]], dtype=np.float32)
    int8 = np.array([[4.0, 0.0], [0.2, 0.0], [0.0, 4.0], [0.0, 0.1]], dtype=np.float32)
    got = answered_by_either(fp32, int8, temperature=1.0, cut=0.9)
    assert got.tolist() == [True, True, True, False]
    # A higher temperature flattens both and moves the cut: nothing reaches 0.9 at T = 10.
    assert not answered_by_either(fp32, int8, temperature=10.0, cut=0.9).any()


def test_failed_fp32_parity_raises_and_writes_nothing(
    tmp_path: Path, bundle: Exported, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = export_mod._torch_logits

    def perturbed(student: EncoderClassifier, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        return real(student, texts) + np.float32(1e-3)

    monkeypatch.setattr(export_mod, "_torch_logits", perturbed)
    out = tmp_path / "out" / "bundle"
    with pytest.raises(ExportError, match=r"failed parity: max \|logit diff\| 0\.001"):
        export_student(bundle.student_dir, out, PROBES[:3])
    assert list((tmp_path / "out").iterdir()) == []


def test_export_rejects_bad_requests(tmp_path: Path, bundle: Exported) -> None:
    with pytest.raises(ExportError, match="probe texts"):
        export_student(bundle.student_dir, tmp_path / "b", [])
    with pytest.raises(ExportError, match="teacher labels"):
        export_student(bundle.student_dir, tmp_path / "b", ["a"], teacher_labels=["bug", "bug"])
    (tmp_path / "taken").mkdir()
    (tmp_path / "taken" / "notes.txt").write_text("mine", encoding="utf-8")
    with pytest.raises(ExportError, match="not an Anvil bundle"):
        export_student(bundle.student_dir, tmp_path / "taken", ["a"])
    with pytest.raises(ExportError, match="not a saved Anvil student"):
        export_student(tmp_path, tmp_path / "b", ["a"])


def test_export_refuses_extraction_students(tmp_path: Path) -> None:
    tagger = EncoderTagger.build(StudentSpec(model="tiny", max_length=16), ["dish"], CORPUS)
    tagger.save(tmp_path / "tagger")
    with pytest.raises(ExportError, match="supports classify students"):
        export_student(tmp_path / "tagger", tmp_path / "b", ["a"])


def test_mlp_down_projection_nodes_finds_every_wo(bundle: Exported) -> None:
    model = onnx.load(str(bundle.bundle / "model.onnx"))
    config = EncoderClassifier.load(bundle.student_dir).model.config
    nodes = mlp_down_projection_nodes(model, config)
    assert len(nodes) == config.num_hidden_layers == 2
    shapes = {i.name: list(i.dims) for i in model.graph.initializer}
    for node in model.graph.node:
        if node.name in nodes:
            assert node.op_type == "MatMul"
            assert [128, 64] in [shapes.get(name) for name in node.input]


def test_strip_initializer_value_info() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = helper.make_tensor("w", TensorProto.FLOAT, [2, 3], [0.0] * 6)
    b = helper.make_tensor("b", TensorProto.FLOAT, [3], [0.0] * 3)
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", "w"], ["h"]),
            helper.make_node("Add", ["h", "b"], ["y"]),
        ],
        "g",
        [x],
        [y],
        initializer=[w, b],
        value_info=[
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [3, 2]),  # stale shape
            helper.make_tensor_value_info("h", TensorProto.FLOAT, [1, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [3]),
        ],
    )
    model = helper.make_model(graph)
    assert strip_initializer_value_info(model) == 2
    assert [vi.name for vi in model.graph.value_info] == ["h"]
    assert strip_initializer_value_info(model) == 0
    onnx.checker.check_model(model)
