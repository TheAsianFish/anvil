from __future__ import annotations

import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from anvil.errors import ExportError
from anvil.runtime import BUNDLE_META, OnnxStudent, softmax
from anvil.students.encoder import EncoderClassifier
from tests.test_export import PROBES, Exported, exported

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: pytest.TempPathFactory) -> Exported:
    return exported(tmp_path_factory)


@pytest.fixture
def edited(
    tmp_path: Path, bundle: Exported
) -> Callable[[Callable[[dict[str, object]], None]], Path]:
    """A copy of the shared bundle with its meta changed by `edit`."""

    def make(edit: Callable[[dict[str, object]], None]) -> Path:
        copy = tmp_path / "bundle"
        shutil.copytree(bundle.bundle, copy)
        meta = json.loads((copy / BUNDLE_META).read_text(encoding="utf-8"))
        edit(meta)
        (copy / BUNDLE_META).write_text(json.dumps(meta), encoding="utf-8")
        return copy

    return make


def test_runtime_tokenizer_matches_hf_tokenizer(bundle: Exported) -> None:
    hf = EncoderClassifier.load(bundle.student_dir)
    rt = OnnxStudent.load(bundle.bundle)
    cases = [
        PROBES,
        ["", "a much longer text " * 20, "ok"],
        ["naïve café 日本語 🎉 Ünïcödé"],
        [""],
        [" ".join(["refund"] * 100)],
    ]
    for texts in cases:
        ids, mask = rt.encode(texts)
        expected = hf.encode(texts)
        assert ids.dtype == mask.dtype == np.int64
        np.testing.assert_array_equal(ids, expected["input_ids"].numpy())
        np.testing.assert_array_equal(mask, expected["attention_mask"].numpy())
    assert rt.encode([" ".join(["refund"] * 100)])[0].shape == (1, 32)


def test_load_refuses_a_missing_bundle(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="anvil export"):
        OnnxStudent.load(tmp_path / "nope")


def test_load_refuses_a_missing_variant(
    edited: Callable[[Callable[[dict[str, object]], None]], Path],
) -> None:
    def drop_int8(meta: dict[str, object]) -> None:
        variants = meta["variants"]
        assert isinstance(variants, dict)
        variants.pop("int8")

    path = edited(drop_int8)
    with pytest.raises(ExportError, match=r"no int8 variant; run `anvil export --int8`"):
        OnnxStudent.load(path, "int8")
    (path / "model.onnx").unlink()
    with pytest.raises(ExportError, match="missing from the bundle"):
        OnnxStudent.load(path)


def test_parity_gate_refuses_a_failed_variant_unless_overridden(
    edited: Callable[[Callable[[dict[str, object]], None]], Path], bundle: Exported
) -> None:
    def fail_fp32(meta: dict[str, object]) -> None:
        variants = meta["variants"]
        assert isinstance(variants, dict)
        variants["fp32"]["passed"] = False

    path = edited(fail_fp32)
    with pytest.raises(ExportError, match="failed its parity gate"):
        OnnxStudent.load(path)
    student = OnnxStudent.load(path, require_parity=False, threads=1)
    assert student.variant == "fp32"
    expected = OnnxStudent.load(bundle.bundle).logits(PROBES)
    np.testing.assert_array_equal(student.logits(PROBES), expected)


def test_int8_variant_loads_when_it_passed(bundle: Exported) -> None:
    assert bundle.report.int8 is not None
    if bundle.report.int8.passed:
        student = OnnxStudent.load(bundle.bundle, "int8")
        assert student.logits(PROBES).shape == (len(PROBES), 3)
    else:
        with pytest.raises(ExportError, match="parity gate"):
            OnnxStudent.load(bundle.bundle, "int8")


def test_predict_proba_applies_the_temperature(
    edited: Callable[[Callable[[dict[str, object]], None]], Path],
) -> None:
    path = edited(lambda meta: meta.update(temperature=2.0, threshold=None))
    student = OnnxStudent.load(path)
    assert student.temperature == 2.0
    assert student.threshold is None
    logits = student.logits(PROBES).astype(np.float64) / 2.0
    expected = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    probs = student.predict_proba(PROBES)
    assert probs.dtype == np.float32
    np.testing.assert_allclose(probs, expected, rtol=1e-6, atol=1e-7)
    assert all(label in student.labels for label in student.predict(PROBES))


def test_empty_input(bundle: Exported) -> None:
    student = OnnxStudent.load(bundle.bundle)
    assert student.logits([]).shape == (0, 3)
    assert student.predict_proba([]).shape == (0, 3)


def test_softmax_is_stable() -> None:
    logits = np.array([[1000.0, 1000.0], [0.0, -1000.0]], dtype=np.float32)
    np.testing.assert_allclose(softmax(logits), [[0.5, 0.5], [1.0, 0.0]])
    np.testing.assert_allclose(softmax(logits, 1e9), [[0.5, 0.5], [0.5, 0.5]], atol=1e-6)


def test_runtime_does_not_import_torch_or_transformers() -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['transformers'] = None\n"
        "import anvil.runtime\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, capture_output=True)
