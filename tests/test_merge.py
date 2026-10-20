"""The merged export of a LoRA generator: the parity gate, atomic replacement and loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoModelForCausalLM

from anvil import merge
from anvil.errors import ExportError
from anvil.merge import MERGED_FORMAT, MERGED_META, export_generator, load_merged
from anvil.outputs import encode_output
from anvil.spec import StudentSpec
from anvil.students.generator import LoraGenerator

OUTPUTS = ["sql"]
SPEC = StudentSpec(
    model="tiny", max_length=96, max_new_tokens=16, epochs=6, batch_size=8,
    learning_rate=1e-2, validation_fraction=0.0, warmup_ratio=0.0, seed=0,
)  # fmt: skip
PROBES = ["show id of cars", "show name of users please", "x", "show age of pets"]


@pytest.fixture(scope="module")
def student_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A briefly trained tiny generator (its adapter is no longer zero), saved."""
    texts, answers = [], []
    for table in ["users", "orders", "pets", "cars"]:
        for column in ["name", "age", "city", "id"]:
            texts.append(f"show {column} of {table}")
            answers.append(encode_output({"sql": f"SELECT {column} FROM {table}"}, OUTPUTS))
    student = LoraGenerator.build(SPEC, OUTPUTS, "SQL.")
    student.fit(texts, answers, SPEC)
    assert any(
        float(p.detach().abs().max()) > 0
        for n, p in student.model.named_parameters()
        if "lora_B" in n
    )
    path = tmp_path_factory.mktemp("merge") / "student"
    student.save(path)
    return path


@pytest.fixture(scope="module")
def merged_dir(student_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("merged") / "out"
    report = export_generator(student_dir, out, PROBES, spec={"name": "t"})
    assert report.parity.passed
    return out


def _with_sentinel(out: Path) -> Path:
    out.mkdir(parents=True)
    sentinel = out / "old.txt"
    sentinel.write_text("previous export", encoding="utf-8")
    return sentinel


def test_no_probes_is_an_export_error(student_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="no probe inputs"):
        export_generator(student_dir, tmp_path / "out", [])
    assert not (tmp_path / "out").exists()


def test_a_failed_logit_gate_writes_nothing(
    student_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    sentinel = _with_sentinel(out)
    monkeypatch.setattr(merge, "MERGE_MAX_ABS_DIFF", -1.0)
    with pytest.raises(ExportError, match="nothing was written"):
        export_generator(student_dir, out, PROBES)
    assert sorted(p.name for p in out.iterdir()) == ["old.txt"]
    assert sentinel.read_text(encoding="utf-8") == "previous export"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]  # no temporary left behind


def test_a_perturbed_merge_fails_the_gate(
    student_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = LoraGenerator.merged

    def perturbed(self: LoraGenerator) -> Any:  # Any: a transformers model
        model = original(self)
        with torch.no_grad():
            for p in model.parameters():
                p.add_(0.5 * torch.randn_like(p, generator=torch.Generator().manual_seed(0)))
        return model

    monkeypatch.setattr(LoraGenerator, "merged", perturbed)
    out = tmp_path / "out"
    with pytest.raises(ExportError, match="does not answer as the adapter does"):
        export_generator(student_dir, out, PROBES)
    assert not out.exists()
    assert not list(tmp_path.iterdir())


def test_a_passing_export_replaces_the_old_one(student_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    _with_sentinel(out)
    report = export_generator(student_dir, out, PROBES)
    assert report.out == out
    assert report.parity.passed
    assert report.parity.same_completions == report.parity.probes == len(PROBES)
    assert report.parity.logit_probes == len(PROBES)
    assert report.parity.max_abs_logit_diff <= merge.MERGE_MAX_ABS_DIFF
    assert report.size_mb > 0
    assert not (out / "old.txt").exists()
    assert (out / MERGED_META).exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["out"]


def test_adapter_parameters_are_the_students_lora_count(student_dir: Path, tmp_path: Path) -> None:
    student = LoraGenerator.load(student_dir)
    expected = sum(p.numel() for n, p in student.model.named_parameters() if "lora_" in n)
    assert expected > 0
    report = export_generator(student_dir, tmp_path / "out", PROBES[:1])
    assert report.adapter_parameters == expected == student.adapter_parameters


def test_merged_meta_describes_the_student(merged_dir: Path, student_dir: Path) -> None:
    meta = json.loads((merged_dir / MERGED_META).read_text(encoding="utf-8"))
    assert meta["format"] == MERGED_FORMAT
    assert meta["student"] == student_dir.name
    assert meta["outputs"] == OUTPUTS
    assert meta["system"] == "SQL."
    assert meta["max_new_tokens"] == SPEC.max_new_tokens
    assert meta["parity"]["passed"] is True
    assert meta["spec"] == {"name": "t"}
    assert "outputs" not in meta["source"]


def test_merged_dir_is_a_plain_transformers_model(merged_dir: Path) -> None:
    assert not (merged_dir / "adapter_config.json").exists()
    assert not list(merged_dir.rglob("adapter_config.json"))
    model: Any = AutoModelForCausalLM.from_pretrained(merged_dir)  # Any: transformers model
    assert type(model).__module__.startswith("transformers.")
    names = [n for n, _ in model.named_parameters()]
    assert names
    assert not [n for n in names if "lora_" in n or "base_layer" in n]


def test_load_merged_answers_as_the_adapter(merged_dir: Path, student_dir: Path) -> None:
    merged = load_merged(merged_dir)
    assert merged.meta["merged_from"] == student_dir.name
    assert merged.adapter_parameters == 0
    texts = [*PROBES, "show city of orders"]
    assert merged.generate(texts) == LoraGenerator.load(student_dir).generate(texts)


def test_load_merged_refuses_a_dir_without_meta(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="not a merged Anvil export"):
        load_merged(tmp_path)


def test_load_merged_refuses_another_format(tmp_path: Path) -> None:
    (tmp_path / MERGED_META).write_text(json.dumps({"format": 99}), encoding="utf-8")
    with pytest.raises(ExportError, match="export format 99; re-export"):
        load_merged(tmp_path)
