from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from anvil.errors import ConfigError
from anvil.spec import TaskSpec, load_spec

BASE: dict[str, Any] = {  # Any: a YAML document
    "name": "t",
    "kind": "classify",
    "instructions": "Pick one.",
    "labels": ["a", "b"],
}


def _write(tmp_path: Path, doc: object) -> Path:
    path = tmp_path / "anvil.yaml"
    path.write_text(doc if isinstance(doc, str) else yaml.safe_dump(doc), encoding="utf-8")
    return path


def _config_error(tmp_path: Path, doc: object) -> ConfigError:
    with pytest.raises(ConfigError) as info:
        load_spec(_write(tmp_path, doc))
    assert info.value.exit_code == 2
    return info.value


def test_toy_spec_loads(toy_spec: TaskSpec) -> None:
    assert toy_spec.name == "ticket-triage"
    assert toy_spec.label_names == ["billing", "bug", "feature_request", "account"]
    assert toy_spec.student.model == "tiny"
    assert toy_spec.fake is not None
    assert toy_spec.fake.default == "account"


def test_labels_accept_plain_strings(tmp_path: Path) -> None:
    spec = load_spec(_write(tmp_path, BASE))
    assert spec.label_names == ["a", "b"]
    assert all(label.description == "" for label in spec.labels)


def test_prompt_hash_tracks_the_question_not_the_teacher(toy_spec: TaskSpec) -> None:
    h = toy_spec.prompt_hash
    assert h == load_spec(Path(__file__).parent.parent / "examples/toy/anvil.yaml").prompt_hash
    assert toy_spec.model_copy(update={"instructions": "Something else."}).prompt_hash != h
    fewer = toy_spec.model_copy(update={"labels": toy_spec.labels[:3]})
    assert fewer.prompt_hash != h
    reworded = [toy_spec.labels[0].model_copy(update={"description": "money"})]
    assert toy_spec.model_copy(update={"labels": reworded + toy_spec.labels[1:]}).prompt_hash != h
    other_teacher = toy_spec.teacher.model_copy(update={"model": "claude-haiku-5-5"})
    other_student = toy_spec.student.model_copy(update={"epochs": 1, "seed": 7})
    same = toy_spec.model_copy(update={"teacher": other_teacher, "student": other_student})
    assert same.prompt_hash == h


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found") as info:
        load_spec(tmp_path / "nope.yaml")
    assert info.value.exit_code == 2


def test_bad_yaml(tmp_path: Path) -> None:
    assert "not valid YAML" in str(_config_error(tmp_path, "name: [unclosed\n"))


def test_non_mapping(tmp_path: Path) -> None:
    assert "must be a mapping" in str(_config_error(tmp_path, "- a\n- b\n"))


def test_unknown_field(tmp_path: Path) -> None:
    assert "lables" in str(_config_error(tmp_path, {**BASE, "lables": ["x"]}))


@pytest.mark.parametrize(("kind", "milestone"), [("extract", "M4"), ("generate", "M6")])
def test_unimplemented_kinds(tmp_path: Path, kind: str, milestone: str) -> None:
    assert milestone in str(_config_error(tmp_path, {**BASE, "kind": kind}))


def test_needs_two_labels(tmp_path: Path) -> None:
    assert "at least two labels" in str(_config_error(tmp_path, {**BASE, "labels": ["a"]}))


def test_duplicate_labels(tmp_path: Path) -> None:
    assert "unique" in str(_config_error(tmp_path, {**BASE, "labels": ["a", "b", "a"]}))


@pytest.mark.parametrize(
    "fake",
    [{"rules": {"zzz": ["word"]}}, {"rules": {"a": ["word"]}, "default": "zzz"}],
)
def test_fake_rules_must_name_known_labels(tmp_path: Path, fake: dict[str, object]) -> None:
    assert "zzz" in str(_config_error(tmp_path, {**BASE, "fake": fake}))
