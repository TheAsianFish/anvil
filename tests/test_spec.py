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


def test_prompt_hash_tracks_what_changes_answers(toy_spec: TaskSpec) -> None:
    h = toy_spec.prompt_hash
    assert h == load_spec(Path(__file__).parent.parent / "examples/toy/anvil.yaml").prompt_hash
    assert toy_spec.model_copy(update={"instructions": "Something else."}).prompt_hash != h
    fewer = toy_spec.model_copy(update={"labels": toy_spec.labels[:3]})
    assert fewer.prompt_hash != h
    reworded = [toy_spec.labels[0].model_copy(update={"description": "money"})]
    assert toy_spec.model_copy(update={"labels": reworded + toy_spec.labels[1:]}).prompt_hash != h
    other_teacher = toy_spec.teacher.model_copy(update={"model": "claude-haiku-5-5"})
    other_student = toy_spec.student.model_copy(update={"epochs": 1, "seed": 7})
    more_tokens = other_teacher.model_copy(update={"max_tokens": 8000})
    same = toy_spec.model_copy(update={"teacher": more_tokens, "student": other_student})
    assert same.prompt_hash == h
    # Effort changes how the teacher answers, so it must change the label key.
    effort = toy_spec.teacher.model_copy(update={"effort": "high"})
    assert toy_spec.model_copy(update={"teacher": effort}).prompt_hash != h


def test_prompt_hash_covers_the_request_format(
    toy_spec: TaskSpec, monkeypatch: pytest.MonkeyPatch
) -> None:
    import anvil.spec

    before = toy_spec.prompt_hash
    monkeypatch.setattr(anvil.spec, "PROMPT_FORMAT_VERSION", 2)
    assert toy_spec.prompt_hash != before


@pytest.mark.parametrize("name", ["t\n", "a b", "../x", "", "x/y"])
def test_names_must_match_in_full(tmp_path: Path, name: str) -> None:
    _config_error(tmp_path, {**BASE, "name": name})
    _config_error(tmp_path, {**BASE, "labels": ["a", name]})


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


GENERATE: dict[str, Any] = {  # Any: a YAML document
    "name": "t",
    "kind": "generate",
    "instructions": "Write the SQL query.",
    "outputs": [{"name": "sql", "description": "the query"}],
}


def test_generate_takes_outputs(tmp_path: Path) -> None:
    spec = load_spec(_write(tmp_path, GENERATE))
    assert spec.output_names == ["sql"]
    assert spec.label_names == []
    two = load_spec(_write(tmp_path, {**GENERATE, "outputs": ["query", "explanation"]}))
    assert two.output_names == ["query", "explanation"]
    assert all(field.description == "" for field in two.outputs)


@pytest.mark.parametrize(
    ("doc", "message"),
    [
        ({**GENERATE, "outputs": []}, "at least one field under `outputs`"),
        ({k: v for k, v in GENERATE.items() if k != "outputs"}, "at least one field"),
        ({**GENERATE, "labels": ["a", "b"]}, "not labels"),
        ({**GENERATE, "outputs": ["sql", "sql"]}, "output field names must be unique"),
        ({**GENERATE, "outputs": ["a b"]}, "must be letters"),
        ({**BASE, "outputs": ["sql"]}, "`outputs` is for generate tasks"),
        ({**BASE, "kind": "extract", "labels": ["Dish"], "outputs": ["sql"]}, "`outputs`"),
    ],
)
def test_invalid_generate_specs(tmp_path: Path, doc: dict[str, Any], message: str) -> None:
    assert message in str(_config_error(tmp_path, doc))


def _rewrite(output: dict[str, str], pattern: str = r"show (\w+)") -> dict[str, object]:
    return {"pattern": pattern, "output": output}


def test_generate_fake_rewrites(tmp_path: Path) -> None:
    doc = {**GENERATE, "fake": {"rewrites": [_rewrite({"sql": r"SELECT \1"})]}}
    spec = load_spec(_write(tmp_path, doc))
    assert spec.fake is not None
    assert spec.fake.rewrites[0].output == {"sql": r"SELECT \1"}


@pytest.mark.parametrize(
    ("fake", "message"),
    [
        ({"rewrites": [_rewrite({"query": "x"})]}, "writes fields ['query']"),
        ({"rewrites": [_rewrite({"sql": "x", "extra": "y"})]}, "the task's outputs are ['sql']"),
        ({"rewrites": [_rewrite({})]}, "writes fields []"),
        ({"rewrites": [_rewrite({"sql": "x"}, "show (")]}, "is not a regular expression"),
        ({"rules": {"sql": ["x"]}}, "uses fake.rewrites, not rules"),
        ({"default": "x"}, "uses fake.rewrites, not rules"),
    ],
)
def test_invalid_generate_fakes(tmp_path: Path, fake: dict[str, object], message: str) -> None:
    assert message in str(_config_error(tmp_path, {**GENERATE, "fake": fake}))


@pytest.mark.parametrize("kind", ["classify", "extract"])
def test_rewrites_are_only_for_generate(tmp_path: Path, kind: str) -> None:
    doc = {**BASE, "kind": kind, "fake": {"rewrites": [_rewrite({"sql": "x"})]}}
    assert "fake.rewrites are for generate tasks" in str(_config_error(tmp_path, doc))


def test_classify_prompt_hash_is_unchanged_by_generate_fields(small_spec: TaskSpec) -> None:
    # Pinned from M5: labels recorded before generate existed must keep their keys.
    assert small_spec.prompt_hash == "e7161d2efbe250ed"


def test_generate_prompt_hash_tracks_the_outputs() -> None:
    spec = TaskSpec.model_validate(GENERATE)
    h = spec.prompt_hash
    reworded = [spec.outputs[0].model_copy(update={"description": "a SQLite query"})]
    assert spec.model_copy(update={"outputs": reworded}).prompt_hash != h
    renamed = [spec.outputs[0].model_copy(update={"name": "query"})]
    assert spec.model_copy(update={"outputs": renamed}).prompt_hash != h
    more = TaskSpec.model_validate({**GENERATE, "outputs": ["sql", "why"]})
    assert more.prompt_hash != h
    assert spec.model_copy(update={"instructions": "Other."}).prompt_hash != h
    rewrites = {"rewrites": [_rewrite({"sql": "x"})]}
    faked = TaskSpec.model_validate({**GENERATE, "fake": rewrites})
    assert faked.prompt_hash == h  # the fake teacher's rules are not the prompt
    classify = TaskSpec.model_validate({**BASE, "instructions": GENERATE["instructions"]})
    assert classify.prompt_hash != h


def test_generate_student_settings(tmp_path: Path) -> None:
    doc = {**GENERATE, "student": {"max_new_tokens": 64, "lora": {"r": 8, "alpha": 16}}}
    spec = load_spec(_write(tmp_path, doc))
    assert spec.student.max_new_tokens == 64
    assert (spec.student.lora.r, spec.student.lora.alpha) == (8, 16)
    assert "q_proj" in spec.student.lora.target_modules
    bad_lora: list[dict[str, object]] = [{"r": 0}, {"target_modules": []}, {"dropout": 1.0}]
    for bad in bad_lora:
        _config_error(tmp_path, {**GENERATE, "student": {"lora": bad}})
    _config_error(tmp_path, {**GENERATE, "student": {"max_new_tokens": 0}})


def test_extract_takes_entity_types(tmp_path: Path) -> None:
    spec = load_spec(_write(tmp_path, {**BASE, "kind": "extract", "labels": ["Dish"]}))
    assert spec.label_names == ["Dish"]
    assert "at least one entity type" in str(
        _config_error(tmp_path, {**BASE, "kind": "extract", "labels": []})
    )


def test_extract_fake_rules_have_no_default(tmp_path: Path) -> None:
    doc = {**BASE, "kind": "extract", "fake": {"rules": {"a": ["x"]}, "default": "a"}}
    assert "fake.default" in str(_config_error(tmp_path, doc))
    doc = {**BASE, "kind": "extract", "fake": {"rules": {"zzz": ["x"]}}}
    assert "unknown labels" in str(_config_error(tmp_path, doc))


def test_kind_is_part_of_the_prompt_hash() -> None:
    classify = TaskSpec.model_validate(BASE)
    extract = TaskSpec.model_validate({**BASE, "kind": "extract"})
    assert classify.prompt_hash != extract.prompt_hash


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
