from __future__ import annotations

import pytest

from anvil.errors import ConfigError
from anvil.spec import TaskSpec
from anvil.teacher.base import Item
from anvil.teacher.fake import FakeTeacher


def _with_noise(spec: TaskSpec, noise: float, seed: int = 0) -> TaskSpec:
    assert spec.fake is not None
    fake = spec.fake.model_copy(update={"noise": noise, "seed": seed})
    return spec.model_copy(update={"fake": fake})


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("I was charged twice and the app crashes", "billing"),  # billing comes first
        ("The export crashes with an error", "bug"),
        ("Could you add dark mode?", "feature_request"),
        ("Reset my PASSWORD please", "account"),
        ("hello there", "account"),  # default
        ("the addition of numbers", "account"),  # 'add' only as a whole word
        ("supported browsers", "account"),  # 'support' only as a whole word
        ("server returned 500", "bug"),
    ],
)
def test_rules(toy_spec: TaskSpec, text: str, label: str) -> None:
    assert FakeTeacher(_with_noise(toy_spec, 0.0)).rule_label(text) == label


def test_noise_zero_equals_rules_and_labels_carry_provenance(toy_spec: TaskSpec) -> None:
    spec = _with_noise(toy_spec, 0.0)
    teacher = FakeTeacher(spec)
    items = [Item(f"x{i}", t) for i, t in enumerate(["refund me", "it crashes", "hi"])]
    batch = teacher.label(items)
    assert [lb.label for lb in batch.labels] == ["billing", "bug", "account"]
    assert all(lb.teacher_model == teacher.model_id for lb in batch.labels)
    assert teacher.model_id.startswith("fake-rules-")
    assert all(lb.prompt_hash == spec.prompt_hash for lb in batch.labels)
    assert batch.failures == []
    assert batch.spent_usd == 0.0


def test_default_is_first_label_when_unset(small_spec: TaskSpec) -> None:
    assert small_spec.fake is not None
    fake = small_spec.fake.model_copy(update={"default": None})
    teacher = FakeTeacher(small_spec.model_copy(update={"fake": fake}))
    assert teacher.rule_label("nothing matches") == "bug"


def test_noise_flips_the_right_fraction_deterministically(toy_spec: TaskSpec) -> None:
    texts = ["I was charged twice", "it crashes", "please add export", "new password", "hm"]
    items = [Item(f"s{i}", texts[i % len(texts)]) for i in range(2000)]
    spec = _with_noise(toy_spec, 0.2)
    teacher = FakeTeacher(spec)
    noisy = [lb.label for lb in teacher.label(items).labels]
    rules = [teacher.rule_label(it.text) for it in items]
    flipped = [(n, r) for n, r in zip(noisy, rules, strict=True) if n != r]
    assert abs(len(flipped) / len(items) - 0.2) < 0.03
    assert noisy == [lb.label for lb in FakeTeacher(spec).label(items).labels]
    other_seed = [
        lb.label for lb in FakeTeacher(_with_noise(toy_spec, 0.2, 1)).label(items).labels
    ]
    assert other_seed != noisy
    assert set(noisy) == set(toy_spec.label_names)


def test_flipped_labels_always_differ(toy_spec: TaskSpec) -> None:
    items = [Item(f"f{i}", "it crashes") for i in range(200)]
    labels = [lb.label for lb in FakeTeacher(_with_noise(toy_spec, 1.0)).label(items).labels]
    assert "bug" not in labels
    assert set(labels) == {"billing", "feature_request", "account"}


def test_model_id_changes_with_the_rules(toy_spec: TaskSpec) -> None:
    base = FakeTeacher.model_id_for(toy_spec)
    assert FakeTeacher(toy_spec).model_id == base
    assert FakeTeacher.model_id_for(_with_noise(toy_spec, 0.2)) != base
    assert FakeTeacher.model_id_for(_with_noise(toy_spec, 0.05, seed=1)) != base
    assert toy_spec.fake is not None
    rules = toy_spec.fake.model_copy(update={"rules": {**toy_spec.fake.rules, "bug": ["boom"]}})
    assert FakeTeacher.model_id_for(toy_spec.model_copy(update={"fake": rules})) != base
    # The spec's prompt hash is the teacher's prompt, not the fake rules.
    assert toy_spec.model_copy(update={"fake": rules}).prompt_hash == toy_spec.prompt_hash


def test_needs_fake_section(toy_spec: TaskSpec) -> None:
    with pytest.raises(ConfigError, match="fake"):
        FakeTeacher(toy_spec.model_copy(update={"fake": None}))


def _generate(rewrites: list[dict[str, object]], outputs: list[str] | None = None) -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "name": "sql",
            "kind": "generate",
            "instructions": "Write SQL.",
            "outputs": outputs or ["sql"],
            "fake": {"rewrites": rewrites},
        }
    )


REWRITES: list[dict[str, object]] = [
    {"pattern": r"how old is (\w+)", "output": {"sql": "SELECT age FROM t WHERE name = '\\1'"}},
    {
        "pattern": r"older than (?P<n>\d+)",
        "output": {"sql": r"SELECT name FROM t WHERE age > \g<n>"},
    },
    {"pattern": r"how old", "output": {"sql": "never reached for the inputs above"}},
]


def test_generate_rewrites_expand_groups_and_carry_provenance() -> None:
    spec = _generate(REWRITES)
    teacher = FakeTeacher(spec)
    items = [
        Item("a", "How OLD is Ada?"),  # case insensitive; the group keeps the input's case
        Item("b", "who is older than 40"),
        Item("c", "how old are they"),
    ]
    batch = teacher.label(items)
    assert batch.failures == []
    assert [lb.label for lb in batch.labels] == [
        '{"sql":"SELECT age FROM t WHERE name = \'Ada\'"}',
        '{"sql":"SELECT name FROM t WHERE age > 40"}',
        '{"sql":"never reached for the inputs above"}',
    ]
    assert all(lb.prompt_hash == spec.prompt_hash for lb in batch.labels)
    assert all(lb.teacher_model == teacher.model_id for lb in batch.labels)
    assert batch.spent_usd == 0.0


def test_generate_first_matching_rewrite_wins() -> None:
    first, _, catch_all = REWRITES
    teacher = FakeTeacher(_generate([catch_all, first]))
    assert teacher.rewrite("how old is Ada") == '{"sql":"never reached for the inputs above"}'
    assert FakeTeacher(_generate(REWRITES)).rewrite("how old is Ada") == (
        '{"sql":"SELECT age FROM t WHERE name = \'Ada\'"}'
    )


def test_generate_without_a_match_is_a_failure() -> None:
    batch = FakeTeacher(_generate(REWRITES)).label([Item("x", "hello"), Item("y", "how old")])
    assert [lb.input_id for lb in batch.labels] == ["y"]
    (failure,) = batch.failures
    assert failure.input_id == "x"
    assert "no fake rewrite matches" in failure.reason


def test_generate_rewrites_write_fields_in_spec_order() -> None:
    rule: dict[str, object] = {
        "pattern": r"(\w+)",
        "output": {"why": "because \\1", "sql": "SELECT \\1"},
    }
    teacher = FakeTeacher(_generate([rule], outputs=["sql", "why"]))
    assert teacher.rewrite("x") == '{"sql":"SELECT x","why":"because x"}'


def test_generate_model_id_changes_with_the_rewrites() -> None:
    other = [{**REWRITES[0], "output": {"sql": "SELECT 1"}}, *REWRITES[1:]]
    assert FakeTeacher.model_id_for(_generate(REWRITES)) != FakeTeacher.model_id_for(
        _generate(other)
    )
