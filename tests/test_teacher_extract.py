"""Extraction teachers: the Claude request and answer parsing, and the fake teacher's mentions."""

from __future__ import annotations

import json

import pytest

from anvil.spans import Mention, decode_answer, encode_answer
from anvil.spec import TaskSpec
from anvil.teacher.base import Item
from anvil.teacher.claude import ClaudeTeacher
from anvil.teacher.fake import FakeTeacher
from tests.conftest import FakeTransport, message

TEXT = "cheap pizza near the pizza hut downtown"


@pytest.fixture
def extract_spec() -> TaskSpec:
    return TaskSpec.model_validate(
        {
            "name": "slots",
            "kind": "extract",
            "instructions": "Find the restaurant search slots.",
            "labels": [
                {"name": "Dish", "description": "a food"},
                "Restaurant_Name",
                "Location",
            ],
            "teacher": {"model": "claude-opus-5-5", "effort": "low", "max_tokens": 1024},
            "fake": {
                "rules": {
                    "Dish": ["pizza"],
                    "Restaurant_Name": ["pizza hut"],
                    "Location": ["downtown", "near me"],
                },
            },
        }
    )


def _answer(*entities: object) -> str:
    return json.dumps({"entities": list(entities)})


def test_request_asks_for_text_type_and_offset(
    extract_spec: TaskSpec, fake_transport: FakeTransport
) -> None:
    params = ClaudeTeacher(extract_spec, fake_transport).request_params(Item("1", TEXT))
    system = params["system"]
    assert "Entity types:" in system
    assert "- Dish: a food" in system
    assert "0-based character offset" in system
    schema = params["output_config"]["format"]["schema"]
    item = schema["properties"]["entities"]["items"]
    assert item["properties"]["type"]["enum"] == ["Dish", "Restaurant_Name", "Location"]
    assert item["properties"]["start"] == {"type": "integer"}  # no `minimum`: unsupported
    assert item["required"] == ["text", "type", "start"]
    assert item["additionalProperties"] is False
    assert schema["additionalProperties"] is False


def test_answer_is_stored_as_canonical_mentions(extract_spec: TaskSpec) -> None:
    transport = FakeTransport(
        message(
            _answer(
                {"text": "downtown", "type": "Location", "start": 31},
                {"text": "pizza", "type": "Dish", "start": 7},  # off by one: kept as claimed
            )
        )
    )
    (label,) = ClaudeTeacher(extract_spec, transport).label([Item("1", TEXT)]).labels
    assert decode_answer(label.label) == [
        Mention("pizza", "Dish", 7),
        Mention("downtown", "Location", 31),
    ]
    assert label.label == encode_answer(decode_answer(label.label))
    assert label.cost_usd > 0


def test_empty_answer_is_a_label(extract_spec: TaskSpec) -> None:
    transport = FakeTransport(message(_answer()))
    (label,) = ClaudeTeacher(extract_spec, transport).label([Item("1", "hello")]).labels
    assert decode_answer(label.label) == []


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('{"label": "x"}', "no `entities`"),
        (_answer({"text": "pizza", "type": "Pasta", "start": 6}), "unknown type"),
        (_answer({"text": "pizza", "type": "Dish", "start": "6"}), "not an integer"),
        (_answer({"text": "pizza", "type": "Dish", "start": True}), "not an integer"),
        (_answer("pizza"), "not an object"),
    ],
)
def test_invalid_answers_are_billed_failures(
    extract_spec: TaskSpec, text: str, reason: str
) -> None:
    batch = ClaudeTeacher(extract_spec, FakeTransport(message(text))).label([Item("1", TEXT)])
    assert batch.labels == []
    (failure,) = batch.failures
    assert reason in failure.reason
    assert failure.cost_usd > 0


def test_decode_rejects_non_answers() -> None:
    for bad in ("nope", '{"a": 1}', '[{"text": "a", "type": "b"}]', "[1]"):
        with pytest.raises(ValueError, match=r"extraction label|mention"):
            decode_answer(bad)


def test_fake_mentions_prefer_the_longest_keyword(extract_spec: TaskSpec) -> None:
    mentions = FakeTeacher(extract_spec).mentions(Item("1", TEXT))
    assert mentions == [
        Mention("pizza", "Dish", 6),
        Mention("pizza hut", "Restaurant_Name", 21),
        Mention("downtown", "Location", 31),
    ]


def test_fake_noise_shifts_offsets_by_one(extract_spec: TaskSpec) -> None:
    assert extract_spec.fake is not None
    noisy = extract_spec.model_copy(
        update={"fake": extract_spec.fake.model_copy(update={"noise": 1.0})}
    )
    items = [Item(str(i), f"{'x ' * i}pizza downtown") for i in range(40)]
    teacher = FakeTeacher(noisy)
    for item in items:
        for m in teacher.mentions(item):
            true_start = item.text.index(m.text)
            assert abs(m.start - true_start) == 1
    again = FakeTeacher(noisy).label(items)
    assert [lb.label for lb in again.labels] == [lb.label for lb in teacher.label(items).labels]
