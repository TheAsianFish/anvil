from __future__ import annotations

from pathlib import Path

import pytest

from anvil.errors import ConfigError, TeacherError
from anvil.spec import TaskSpec
from anvil.teacher.base import Failure, Item
from anvil.teacher.cache import DiskCache
from anvil.teacher.claude import ClaudeTeacher
from tests.conftest import FakeTransport, message

ITEM = Item("t1", "The app crashes on launch.")


def test_request_params_shape(toy_spec: TaskSpec, fake_transport: FakeTransport) -> None:
    params = ClaudeTeacher(toy_spec, fake_transport).request_params(ITEM)
    assert set(params) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert params["model"] == "claude-opus-5-5"
    assert params["max_tokens"] == toy_spec.teacher.max_tokens
    system = params["system"]
    assert toy_spec.instructions.strip() in system
    for label in toy_spec.labels:
        assert f"- {label.name}: {label.description}" in system
    assert params["messages"] == [{"role": "user", "content": ITEM.text}]
    config = params["output_config"]
    assert config["effort"] == "low"
    assert config["format"]["type"] == "json_schema"
    schema = config["format"]["schema"]
    assert schema["properties"]["label"]["enum"] == toy_spec.label_names
    assert schema["required"] == ["label"]
    assert schema["additionalProperties"] is False


def test_label_with_tokens_and_cost(toy_spec: TaskSpec, fake_transport: FakeTransport) -> None:
    batch = ClaudeTeacher(toy_spec, fake_transport).label([ITEM])
    assert batch.failures == []
    (lb,) = batch.labels
    assert (lb.input_id, lb.label, lb.teacher_model) == ("t1", "bug", "claude-opus-5-5")
    assert lb.prompt_hash == toy_spec.prompt_hash
    assert (lb.input_tokens, lb.output_tokens) == (1000, 50)
    assert lb.cost_usd == pytest.approx(0.005)
    assert not lb.cached
    assert batch.spent_usd == pytest.approx(0.005)
    assert fake_transport.calls == [ClaudeTeacher(toy_spec, fake_transport).request_params(ITEM)]


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (
            {**message(""), "stop_reason": "refusal", "stop_details": {"category": "cyber"}},
            "refusal (cyber)",
        ),
        (message('{"lab', stop_reason="max_tokens"), "max_tokens"),
        (message("Sure! It is a bug."), "not JSON"),
        (message('{"label": "spam"}'), "'spam' is not in the label set"),
        (message('["bug"]'), "not in the label set"),
    ],
)
def test_bad_answers_become_failures(
    toy_spec: TaskSpec, fake_transport: FakeTransport, response: dict[str, object], reason: str
) -> None:
    ok = Item("t2", "fine")
    fake_transport.program(ITEM.text, response)
    batch = ClaudeTeacher(toy_spec, fake_transport).label([ITEM, ok])
    assert [lb.input_id for lb in batch.labels] == ["t2"]
    assert len(batch.failures) == 1
    assert isinstance(batch.failures[0], Failure)
    assert batch.failures[0].input_id == "t1"
    assert reason in batch.failures[0].reason
    # A failed response is still billed (1000 in / 50 out on Opus 5.5 = $0.005).
    assert batch.failures[0].cost_usd == pytest.approx(0.005)
    assert batch.spent_usd == pytest.approx(0.010)


def test_only_text_blocks_are_parsed(toy_spec: TaskSpec, fake_transport: FakeTransport) -> None:
    # Opus 5.5 always thinks, so a thinking block precedes the answer in real responses.
    thought = {"type": "thinking", "thinking": "A crash is a bug.", "signature": "sig"}
    two_texts = {**message(""), "content": [thought, {"type": "text", "text": '{"label"'}]}
    two_texts["content"].append({"type": "text", "text": ': "bug"}'})
    fake_transport.program(ITEM.text, two_texts)
    (lb,) = ClaudeTeacher(toy_spec, fake_transport).label([ITEM]).labels
    assert lb.label == "bug"


def test_response_without_usage_costs_nothing(
    toy_spec: TaskSpec, fake_transport: FakeTransport
) -> None:
    fake_transport.program(ITEM.text, {k: v for k, v in message().items() if k != "usage"})
    (lb,) = ClaudeTeacher(toy_spec, fake_transport).label([ITEM]).labels
    assert (lb.input_tokens, lb.output_tokens, lb.cost_usd) == (0, 0, 0.0)


def test_cache_makes_reruns_free(
    toy_spec: TaskSpec, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    items = [Item(f"c{i}", f"ticket {i}") for i in range(5)]
    cache = DiskCache(tmp_path / "cache")
    first = ClaudeTeacher(toy_spec, fake_transport, cache=cache).label(items)
    assert len(fake_transport.calls) == 5
    second = ClaudeTeacher(toy_spec, fake_transport, cache=cache).label(items)
    assert len(fake_transport.calls) == 5
    assert all(lb.cached for lb in second.labels)
    assert second.cache_hits == 5
    assert second.spent_usd == 0
    assert [lb.cost_usd for lb in second.labels] == [lb.cost_usd for lb in first.labels]
    assert [lb.label for lb in second.labels] == [lb.label for lb in first.labels]
    assert second.labels[0].cost_usd == pytest.approx(0.005)


def test_failures_are_not_cached(
    toy_spec: TaskSpec, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    fake_transport.program(ITEM.text, message("nope"))
    teacher = ClaudeTeacher(toy_spec, fake_transport, cache=DiskCache(tmp_path))
    assert len(teacher.label([ITEM]).failures) == 1
    del fake_transport.responses[ITEM.text]
    batch = teacher.label([ITEM])
    assert len(fake_transport.calls) == 2
    assert [lb.cached for lb in batch.labels] == [False]


def test_changed_instructions_miss_the_cache(
    toy_spec: TaskSpec, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    cache = DiskCache(tmp_path)
    ClaudeTeacher(toy_spec, fake_transport, cache=cache).label([ITEM])
    edited = toy_spec.model_copy(update={"instructions": "Route tickets carefully."})
    batch = ClaudeTeacher(edited, fake_transport, cache=cache).label([ITEM])
    assert len(fake_transport.calls) == 2
    assert batch.cache_hits == 0
    assert batch.labels[0].prompt_hash == edited.prompt_hash != toy_spec.prompt_hash


def test_unknown_model_fails_at_construction(
    toy_spec: TaskSpec, fake_transport: FakeTransport
) -> None:
    spec = toy_spec.model_copy(
        update={"teacher": toy_spec.teacher.model_copy(update={"model": "claude-nope"})}
    )
    with pytest.raises(ConfigError, match="claude-nope"):
        ClaudeTeacher(spec, fake_transport)


def test_concurrent_labels_keep_input_order(
    toy_spec: TaskSpec, fake_transport: FakeTransport
) -> None:
    items = [Item(f"w{i:02d}", f"ticket number {i}") for i in range(20)]
    for i, it in enumerate(items):
        name = toy_spec.label_names[i % 4]
        fake_transport.program(it.text, message(f'{{"label": "{name}"}}'))
        fake_transport.delays[it.text] = (20 - i) * 0.002  # the last submitted finishes first
    batch = ClaudeTeacher(toy_spec, fake_transport, workers=4).label(items)
    assert [lb.input_id for lb in batch.labels] == [it.id for it in items]
    assert [lb.label for lb in batch.labels] == [toy_spec.label_names[i % 4] for i in range(20)]
    assert len(fake_transport.calls) == 20


def test_transport_error_fails_one_input_only(
    toy_spec: TaskSpec, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    fake_transport.errors[ITEM.text] = TeacherError("API error 529 after retries")
    items = [Item("a", "first"), ITEM, Item("b", "last")]
    teacher = ClaudeTeacher(toy_spec, fake_transport, cache=DiskCache(tmp_path))
    batch = teacher.label(items)
    assert [lb.input_id for lb in batch.labels] == ["a", "b"]
    assert batch.failures == [Failure("t1", "API error 529 after retries")]
    del fake_transport.errors[ITEM.text]
    retry = teacher.label([ITEM])
    assert [lb.cached for lb in retry.labels] == [False]


def test_transport_config_error_propagates(
    toy_spec: TaskSpec, fake_transport: FakeTransport
) -> None:
    fake_transport.errors[ITEM.text] = ConfigError("the API rejected the key")
    with pytest.raises(ConfigError, match="rejected the key"):
        ClaudeTeacher(toy_spec, fake_transport).label([Item("a", "first"), ITEM])
