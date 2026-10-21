from __future__ import annotations

from pathlib import Path

import pytest

from anvil.errors import ConfigError, TeacherError
from anvil.spec import TaskSpec
from anvil.teacher.base import BatchAbortedError, Failure, Item
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


def test_config_error_aborts_the_batch_but_keeps_completed_answers(
    toy_spec: TaskSpec, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    fake_transport.errors[ITEM.text] = ConfigError("the API rejected the key")
    teacher = ClaudeTeacher(toy_spec, fake_transport, cache=DiskCache(tmp_path), workers=1)
    with pytest.raises(BatchAbortedError) as info:
        teacher.label([Item("a", "first"), ITEM, Item("b", "never sent")])
    assert isinstance(info.value.cause, ConfigError)
    assert str(info.value.cause) == "the API rejected the key"
    # "a" was answered and paid for before the abort, so it is kept. "b" is cancelled unless
    # the worker had already picked it up; whatever was sent (and billed) is kept too.
    sent = [c["messages"][0]["content"] for c in fake_transport.calls]
    assert sent[:2] == ["first", ITEM.text]
    kept = {lb.input_id for lb in info.value.batch.labels}
    assert kept == {"a"} | ({"b"} if "never sent" in sent else set())
    assert all(not lb.cached and lb.cost_usd == 0.005 for lb in info.value.batch.labels)


def test_identical_texts_are_requested_once(
    toy_spec: TaskSpec, fake_transport: FakeTransport, tmp_path: Path
) -> None:
    items = [Item("x1", "same ticket"), Item("other", "different"), Item("x2", "same ticket")]
    items.append(Item("x3", "same ticket"))
    batch = ClaudeTeacher(toy_spec, fake_transport, cache=DiskCache(tmp_path)).label(items)
    assert len(fake_transport.calls) == 2
    assert [(lb.input_id, lb.cached) for lb in batch.labels] == [
        ("x1", False),
        ("other", False),
        ("x2", True),
        ("x3", True),
    ]
    assert batch.spent_usd == pytest.approx(0.010)
    assert batch.cache_hits == 2
    # The shared answer is in the cache exactly once, under the shared request key.
    assert len(list(tmp_path.rglob("*.json"))) == 2


def test_identical_texts_share_a_failure(
    toy_spec: TaskSpec, fake_transport: FakeTransport
) -> None:
    fake_transport.program("same ticket", message("nope"))
    items = [Item("x1", "same ticket"), Item("x2", "same ticket")]
    batch = ClaudeTeacher(toy_spec, fake_transport).label(items)
    assert len(fake_transport.calls) == 1
    assert [f.input_id for f in batch.failures] == ["x1", "x2"]
    assert batch.spent_usd == pytest.approx(0.005)


GENERATE = TaskSpec.model_validate(
    {
        "name": "sql",
        "kind": "generate",
        "instructions": "  Write the SQLite query.  ",
        "outputs": [{"name": "sql", "description": "the query"}, "explanation"],
    }
)


def test_generate_request_lists_fields_and_asks_for_strings(
    fake_transport: FakeTransport,
) -> None:
    params = ClaudeTeacher(GENERATE, fake_transport).request_params(ITEM)
    system = params["system"]
    assert system.startswith("Write the SQLite query.\n\nFields:\n")
    assert "- sql: the query\n- explanation\n" in system
    assert "Labels:" not in system
    assert "JSON object" in system
    schema = params["output_config"]["format"]["schema"]
    assert schema == {
        "type": "object",
        "properties": {"sql": {"type": "string"}, "explanation": {"type": "string"}},
        "required": ["sql", "explanation"],
        "additionalProperties": False,
    }


def test_generate_answer_is_stored_canonically(fake_transport: FakeTransport) -> None:
    text = '{"explanation": "all rows", "sql": "SELECT * FROM \\"t\\" -- é"}'
    fake_transport.program(ITEM.text, message(text))
    batch = ClaudeTeacher(GENERATE, fake_transport).label([ITEM])
    assert batch.failures == []
    (lb,) = batch.labels
    assert lb.label == '{"sql":"SELECT * FROM \\"t\\" -- é","explanation":"all rows"}'
    assert lb.prompt_hash == GENERATE.prompt_hash
    assert lb.cost_usd == pytest.approx(0.005)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ('{"sql": "SELECT 1"}', "'explanation' must be a string"),
        ('{"sql": "SELECT 1", "explanation": "x", "extra": "y"}', "['extra']"),
        ('{"sql": 1, "explanation": "x"}', "'sql' must be a string"),
        ('{"sql": null, "explanation": "x"}', "'sql' must be a string"),
        ('["SELECT 1", "x"]', "not a JSON object"),
        ('"SELECT 1"', "not a JSON object"),
        ("SELECT 1", "not JSON"),
    ],
)
def test_bad_generate_answers_become_failures(
    fake_transport: FakeTransport, text: str, reason: str
) -> None:
    fake_transport.program(ITEM.text, message(text))
    batch = ClaudeTeacher(GENERATE, fake_transport).label([ITEM])
    assert batch.labels == []
    (failure,) = batch.failures
    assert failure.input_id == ITEM.id
    assert reason in failure.reason
    assert failure.cost_usd == pytest.approx(0.005)
