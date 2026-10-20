from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from anvil.errors import ConfigError, TeacherError
from anvil.evaluate.judge import (
    JUDGE_INSTRUCTIONS,
    NOT_RECORDED,
    ClaudeJudge,
    JudgeItem,
    RecordedThenLiveJudge,
    RecordedVerdict,
    Verdict,
    VerdictFileJudge,
    judge_sample,
    judge_vs_metric,
    read_verdict_file,
    render,
    rubric_hash,
    spent_usd,
    write_verdict_file,
)
from anvil.teacher.cache import DiskCache
from tests.conftest import FakeTransport, JSONDict, message

HAIKU = "claude-haiku-5-5"
# 1000 input and 50 output tokens on Haiku 5.5: (1000 * 0.10 + 50 * 0.50) / 1e6.
COST = 0.000125

ITEM = JudgeItem(
    "q1",
    "Table: cars (name, year, price)\nQuestion: How many cars were made after 2010?",
    "SELECT COUNT(*) FROM cars WHERE year > 2010",
    "select count(name) from cars where year > 2010",
)


def answer(verdict: str = "equivalent", reason: str = "Both count the same rows.") -> JSONDict:
    return message(json.dumps({"reason": reason, "verdict": verdict}), model=HAIKU)


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport(default=answer())


def item(i: int) -> JudgeItem:
    return JudgeItem(f"q{i}", f"Question {i}?", f"SELECT {i}", f"SELECT {i} AS x")


def test_request_shape(transport: FakeTransport) -> None:
    params = ClaudeJudge(transport).request_params(ITEM)
    assert set(params) == {"model", "max_tokens", "system", "messages", "output_config"}
    assert (params["model"], params["max_tokens"]) == (HAIKU, 2048)
    assert params["system"] == JUDGE_INSTRUCTIONS
    config = params["output_config"]
    assert config["effort"] == "low"
    assert config["format"]["type"] == "json_schema"
    schema = config["format"]["schema"]
    assert list(schema["properties"]) == ["reason", "verdict"]  # reason before the verdict
    assert schema["properties"]["verdict"]["enum"] == ["equivalent", "different"]
    assert set(schema["required"]) == {"reason", "verdict"}
    assert schema["additionalProperties"] is False
    (msg,) = params["messages"]
    assert msg["role"] == "user"
    text = msg["content"]
    positions = [
        text.index(ITEM.question),
        text.index("Reference answer:"),
        text.index(ITEM.reference),
        text.index("Candidate answer:"),
        text.index(ITEM.candidate),
    ]
    assert positions == sorted(positions)


@pytest.mark.parametrize(("verdict", "equivalent"), [("equivalent", True), ("different", False)])
def test_parses_verdicts(transport: FakeTransport, verdict: str, equivalent: bool) -> None:
    transport.program(render(ITEM), answer(verdict, "Same filter."))
    (v,) = ClaudeJudge(transport).judge([ITEM])
    assert (v.id, v.equivalent, v.reason, v.cached) == ("q1", equivalent, "Same filter.", False)
    assert (v.input_tokens, v.output_tokens) == (1000, 50)
    assert v.cost_usd == pytest.approx(COST)
    assert transport.calls == [ClaudeJudge(transport).request_params(ITEM)]


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (
            {**message("", model=HAIKU), "stop_reason": "refusal", "stop_details": None},
            "refusal (no category)",
        ),
        (message('{"reason": "Bo', stop_reason="max_tokens", model=HAIKU), "max_tokens"),
        (message("They look the same to me.", model=HAIKU), "not JSON"),
        (message('{"reason": "x", "verdict": "maybe"}', model=HAIKU), "'maybe' is not"),
        (message('["equivalent"]', model=HAIKU), "not an object"),
        (message('{"verdict": "equivalent"}', model=HAIKU), "no `reason`"),
    ],
)
def test_bad_answers_fail_and_are_billed(
    transport: FakeTransport, response: JSONDict, reason: str
) -> None:
    transport.program(render(ITEM), response)
    v, ok = ClaudeJudge(transport).judge([ITEM, item(2)])
    assert v.failed
    assert v.equivalent is None
    assert reason in v.reason
    assert v.cost_usd == pytest.approx(COST)
    assert ok.equivalent is True


def test_transport_error_fails_one_item_at_no_cost(transport: FakeTransport) -> None:
    transport.errors[render(ITEM)] = TeacherError("API error 529 after retries")
    v, ok = ClaudeJudge(transport).judge([ITEM, item(2)])
    assert v == Verdict("q1", None, "API error 529 after retries")
    assert not ok.failed


def test_config_error_propagates(transport: FakeTransport) -> None:
    transport.errors[render(ITEM)] = ConfigError("bad key")
    with pytest.raises(ConfigError, match="bad key"):
        ClaudeJudge(transport).judge([ITEM])


def test_cache_makes_reruns_free(transport: FakeTransport, tmp_path: Path) -> None:
    items = [item(i) for i in range(4)]
    cache = DiskCache(tmp_path)
    first = ClaudeJudge(transport, cache=cache).judge(items)
    second = ClaudeJudge(transport, cache=cache).judge(items)
    assert len(transport.calls) == 4
    assert all(v.cached for v in second)
    assert not any(v.cached for v in first)
    assert spent_usd(second) == 0
    assert spent_usd(first) == pytest.approx(4 * COST)
    assert [v.cost_usd for v in second] == [v.cost_usd for v in first]


def test_failures_are_not_cached(transport: FakeTransport, tmp_path: Path) -> None:
    transport.program(render(ITEM), message("nope", model=HAIKU))
    judge = ClaudeJudge(transport, cache=DiskCache(tmp_path))
    assert judge.judge([ITEM])[0].failed
    del transport.responses[render(ITEM)]
    (v,) = judge.judge([ITEM])
    assert len(transport.calls) == 2
    assert v.equivalent is True
    assert not v.cached


def test_duplicate_items_are_sent_once(transport: FakeTransport) -> None:
    twin = JudgeItem("q9", ITEM.question, ITEM.reference, ITEM.candidate)
    a, b = ClaudeJudge(transport).judge([ITEM, twin])
    assert len(transport.calls) == 1
    assert (a.id, b.id) == ("q1", "q9")
    assert not a.cached
    assert b.cached
    assert spent_usd([a, b]) == pytest.approx(COST)


def test_shared_failure_is_billed_once(transport: FakeTransport) -> None:
    transport.program(render(ITEM), message("nope", model=HAIKU))
    twin = JudgeItem("q9", ITEM.question, ITEM.reference, ITEM.candidate)
    a, b = ClaudeJudge(transport).judge([ITEM, twin])
    assert a.failed
    assert b.failed
    assert a.cost_usd == pytest.approx(COST)
    assert b.cost_usd == 0.0


def test_input_order_is_kept_under_concurrency(transport: FakeTransport) -> None:
    items = [item(i) for i in range(6)]
    for i, it in enumerate(items):
        transport.delays[render(it)] = 0.05 * (6 - i)  # earlier items finish last
        transport.program(render(it), answer("different" if i % 2 else "equivalent", f"r{i}"))
    verdicts = ClaudeJudge(transport, workers=6).judge(items)
    assert [v.id for v in verdicts] == [it.id for it in items]
    assert [v.reason for v in verdicts] == [f"r{i}" for i in range(6)]
    assert [v.equivalent for v in verdicts] == [i % 2 == 0 for i in range(6)]


def test_unknown_model_fails_at_construction(transport: FakeTransport) -> None:
    with pytest.raises(ConfigError, match="no price"):
        ClaudeJudge(transport, model="claude-unknown")


def test_empty_batch_makes_no_calls(transport: FakeTransport) -> None:
    assert ClaudeJudge(transport).judge([]) == []
    assert transport.calls == []


# --- Verdict files -------------------------------------------------------------------------


def test_rubric_hash_is_stable_and_short() -> None:
    assert rubric_hash() == rubric_hash()
    assert len(rubric_hash()) == 16
    int(rubric_hash(), 16)


def test_verdict_file_round_trip(transport: FakeTransport, tmp_path: Path) -> None:
    transport.program(render(item(1)), answer("different", "Other column."))
    transport.program(render(item(2)), {**message("", model=HAIKU), "stop_reason": "refusal"})
    transport.errors[render(item(3))] = TeacherError("connection dropped")
    verdicts = ClaudeJudge(transport).judge([item(i) for i in (4, 1, 2, 3)])
    path = tmp_path / "verdicts" / "judge.jsonl"
    # The transport failure never reached the judge, so it is not recorded.
    assert write_verdict_file(path, verdicts, HAIKU, rubric_hash()) == 3
    rows = read_verdict_file(path)
    assert [r.id for r in rows] == ["q1", "q2", "q4"]  # sorted by id
    assert rows[0] == RecordedVerdict(
        "q1", False, "Other column.", HAIKU, rubric_hash(), 1000, 50, rows[0].cost_usd
    )
    assert rows[0].cost_usd == pytest.approx(COST)
    assert rows[1].equivalent is None
    replay = VerdictFileJudge(rows, HAIKU).judge([item(i) for i in (1, 2, 3, 4)])
    assert [(v.id, v.equivalent) for v in replay] == [
        ("q1", False),
        ("q2", None),
        ("q3", None),
        ("q4", True),
    ]
    assert replay[1].reason == "refusal (no category)"
    assert replay[2].reason == NOT_RECORDED
    assert all(v.cached for v in replay if v.reason != NOT_RECORDED)
    assert replay[0].cost_usd == pytest.approx(COST)
    assert spent_usd(replay) == 0
    assert not list(path.parent.glob("*.tmp"))


def test_valid_verdict_supersedes_failure(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    failed = Verdict("q1", None, "refusal (cyber)", 10, 0, 0.001)
    good = Verdict("q1", True, "Same.", 10, 5, 0.002)
    write_verdict_file(path, [good, failed], HAIKU, rubric_hash())
    (row,) = read_verdict_file(path)
    assert row.equivalent is True


def test_changed_rubric_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    write_verdict_file(path, [Verdict("q1", True, "Same.", 10, 5, 0.1)], HAIKU, "0" * 16)
    with pytest.raises(ConfigError, match="re-judge"):
        VerdictFileJudge.from_file(path, HAIKU)


def test_other_model_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "v.jsonl"
    write_verdict_file(path, [Verdict("q1", True, "Same.", 10, 5, 0.1)], HAIKU, rubric_hash())
    with pytest.raises(ConfigError, match="model changed"):
        VerdictFileJudge.from_file(path, "claude-opus-5-5")


def test_bad_rows_and_missing_file_are_config_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        read_verdict_file(tmp_path / "missing.jsonl")
    path = tmp_path / "bad.jsonl"
    path.write_text('{"id": "q1"}\n', encoding="utf-8")
    with pytest.raises(ConfigError, match=":1 is not a recorded verdict"):
        read_verdict_file(path)


def test_recorded_then_live_only_pays_for_missing(transport: FakeTransport) -> None:
    rows = [
        RecordedVerdict("q1", False, "Recorded.", HAIKU, rubric_hash(), 10, 5, 0.5),
        RecordedVerdict("q3", None, "refusal (cyber)", HAIKU, rubric_hash(), 10, 0, 0.5),
    ]
    judge = RecordedThenLiveJudge(VerdictFileJudge(rows, HAIKU), ClaudeJudge(transport))
    assert judge.model_id == HAIKU
    verdicts = judge.judge([item(i) for i in (1, 2, 3, 4)])
    sent = {m["messages"][0]["content"] for m in transport.calls}
    assert sent == {render(item(2)), render(item(4))}
    assert len(transport.calls) == 2
    assert [v.id for v in verdicts] == ["q1", "q2", "q3", "q4"]
    assert [v.cached for v in verdicts] == [True, False, True, False]
    assert verdicts[2].reason == "refusal (cyber)"  # a recorded failure is not retried
    assert spent_usd(verdicts) == pytest.approx(2 * COST)


def test_recorded_then_live_skips_live_when_complete() -> None:
    rows = [RecordedVerdict("q1", True, "Same.", HAIKU, rubric_hash())]

    class Exploding(FakeTransport):
        def __call__(self, params: JSONDict) -> JSONDict:
            raise AssertionError("live judge called")

    judge = RecordedThenLiveJudge(VerdictFileJudge(rows, HAIKU), ClaudeJudge(Exploding()))
    assert [v.equivalent for v in judge.judge([item(1)])] == [True]


def test_recorded_then_live_needs_one_model(transport: FakeTransport) -> None:
    with pytest.raises(ConfigError, match="same judge model"):
        RecordedThenLiveJudge(
            VerdictFileJudge([], "claude-opus-5-5"), ClaudeJudge(transport, model=HAIKU)
        )


# --- Agreement and sampling ----------------------------------------------------------------


def test_judge_vs_metric_on_a_hand_example() -> None:
    def v(i: str, eq: bool | None) -> Verdict:
        return Verdict(i, eq, "r")

    verdicts = [
        v("a", True),
        v("b", True),
        v("c", True),
        v("d", False),
        v("e", False),
        v("f", False),
        v("g", None),
    ]
    metric = {"a": True, "b": True, "c": False, "d": True, "e": False, "f": False}
    agreement = judge_vs_metric(verdicts, metric)
    assert (agreement.n, agreement.judged, agreement.failed) == (7, 6, 1)
    assert agreement.equivalent_rate == pytest.approx(0.5)
    assert (agreement.both_match, agreement.judge_only) == (2, 1)
    assert (agreement.exec_only, agreement.neither) == (1, 2)
    # p_o = 4/6, p_e = (3*3 + 3*3)/36 = 1/2, kappa = (2/3 - 1/2) / (1/2) = 1/3.
    assert agreement.kappa == pytest.approx(1 / 3)


def test_judge_vs_metric_edge_cases() -> None:
    nothing = judge_vs_metric([Verdict("a", None, "refusal")], {})
    assert (nothing.judged, nothing.failed, nothing.kappa) == (0, 1, None)
    assert math.isnan(nothing.equivalent_rate)
    assert judge_vs_metric([Verdict("a", True, "r")], {"a": True}).kappa is None
    with pytest.raises(ValueError, match="no metric value"):
        judge_vs_metric([Verdict("a", True, "r")], {})


def test_judge_sample_is_seeded_and_ordered() -> None:
    ids = [f"id{i:03d}" for i in range(500)]
    a = judge_sample(ids, 100, seed=7)
    assert a == judge_sample(ids, 100, seed=7)
    assert a != judge_sample(ids, 100, seed=8)
    assert len(a) == len(set(a)) == 100
    assert a == [i for i in ids if i in set(a)]  # input order
    assert judge_sample(ids[:5], 100, seed=7) == ids[:5]
    assert judge_sample(ids, 0, seed=7) == []
    with pytest.raises(ValueError, match="non-negative"):
        judge_sample(ids, -1, seed=7)
