"""Chat-template alignment: targets, completions, masking, end-of-turn and the drop report."""

from __future__ import annotations

import json

import pytest
from hypothesis import given
from hypothesis import strategies as st
from transformers import PreTrainedTokenizerFast

from anvil.chat import (
    DROP_REASONS,
    IGNORE,
    MAX_DROPPED_RATE,
    ChatReport,
    Example,
    build_example,
    build_examples,
    end_of_turn_id,
    messages,
    parse_completion,
    prompt_ids,
    render_target,
)
from anvil.errors import AlignmentError, StudentError
from anvil.students.generator import build_tiny_chat_tokenizer

SYSTEM = "Rewrite the question as SQL."
TOK = build_tiny_chat_tokenizer()
END = end_of_turn_id(TOK)

NAMES = st.lists(st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,8}", fullmatch=True), unique=True)


def _tokenizer(template: str) -> PreTrainedTokenizerFast:
    tok = build_tiny_chat_tokenizer()
    tok.chat_template = template
    return tok


def _dropped(**counts: int) -> dict[str, int]:
    return {**dict.fromkeys(DROP_REASONS, 0), **counts}


# --- targets and completions --------------------------------------------------------------


def test_messages_layout() -> None:
    assert messages("s", "u") == [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    assert messages("s", "u", "a")[-1] == {"role": "assistant", "content": "a"}


def test_one_field_target_is_the_bare_value() -> None:
    assert render_target({"sql": 'SELECT "a b" FROM t'}, ["sql"]) == 'SELECT "a b" FROM t'


def test_one_field_target_is_stripped() -> None:
    # A leading newline would merge with the template's own into one token on a BPE
    # tokenizer like Qwen's, so the generation prompt would no longer be a token prefix.
    assert render_target({"sql": "\nSELECT 1  \n"}, ["sql"]) == "SELECT 1"
    assert render_target({"a": " x ", "b": "y"}, ["a", "b"]) == '{"a":" x ","b":"y"}'


def test_several_fields_are_canonical_json_in_spec_order() -> None:
    fields = {"b": "2", "a": "é"}
    assert render_target(fields, ["a", "b"]) == '{"a":"é","b":"2"}'
    assert render_target(fields, ["b", "a"]) == '{"b":"2","a":"é"}'


def test_parse_one_field_strips() -> None:
    assert parse_completion("  SELECT 1 \n", ["sql"]) == {"sql": "SELECT 1"}
    assert parse_completion("", ["sql"]) == {"sql": ""}


@pytest.mark.parametrize(
    "text",
    [
        '{"a": "1"}',  # a field missing
        '{"a": "1", "b": "2", "c": "3"}',  # an extra field
        '{"a": "1", "c": "2"}',  # a wrong key
        '{"a": 1, "b": "2"}',  # not a string
        '{"a": null, "b": "2"}',
        '["1", "2"]',  # not an object
        '"a"',
        "SELECT 1",  # not JSON
        '{"a": "1", "b": "2"',  # cut short
    ],
)
def test_parse_several_fields_rejects_non_answers(text: str) -> None:
    assert parse_completion(text, ["a", "b"]) is None


def test_parse_several_fields_accepts_any_order_and_spacing() -> None:
    assert parse_completion(' {"b": "2",  "a": "1"} ', ["a", "b"]) == {"a": "1", "b": "2"}


@given(names=NAMES.filter(lambda n: len(n) >= 2), data=st.data())
def test_multi_field_round_trip(names: list[str], data: st.DataObject) -> None:
    fields = {n: data.draw(st.text(max_size=20)) for n in names}
    assert parse_completion(render_target(fields, names), names) == fields


@given(value=st.text(max_size=40).filter(lambda v: v == v.strip()))
def test_one_field_round_trip(value: str) -> None:
    assert parse_completion(render_target({"sql": value}, ["sql"]), ["sql"]) == {"sql": value}


# --- building training sequences ----------------------------------------------------------


def _example(text: str, target: str, max_length: int = 1000) -> Example:
    built = build_example(TOK, SYSTEM, text, target, max_length, END)
    assert isinstance(built, Example), built
    return built


def test_end_of_turn_is_im_end() -> None:
    assert TOK.convert_tokens_to_ids("<|im_end|>") == END


def test_prompt_is_masked_and_only_the_answer_and_end_are_labeled() -> None:
    text, target = "how many users are there?", "SELECT COUNT(*) FROM users"
    ex = _example(text, target)
    prompt = prompt_ids(TOK, SYSTEM, text)
    assert ex.input_ids[: len(prompt)] == prompt
    assert ex.labels[: len(prompt)] == [IGNORE] * len(prompt)
    answer = ex.labels[len(prompt) :]
    assert IGNORE not in answer
    assert answer == ex.input_ids[len(prompt) :]
    # The template writes "\n" after <|im_end|>; it is never trained.
    assert ex.input_ids[-1] == END
    assert ex.labels[-1] == END
    assert answer.count(END) == 1
    assert answer[:-1] == TOK.encode(target, add_special_tokens=False)
    assert TOK.decode(answer[:-1]) == target
    assert ex.answer_tokens == len(answer) == len(target.encode()) + 1
    assert len(ex.input_ids) == len(ex.labels)


@pytest.mark.parametrize("target", ["café crème", "選択 * から 表", "SELECT '🦀' AS x", "a\tb\nc"])
def test_unicode_round_trips_through_the_byte_tokenizer(target: str) -> None:
    ex = _example("question ünïcode 日本", target)
    answer = [t for t in ex.labels if t != IGNORE]
    assert TOK.decode(answer[:-1]) == target
    assert ex.answer_tokens == len(target.encode("utf-8")) + 1


def test_too_long_is_dropped_exactly_past_max_length() -> None:
    text, target = "question", "SELECT 1"
    length = len(_example(text, target).input_ids)
    assert isinstance(build_example(TOK, SYSTEM, text, target, length, END), Example)
    assert build_example(TOK, SYSTEM, text, target, length - 1, END) == "too_long"


def test_build_examples_reports_drops_with_ids() -> None:
    texts = ["short", "a much longer question " * 5, "short too"]
    targets = ["SELECT 1", "SELECT 2", "SELECT 3"]
    limit = max(
        len(_example(t, a).input_ids) for t, a in zip(texts[::2], targets[::2], strict=True)
    )
    built, report = build_examples(TOK, SYSTEM, texts, targets, limit, END, ["x", "y", "z"])
    assert [b is None for b in built] == [False, True, False]
    assert report.examples == 3
    assert report.dropped == _dropped(too_long=1)
    assert report.dropped_ids == ["y"]
    assert report.unusable == 1
    _, unnamed = build_examples(TOK, SYSTEM, texts, targets, limit, END)
    assert unnamed.dropped_ids == ["1"]
    with pytest.raises(StudentError, match="2 texts but 3 targets"):
        build_examples(TOK, SYSTEM, texts[:2], targets, limit, END)


def test_longest_counts_the_full_sequence_of_a_dropped_example() -> None:
    text, target = "question", "SELECT a_long_column_name FROM a_long_table_name"
    length = len(_example(text, target).input_ids)
    _, report = build_examples(TOK, SYSTEM, [text], [target], length - 1, END)
    assert report.dropped["too_long"] == 1
    # The error message tells the user to raise max_length to this; it must suffice.
    assert report.longest == length


# --- unusable templates -------------------------------------------------------------------

NOT_PREFIX = (
    "{% for m in messages %}"
    "{% if m['role'] == 'assistant' %}"
    "{{ '<|im_start|>assistant: ' + m['content'] + '<|im_end|>\\n' }}"
    "{% else %}{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '<|im_end|>\\n' }}"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
NO_END = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\\n' + m['content'] + '\\n\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
TRIMS = (
    "{% for m in messages %}"
    "{{ '<|im_start|>' + m['role'] + '\\n' + (m['content'] | trim) + '<|im_end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)


def test_a_generation_prompt_that_is_not_a_prefix_drops_the_example() -> None:
    tok = _tokenizer(NOT_PREFIX)
    assert build_example(tok, SYSTEM, "q", "SELECT 1", 1000, END) == "not_prefix"
    # A template that does it to every answer fails the gate, with the reason named.
    _, report = build_examples(tok, SYSTEM, ["q", "r"], ["SELECT 1", "SELECT 2"], 1000, END)
    assert report.dropped == _dropped(not_prefix=2)
    with pytest.raises(AlignmentError, match="merges the end of the prompt"):
        report.check()


def test_a_merge_at_the_boundary_drops_only_that_answer() -> None:
    # A tokenizer whose merges join the template's newline with an answer's leading one
    # (Qwen's `\n\n` token): the answer with the leading newline is the one dropped.
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    vocab["ĊĊ"] = len(vocab)
    inner = Tokenizer(models.BPE(vocab=vocab, merges=[("Ċ", "Ċ")]))
    inner.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    inner.decoder = decoders.ByteLevel()
    inner.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    tok = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=inner,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    tok.chat_template = TOK.chat_template
    end = end_of_turn_id(tok)
    built, report = build_examples(tok, SYSTEM, ["q", "r"], ["SELECT 1", "\nSELECT 2"], 1000, end)
    assert isinstance(built[0], Example)
    assert built[1] is None
    assert report.dropped == _dropped(not_prefix=1)


def test_an_answer_longer_than_max_new_tokens_is_dropped() -> None:
    short = build_example(TOK, SYSTEM, "q", "SELECT 1", 1000, END, max_answer=9)
    assert isinstance(short, Example)
    assert short.answer_tokens == 9  # 8 bytes and the end token
    assert build_example(TOK, SYSTEM, "q", "SELECT 1", 1000, END, max_answer=8) == (
        "answer_too_long"
    )
    _, report = build_examples(
        TOK, SYSTEM, ["q", "r"], ["SELECT 1", "SELECT 10"], 1000, END, max_answer=9
    )
    assert report.dropped == _dropped(answer_too_long=1)
    assert report.longest_answer == 10
    with pytest.raises(AlignmentError, match=r"raise student\.max_new_tokens .*10 tokens"):
        report.check(max_rate=0.1)


def test_a_template_without_an_end_token_is_refused() -> None:
    with pytest.raises(StudentError, match="special token"):
        end_of_turn_id(_tokenizer(NO_END))


def test_an_answer_without_the_given_end_id_is_refused() -> None:
    pad = TOK.pad_token_id
    assert isinstance(pad, int)
    with pytest.raises(StudentError, match="end-of-turn"):
        build_example(TOK, SYSTEM, "q", "SELECT 1", 1000, pad)


def test_a_trimming_template_does_not_round_trip() -> None:
    tok = _tokenizer(TRIMS)
    end = end_of_turn_id(tok)
    assert isinstance(build_example(tok, SYSTEM, "q", "SELECT 1", 1000, end), Example)
    assert build_example(tok, SYSTEM, "q", "SELECT 1  ", 1000, end) == "not_roundtrip"
    _, report = build_examples(tok, SYSTEM, ["q", "r"], ["SELECT 1", " x"], 1000, end)
    assert report.dropped == _dropped(not_roundtrip=1)


# --- the report ---------------------------------------------------------------------------


def _report(examples: int, too_long: int, not_roundtrip: int = 0) -> ChatReport:
    report = ChatReport(examples=examples, longest=77)
    report.dropped["too_long"] = too_long
    report.dropped["not_roundtrip"] = not_roundtrip
    return report


def test_check_allows_up_to_the_limit() -> None:
    assert MAX_DROPPED_RATE == 0.05
    _report(100, 5).check()
    _report(100, 3, 2).check()
    _report(20, 1).check()
    ChatReport().check()  # nothing aligned, nothing to refuse


def test_check_refuses_above_the_limit() -> None:
    with pytest.raises(AlignmentError, match=r"6 of 100 examples .*too_long 4, not_roundtrip 2"):
        _report(100, 4, 2).check()
    with pytest.raises(AlignmentError, match="longest sequence is 77 tokens"):
        _report(10, 1).check()
    _report(10, 1).check(max_rate=0.1)


def test_as_dict() -> None:
    report = _report(40, 2)
    report.dropped_ids = [str(i) for i in range(30)]
    d = report.as_dict()
    assert set(d) == {
        "examples",
        "dropped",
        "dropped_rate",
        "longest",
        "longest_answer",
        "dropped_ids",
    }
    assert d["examples"] == 40
    assert d["dropped"] == _dropped(too_long=2)
    assert d["dropped_rate"] == pytest.approx(0.05)
    assert d["longest"] == 77
    assert d["dropped_ids"] == [str(i) for i in range(20)]  # capped for the log
    json.dumps(d)
    assert tuple(report.dropped) == DROP_REASONS
