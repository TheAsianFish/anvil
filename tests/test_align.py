"""Alignment: teacher mentions verified against the input, then snapped to student tokens."""

from __future__ import annotations

import itertools
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from anvil.align import (
    MATCH_RULES,
    MAX_UNALIGNED_RATE,
    AlignmentReport,
    align_all,
    align_mentions,
    snap_to_tokens,
    tag_names,
    tags_to_spans,
)
from anvil.errors import AlignmentError, ConfigError
from anvil.spans import Mention, Span

TYPES = ["Dish", "Cuisine", "Location", "Restaurant_Name"]


def _texts(text: str, spans: list[Span]) -> list[tuple[str, str]]:
    return [(s.text_in(text), s.type) for s in spans]


def _one(text: str, mention: Mention) -> tuple[Span | None, AlignmentReport]:
    aligned = align_mentions(text, [mention], TYPES)
    return (aligned.spans[0] if aligned.spans else None), aligned.report


def _rule(report: AlignmentReport) -> str:
    (rule,) = [k for k, v in report.matched.items() if v]
    return rule


# --- off by one -------------------------------------------------------------------------


def test_exact_offset_is_kept() -> None:
    span, report = _one("cheap pizza", Mention("pizza", "Dish", 6))
    assert span == Span(6, 11, "Dish")
    assert _rule(report) == "exact"
    assert report.repaired == 0


@pytest.mark.parametrize("claimed", [5, 7, 4, 8, 0, 100, -3])
def test_offset_off_by_some_is_repaired_to_the_only_occurrence(claimed: int) -> None:
    span, report = _one("cheap pizza", Mention("pizza", "Dish", claimed))
    assert span == Span(6, 11, "Dish")
    assert _rule(report) == "shifted"
    assert report.unaligned == 0


def test_off_by_one_picks_the_nearest_of_two_occurrences() -> None:
    text = "pizza or pizza"  # occurrences at 0 and 9
    assert _one(text, Mention("pizza", "Dish", 8))[0] == Span(9, 14, "Dish")
    assert _one(text, Mention("pizza", "Dish", 1))[0] == Span(0, 5, "Dish")


def test_two_mentions_of_a_repeated_word_claim_two_occurrences() -> None:
    text = "pizza or pizza"
    aligned = align_mentions(
        text, [Mention("pizza", "Dish", 1), Mention("pizza", "Dish", 2)], TYPES
    )
    assert aligned.spans == [Span(0, 5, "Dish"), Span(9, 14, "Dish")]
    assert aligned.report.unaligned == 0


def test_identical_repeat_is_a_duplicate_not_an_error() -> None:
    m = Mention("pizza", "Dish", 6)
    aligned = align_mentions("cheap pizza", [m, m], TYPES)
    assert aligned.spans == [Span(6, 11, "Dish")]
    assert aligned.report.duplicates == 1
    assert aligned.report.counted == 1
    assert aligned.report.unaligned_rate == 0


def test_off_by_one_end_from_trailing_punctuation_is_trimmed() -> None:
    span, report = _one("is it open late?", Mention("open late?!", "Location", 6))
    assert span == Span(6, 15, "Location")
    assert _rule(report) == "trimmed"


def test_case_mismatch_is_repaired() -> None:
    span, report = _one("find burger king near me", Mention("Burger King", "Restaurant_Name", 5))
    assert span == Span(5, 16, "Restaurant_Name")
    assert _rule(report) == "case"


def test_whitespace_mismatch_is_repaired() -> None:
    text = "the  pizza\nhut please"
    span, report = _one(text, Mention("pizza hut", "Restaurant_Name", 5))
    assert span is not None
    assert span.text_in(text) == "pizza\nhut"
    assert _rule(report) == "spacing"


def test_text_not_in_the_input_is_dropped() -> None:
    span, report = _one("cheap pizza", Mention("sushi", "Dish", 6))
    assert span is None
    assert report.dropped["not_found"] == 1
    assert report.unaligned_rate == 1.0
    assert report.examples[0]["outcome"] == "not_found"


@pytest.mark.parametrize("mention", [Mention("  ", "Dish", 0), Mention("pizza", "Pasta", 6)])
def test_empty_text_or_unknown_type_is_invalid(mention: Mention) -> None:
    span, report = _one("cheap pizza", mention)
    assert span is None
    assert report.dropped["invalid"] == 1


# --- overlapping ------------------------------------------------------------------------


def test_nested_spans_keep_the_longer() -> None:
    text = "the pizza hut downtown"
    aligned = align_mentions(
        text, [Mention("pizza", "Dish", 4), Mention("pizza hut", "Restaurant_Name", 4)], TYPES
    )
    assert _texts(text, aligned.spans) == [("pizza hut", "Restaurant_Name")]
    assert aligned.report.dropped["overlap"] == 1


def test_crossing_spans_keep_the_longer_then_the_earlier() -> None:
    text = "new york ave pizza"
    aligned = align_mentions(
        text,
        [Mention("york ave", "Location", 4), Mention("new york", "Location", 0)],
        TYPES,
    )
    assert _texts(text, aligned.spans) == [("new york", "Location")]  # same length: earlier
    assert aligned.report.dropped["overlap"] == 1


def test_same_extent_two_types_keeps_the_type_listed_first() -> None:
    text = "some thai food"
    aligned = align_mentions(
        text, [Mention("thai", "Cuisine", 5), Mention("thai", "Dish", 5)], TYPES
    )
    assert aligned.spans == [Span(5, 9, "Dish")]  # Dish precedes Cuisine in TYPES
    assert aligned.report.dropped["overlap"] == 1


def test_a_second_type_on_the_same_text_is_not_pushed_to_another_occurrence() -> None:
    text = "thai or thai"
    aligned = align_mentions(
        text, [Mention("thai", "Dish", 0), Mention("thai", "Cuisine", 0)], TYPES
    )
    assert aligned.spans == [Span(0, 4, "Dish")]
    assert aligned.report.dropped["overlap"] == 1


def test_adjacent_spans_do_not_overlap() -> None:
    text = "pizzapasta"
    aligned = align_mentions(
        text, [Mention("pizza", "Dish", 0), Mention("pasta", "Dish", 5)], TYPES
    )
    assert aligned.spans == [Span(0, 5, "Dish"), Span(5, 10, "Dish")]


# --- unicode ----------------------------------------------------------------------------


def test_offsets_are_code_points_not_bytes() -> None:
    text = "crème brûlée in zürich"
    start = text.index("zürich")
    span, report = _one(text, Mention("zürich", "Location", start))
    assert span == Span(start, start + 6, "Location")
    assert _rule(report) == "exact"
    utf8_start = len(text[:start].encode("utf-8"))  # a teacher counting bytes
    span, report = _one(text, Mention("zürich", "Location", utf8_start))
    assert span == Span(start, start + 6, "Location")
    assert _rule(report) == "shifted"


def test_utf16_offsets_after_an_emoji_are_repaired() -> None:
    text = "🍕 pizza 🍕 near tokyo"
    start = text.index("tokyo")
    utf16_start = len(text[:start].encode("utf-16-le")) // 2  # each emoji is two units
    assert utf16_start == start + 2
    span, _ = _one(text, Mention("tokyo", "Location", utf16_start))
    assert span == Span(start, start + 5, "Location")


def test_decomposed_input_matches_a_precomposed_mention() -> None:
    text = unicodedata.normalize("NFD", "a café near me")  # "e" + combining acute
    assert len(text) == 15
    span, report = _one(text, Mention("café", "Location", 2))
    assert span is not None
    assert unicodedata.normalize("NFC", span.text_in(text)) == "café"
    assert span.end - span.start == 5  # the combining mark is inside the span
    assert _rule(report) == "normalized"


def test_normalized_match_never_splits_a_letter_from_its_accent() -> None:
    text = unicodedata.normalize("NFD", "cafés")
    span, _ = _one(text, Mention("cafe", "Dish", 0))
    # "cafe" matches only by dropping the accent, which normalisation does not do.
    assert span is None


def test_non_latin_scripts_align() -> None:
    text = "東京で寿司を食べたい"
    span, report = _one(text, Mention("寿司", "Dish", 4))  # true start 3
    assert span == Span(3, 5, "Dish")
    assert _rule(report) == "shifted"


# --- reports and the gate ---------------------------------------------------------------


def test_reports_merge_and_gate() -> None:
    rows = [
        ("a", "cheap pizza", [Mention("pizza", "Dish", 6)]),
        ("b", "sushi bar", [Mention("ramen", "Dish", 0)]),
    ]
    spans, report = align_all(rows, TYPES)
    assert spans == [[Span(6, 11, "Dish")], []]
    assert report.mentions == 2
    assert report.unaligned == 1
    assert report.unaligned_rate == 0.5
    with pytest.raises(AlignmentError) as info:
        report.check()
    assert isinstance(info.value, ConfigError)  # a labels/prompt problem: exit code 2
    assert "not_found 1" in str(info.value)
    as_dict = report.as_dict()
    assert as_dict["unaligned_rate"] == 0.5
    assert as_dict["matched"] == {**dict.fromkeys(MATCH_RULES, 0), "exact": 1}


def test_gate_passes_at_the_limit() -> None:
    report = AlignmentReport(mentions=100)
    report.dropped["not_found"] = int(100 * MAX_UNALIGNED_RATE)
    report.check()
    report.dropped["not_found"] += 1
    with pytest.raises(AlignmentError):
        report.check()


# --- snapping to tokens -----------------------------------------------------------------


def _word_offsets(text: str) -> list[tuple[int, int]]:
    """A whitespace tokenizer's offset mapping with [CLS] and [SEP] around it."""
    out = [(0, 0)]
    i = 0
    for word in text.split(" "):
        out.append((i, i + len(word)))
        i += len(word) + 1
    return [*out, (0, 0)]


def test_snap_gives_bio_and_none_for_special_tokens() -> None:
    text = "book the pizza hut in new york"
    spans = [Span(9, 18, "Restaurant_Name"), Span(22, 30, "Location")]
    snapped = snap_to_tokens(text, spans, _word_offsets(text))
    assert snapped.tags == [
        None,
        "O",
        "O",
        "B-Restaurant_Name",
        "I-Restaurant_Name",
        "O",
        "B-Location",
        "I-Location",
        None,
    ]
    assert snapped.spans == spans
    assert snapped.report.snapped == 0


def test_snap_grows_a_span_that_ends_inside_a_token() -> None:
    text = "two pizzas please"
    snapped = snap_to_tokens(text, [Span(4, 9, "Dish")], _word_offsets(text))  # "pizza"
    assert snapped.spans == [Span(4, 10, "Dish")]  # "pizzas"
    assert snapped.report.snapped == 1
    assert snapped.report.unaligned == 0


def test_snap_drops_a_span_cut_off_by_max_length() -> None:
    text = "pizza near tokyo"
    offsets = [*_word_offsets(text)[:3], (0, 0)]  # truncated after "near"
    snapped = snap_to_tokens(text, [Span(0, 5, "Dish"), Span(11, 16, "Location")], offsets)
    assert snapped.spans == [Span(0, 5, "Dish")]
    assert snapped.report.dropped["no_token"] == 1


def test_snap_drops_a_second_span_sharing_a_token() -> None:
    text = "pizzapasta tonight"
    offsets = [(0, 0), (0, 10), (11, 18), (0, 0)]  # one token for both dishes
    snapped = snap_to_tokens(text, [Span(0, 5, "Dish"), Span(5, 10, "Dish")], offsets)
    assert snapped.spans == [Span(0, 10, "Dish")]
    assert snapped.report.dropped["token_conflict"] == 1
    assert snapped.report.snapped == 1


def test_snap_strips_the_space_byte_level_tokens_carry() -> None:
    text = "cheap pizza"
    offsets = [(0, 0), (0, 5), (5, 11), (0, 0)]  # " pizza" includes the space, as in BPE
    snapped = snap_to_tokens(text, [Span(6, 11, "Dish")], offsets)
    assert snapped.tags[2] == "B-Dish"
    assert snapped.spans == [Span(6, 11, "Dish")]
    assert snapped.report.snapped == 0


def test_snap_report_merges_without_recounting_mentions() -> None:
    text = "pizza near tokyo"
    aligned = align_mentions(text, [Mention("tokyo", "Location", 11)], TYPES)
    offsets = [*_word_offsets(text)[:3], (0, 0)]
    snapped = snap_to_tokens(text, aligned.spans, offsets)
    report = aligned.report
    report.merge(snapped.report)
    assert report.mentions == 1
    assert report.unaligned == 1
    assert report.unaligned_rate == 1.0


# --- decoding predictions ---------------------------------------------------------------


def test_tags_to_spans_follows_conll_chunking() -> None:
    text = "a b c d e f"
    offsets = _word_offsets(text)
    tags: list[str | None] = [
        None,
        "B-Dish",
        "I-Dish",
        "I-Location",  # another type: starts a chunk
        "O",
        "I-Dish",  # I after O: starts a chunk
        "B-Dish",  # B right after: a separate chunk of the same type
        None,
    ]
    assert tags_to_spans(text, tags, offsets) == [
        Span(0, 3, "Dish"),
        Span(4, 5, "Location"),
        Span(8, 9, "Dish"),
        Span(10, 11, "Dish"),
    ]


def test_tags_to_spans_rejects_a_length_mismatch() -> None:
    with pytest.raises(ValueError, match="2 tokens"):
        tags_to_spans("a", ["O"], [(0, 0), (0, 1)])


def test_tag_names() -> None:
    assert tag_names(["Dish", "Location"]) == ["O", "B-Dish", "I-Dish", "B-Location", "I-Location"]


# --- properties -------------------------------------------------------------------------

WORDS = ["pizza", "thai", "near", "me", "open", "late", "cheap", "sushi", "bar", "la"]


@st.composite
def _sentences(draw: st.DrawFn) -> tuple[str, list[Span]]:
    """A sentence of distinct words and non-overlapping word-aligned spans over them."""
    words = draw(st.lists(st.sampled_from(WORDS), min_size=1, max_size=8, unique=True))
    text = " ".join(words)
    starts = [sum(len(w) + 1 for w in words[:i]) for i in range(len(words))]
    spans: list[Span] = []
    i = 0
    while i < len(words):
        if draw(st.booleans()):
            n = draw(st.integers(1, min(3, len(words) - i)))
            end = starts[i + n - 1] + len(words[i + n - 1])
            spans.append(Span(starts[i], end, draw(st.sampled_from(TYPES))))
            i += n
        else:
            i += 1
    return text, spans


@settings(max_examples=200, deadline=None)
@given(data=_sentences(), shifts=st.lists(st.integers(-2, 2), min_size=8, max_size=8))
def test_round_trip_and_shift_repair(data: tuple[str, list[Span]], shifts: list[int]) -> None:
    text, spans = data
    exact = [Mention(s.text_in(text), s.type, s.start) for s in spans]
    aligned = align_mentions(text, exact, TYPES)
    assert aligned.spans == spans
    assert aligned.report.matched["exact"] == len(spans)
    assert aligned.report.unaligned == 0

    # Words are distinct, so every span's text occurs once and any offset error is repaired.
    noisy = [Mention(m.text, m.type, m.start + d) for m, d in zip(exact, shifts, strict=False)]
    assert align_mentions(text, noisy, TYPES).spans == spans

    snapped = snap_to_tokens(text, aligned.spans, _word_offsets(text))
    assert snapped.spans == spans
    assert tags_to_spans(text, snapped.tags, _word_offsets(text)) == spans


@settings(max_examples=200, deadline=None)
@given(
    text=st.text(min_size=0, max_size=30),
    mentions=st.lists(
        st.builds(
            Mention,
            text=st.text(min_size=0, max_size=6),
            type=st.sampled_from([*TYPES, "Bogus"]),
            start=st.integers(-5, 40),
        ),
        max_size=6,
    ),
)
def test_any_answer_aligns_to_valid_non_overlapping_spans(
    text: str, mentions: list[Mention]
) -> None:
    aligned = align_mentions(text, mentions, TYPES)
    spans = aligned.spans
    assert spans == sorted(spans)
    for a, b in itertools.pairwise(spans):
        assert a.end <= b.start
    for span in spans:
        assert 0 <= span.start < span.end <= len(text)
        assert span.type in TYPES
    report = aligned.report
    assert len(spans) == sum(report.matched.values()) - report.dropped["overlap"]
    assert report.mentions == len(mentions)
    assert sum(report.matched.values()) + report.duplicates + report.dropped["not_found"] + (
        report.dropped["invalid"]
    ) == len(mentions)


def test_exact_only_rules_verify_without_repairing() -> None:
    text = "cheap pizza"
    strict = align_mentions(text, [Mention("pizza", "Dish", 7)], TYPES, rules=("exact",))
    assert strict.spans == []
    assert strict.report.dropped["not_found"] == 1
    with pytest.raises(ValueError, match="must include 'exact'"):
        align_mentions(text, [], TYPES, rules=("shifted",))


def test_snap_tags_a_whitespace_token_inside_an_entity() -> None:
    text = "the pizza  hut"
    # Byte-level BPE: "Ġthe" (0, 3), "Ġpizza" (3, 9), a token for the extra space, "Ġhut".
    offsets = [(0, 0), (0, 3), (3, 9), (9, 10), (10, 14), (0, 0)]
    snapped = snap_to_tokens(text, [Span(4, 14, "Restaurant_Name")], offsets)
    assert snapped.tags[2:5] == ["B-Restaurant_Name", "I-Restaurant_Name", "I-Restaurant_Name"]
    assert tags_to_spans(text, snapped.tags, offsets) == [Span(4, 14, "Restaurant_Name")]


@pytest.mark.parametrize("raw", [Mention("pizza ", "Dish", 7), Mention(" pizza", "Dish", 6)])
def test_edge_whitespace_never_enters_a_span(raw: Mention) -> None:
    span, report = _one("i want pizza hut", raw)
    assert span == Span(7, 12, "Dish")
    assert _rule(report) == "exact"


def test_one_entity_listed_twice_with_different_offsets_is_a_duplicate() -> None:
    aligned = align_mentions(
        "i want pizza", [Mention("pizza", "Dish", 7), Mention("pizza", "Dish", 6)], TYPES
    )
    assert aligned.spans == [Span(7, 12, "Dish")]
    assert aligned.report.duplicates == 1
    assert aligned.report.unaligned == 0


def test_a_report_that_counts_more_drops_than_mentions_is_refused() -> None:
    report = AlignmentReport(mentions=1)
    report.dropped["no_token"] = 2
    with pytest.raises(ValueError, match="merged twice"):
        report.check()


# --- the cases a review of the ladder turned up --------------------------------------------


def test_normalized_match_survives_an_expanding_normalisation() -> None:
    # A decomposed Hangul syllable is three jamo; NFC makes it one code point, so the
    # decomposed text is far longer than the mention it has to match.
    needle = "한국어는요"
    text = "저는 " + unicodedata.normalize("NFD", needle) + " 공부"
    span, report = _one(text, Mention(needle, "Dish", 3))
    assert span == Span(3, 3 + len(unicodedata.normalize("NFD", needle)), "Dish")
    assert _rule(report) == "normalized"


def test_shifted_match_prefers_a_whole_word_over_a_substring() -> None:
    text = "fine dining steakhouse in the area"  # "in" sits inside "fine" and "dining"
    span, report = _one(text, Mention("in", "Location", 0))
    assert span == Span(23, 25, "Location")
    assert _rule(report) == "shifted"
    # When the mention occurs only inside a word, that occurrence is still better than none.
    span, _ = _one("banana", Mention("an", "Dish", 50))
    assert span == Span(3, 5, "Dish")  # nearest to the claimed offset


def test_identical_mentions_take_two_occurrences_when_the_text_has_two() -> None:
    text = "Pizza Hut beats Pizza Hut Express"
    m = Mention("Pizza Hut", "Restaurant_Name", 0)
    aligned = align_mentions(text, [m, m], TYPES)
    assert aligned.spans == [Span(0, 9, "Restaurant_Name"), Span(16, 25, "Restaurant_Name")]
    assert aligned.report.matched["exact"] == 1
    assert aligned.report.matched["shifted"] == 1
    assert aligned.report.duplicates == 0


def test_trimmed_mention_goes_down_the_whole_ladder() -> None:
    text = "the Burger  King here"  # quotes to trim, then case and a double space
    span, report = _one(text, Mention('"burger king"', "Restaurant_Name", 4))
    assert span == Span(4, 16, "Restaurant_Name")
    assert _rule(report) == "trimmed"


def test_snap_keeps_a_trailing_combining_mark_with_its_letter() -> None:
    # A tokenizer that normalises its input leaves the accent of a decomposed "café"
    # outside every token: "a cafe\u0301 near me" tokenizes to " cafe" (1, 6), " near" (7, 12).
    text = "a cafe\u0301 near me"
    offsets = [(0, 0), (0, 1), (1, 6), (7, 12), (13, 15), (0, 0)]
    snapped = snap_to_tokens(text, [Span(2, 7, "Location")], offsets)
    assert snapped.spans == [Span(2, 7, "Location")]
    assert snapped.report.snapped == 0
    assert tags_to_spans(text, snapped.tags, offsets) == [Span(2, 7, "Location")]


def test_a_span_the_token_limit_cuts_through_is_dropped_not_snapped() -> None:
    text = "aaa bbb ccc ddd eee"
    offsets = [(0, 0), (0, 3), (3, 7), (7, 11), (0, 0)]  # max_length stopped after "ccc"
    snapped = snap_to_tokens(text, [Span(4, 15, "Dish")], offsets)
    assert snapped.spans == []
    assert snapped.report.dropped["truncated"] == 1
    assert snapped.report.snapped == 0
    assert all(tag in (None, "O") for tag in snapped.tags)
    # A span entirely past the cut is `no_token`, one that ends at the cut is whole.
    both = snap_to_tokens(text, [Span(4, 11, "Dish"), Span(12, 15, "Dish")], offsets)
    assert both.spans == [Span(4, 11, "Dish")]
    assert both.report.dropped["no_token"] == 1
    assert both.report.dropped["truncated"] == 0
