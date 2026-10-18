"""The encoder tagger: offsets, BIO targets, training, confidence, save/load and resume."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import PreTrainedTokenizerFast

from anvil.errors import StudentError
from anvil.evaluate.extraction import entity_scores
from anvil.spans import Span
from anvil.spec import StudentSpec
from anvil.students.tagger import IGNORE, META_FILE, EncoderTagger
from anvil.train.classifier import validation_split
from anvil.train.tagger import train_tagger

TYPES = ["Restaurant_Name", "Location"]
NAMES = ["the golden fork", "sushi palace", "luigi's", "blue door bistro", "taco town", "olive"]
CITIES = ["paris", "new york", "san francisco", "berlin", "rome"]
TEMPLATES = ["book a table at {name} in {city}", "is {name} in {city} open tonight"]
SPEC = StudentSpec(
    model="tiny", max_length=24, epochs=12, batch_size=8, learning_rate=2e-3,
    validation_fraction=0.25, patience=3,
)  # fmt: skip


class SimulatedCrashError(Exception):
    pass


def _toy_set() -> tuple[list[str], list[list[Span]]]:
    """60 distinct sentences, each with one restaurant and one location."""
    texts: list[str] = []
    spans: list[list[Span]] = []
    for template in TEMPLATES:
        for name in NAMES:
            for city in CITIES:
                text = template.format(name=name, city=city)
                a = text.index(name)
                b = text.index(f" in {city}") + 4
                texts.append(text)
                spans.append(
                    [Span(a, a + len(name), "Restaurant_Name"), Span(b, b + len(city), "Location")]
                )
    return texts, spans


def _tagger(spec: StudentSpec = SPEC) -> EncoderTagger:
    texts, _ = _toy_set()
    return EncoderTagger.build(spec, TYPES, texts)


def test_tiny_tagger_builds_with_the_bio_alphabet() -> None:
    tagger = _tagger()
    assert tagger.tags == [
        "O", "B-Restaurant_Name", "I-Restaurant_Name", "B-Location", "I-Location",
    ]  # fmt: skip
    assert tagger.model.config.num_labels == 5
    assert tagger.model.config.id2label[1] == "B-Restaurant_Name"


def test_offsets_index_the_original_text() -> None:
    # The tiny normalizer lowercases and strips accents; offsets must still point into the
    # text as given, combining accents and all.
    text = "Book a table at  Café ZÜRICH in Néw-York!"
    tagger = EncoderTagger.build(SPEC, TYPES, [text])
    (offsets,) = tagger.offsets([text])
    assert offsets[0] == offsets[-1] == (0, 0)  # [CLS] and [SEP]
    pieces = [text[s:e] for s, e in offsets[1:-1]]
    assert pieces == ["Book", "a", "table", "at", "Café", "ZÜRICH", "in", "Néw", "-", "York", "!"]
    assert len(offsets) == tagger.encode([text])["input_ids"].shape[1]


def test_targets_tag_b_i_o_and_ignore_specials() -> None:
    tagger = _tagger()
    text = "book a table at the golden fork in new york"
    spans = [Span(16, 31, "Restaurant_Name"), Span(35, 43, "Location")]
    ids, report, snapped = tagger.targets([text], [spans])
    tags = [None if i == IGNORE else tagger.tags[i] for i in ids[0]]
    assert tags == [
        None, "O", "O", "O", "O",
        "B-Restaurant_Name", "I-Restaurant_Name", "I-Restaurant_Name",
        "O", "B-Location", "I-Location", None,
    ]  # fmt: skip
    assert snapped == [spans]
    assert report.unaligned == 0
    assert report.snapped == 0


def test_targets_snap_a_partial_word_and_count_truncated_spans() -> None:
    tagger = _tagger(SPEC.model_copy(update={"max_length": 8}))
    # 10 words: [CLS] + 6 words + [SEP] fit in 8 tokens, so "rome" (word 10) is cut off.
    text = "book a table at sushi palace in rome tonight please"
    spans = [Span(17, 26, "Restaurant_Name"), Span(32, 36, "Location")]  # "ushi pala"
    ids, report, snapped = tagger.targets([text], [spans])
    assert len(ids[0]) == 8
    assert report.dropped["no_token"] == 1
    assert report.snapped == 1
    assert snapped == [[Span(16, 28, "Restaurant_Name")]]
    assert report.as_dict()["unaligned"] == 1


def test_targets_reject_mismatched_inputs() -> None:
    tagger = _tagger()
    with pytest.raises(StudentError, match="span lists"):
        tagger.targets(["a", "b"], [[]])


def test_untrained_predictions_are_well_formed() -> None:
    texts, _ = _toy_set()
    tagger = _tagger()
    tags, probs = tagger.predict_tags(texts[:3])
    offsets = tagger.offsets(texts[:3])
    for row_tags, row_probs, row_offsets in zip(tags, probs, offsets, strict=True):
        assert len(row_tags) == len(row_probs) == len(row_offsets)
        assert row_tags[0] is None
        assert row_tags[-1] is None
        assert all(t in tagger.tags for t in row_tags[1:-1])
        assert all(0.0 < p <= 1.0 for p in row_probs)
    spans, confidence = tagger.predict_with_confidence(texts[:3])
    assert len(spans) == 3
    assert confidence.shape == (3,)
    assert confidence.dtype == np.float32
    # The confidence is the weakest real token's probability.
    assert confidence[0] == pytest.approx(min(probs[0][1:-1]))
    empty_spans, empty_conf = tagger.predict_with_confidence([])
    assert empty_spans == []
    assert empty_conf.shape == (0,)


def test_training_learns_the_template(tmp_path: Path) -> None:
    texts, spans = _toy_set()
    tagger = _tagger()
    result, report = train_tagger(tagger, texts, spans, SPEC, tmp_path / "m.jsonl")
    _, val = validation_split(texts, SPEC.validation_fraction, SPEC.seed)
    assert result.validation_examples == len(val) >= 10
    assert result.examples + result.validation_examples == len(texts)
    assert result.epoch_losses[-1] < result.epoch_losses[0] / 10
    assert result.best_metric is not None
    assert result.best_metric >= 0.9
    assert report.unaligned == 0
    assert entity_scores(spans, tagger.predict_spans(texts)).f1 >= 0.9
    _, confidence = tagger.predict_with_confidence(texts)
    assert float(confidence.min()) > 0.5
    epochs = [
        json.loads(line)
        for line in (tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "epoch"
    ]
    assert result.best_metric == max(r["metric"] for r in epochs)


def test_fit_keeps_the_alignment_report() -> None:
    texts, spans = _toy_set()
    spec = SPEC.model_copy(update={"epochs": 1, "validation_fraction": 0.0})
    tagger = _tagger(spec)
    before = tagger.alignment
    result = tagger.fit(texts, spans, spec)
    assert before is None
    assert result.validation_examples == 0
    assert result.best_epoch is None
    assert tagger.alignment is not None
    assert tagger.alignment.unaligned == 0


def test_a_tiny_validation_slice_skips_early_stopping() -> None:
    texts, spans = _toy_set()
    spec = SPEC.model_copy(update={"epochs": 1})
    tagger = _tagger(spec)
    result, _ = train_tagger(tagger, texts[:12], spans[:12], spec)
    assert result.validation_examples == 0
    assert result.best_epoch is None
    assert result.examples == 12


def test_train_rejects_bad_inputs() -> None:
    texts, spans = _toy_set()
    tagger = _tagger()
    with pytest.raises(StudentError, match="span lists"):
        train_tagger(tagger, texts, spans[:-1], SPEC)
    with pytest.raises(StudentError, match="no training examples"):
        train_tagger(tagger, [], [], SPEC)


def test_save_load_round_trip(tmp_path: Path) -> None:
    texts, spans = _toy_set()
    spec = SPEC.model_copy(update={"epochs": 2})
    tagger = _tagger(spec)
    tagger.fit(texts, spans, spec)
    tagger.meta["task"] = "toy"
    tagger.save(tmp_path / "saved")
    loaded = EncoderTagger.load(tmp_path / "saved")
    assert loaded.types == TYPES
    assert loaded.tags == tagger.tags
    assert loaded.max_length == spec.max_length
    assert loaded.meta == {"task": "toy"}
    assert loaded.predict_tags(texts) == tagger.predict_tags(texts)
    a_spans, a_conf = tagger.predict_with_confidence(texts)
    b_spans, b_conf = loaded.predict_with_confidence(texts)
    assert a_spans == b_spans
    assert np.array_equal(a_conf, b_conf)


def test_load_rejects_incomplete_dirs(tmp_path: Path) -> None:
    with pytest.raises(StudentError, match="not a saved Anvil tagger"):
        EncoderTagger.load(tmp_path)
    (tmp_path / META_FILE).write_text('{"types": ["a"]}', encoding="utf-8")
    with pytest.raises(StudentError, match="not a complete Anvil tagger"):
        EncoderTagger.load(tmp_path)


def test_a_slow_tokenizer_is_a_student_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PreTrainedTokenizerFast, "is_fast", property(lambda self: False))
    with pytest.raises(StudentError, match="not a fast tokenizer"):
        _tagger()


def test_pretrained_tagger_that_cannot_load_is_a_student_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(StudentError, match=r"student\.model: tiny"):
        EncoderTagger.build(StudentSpec(model="nonexistent/model"), TYPES, [])


def _crash_on(monkeypatch: pytest.MonkeyPatch, call: int) -> None:
    encode = EncoderTagger.encode
    calls = 0

    def failing_encode(self: EncoderTagger, batch: list[str]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == call:
            raise SimulatedCrashError
        return encode(self, batch)

    monkeypatch.setattr(EncoderTagger, "encode", failing_encode)


def test_resume_is_bit_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    texts, spans = _toy_set()
    spec = SPEC.model_copy(update={"epochs": 3, "grad_accum": 2, "batch_size": 4, "patience": 5})
    ref = _tagger(spec)
    ref_result = ref.fit(texts, spans, spec)

    # About 12 micro-batches of 4 per epoch plus one encode call for the validation
    # predictions, so call 20 is mid epoch 2, after the first epoch-end checkpoint.
    with monkeypatch.context() as m:
        _crash_on(m, 20)
        with pytest.raises(SimulatedCrashError):
            _tagger(spec).fit(texts, spans, spec, checkpoint_dir=tmp_path / "ckpt")
    student = _tagger(spec)
    result = student.fit(texts, spans, spec, checkpoint_dir=tmp_path / "ckpt", resume=True)
    assert result.resumed_at_step is not None
    assert result.epoch_losses == ref_result.epoch_losses
    assert result.best_epoch == ref_result.best_epoch
    for a, b in zip(ref.model.parameters(), student.model.parameters(), strict=True):
        assert torch.equal(a, b)


def _retyped(spans: list[list[Span]], mapping: dict[str, str]) -> list[list[Span]]:
    return [[Span(s.start, s.end, mapping[s.type]) for s in row] for row in spans]


def test_resume_on_other_spans_or_types_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts, spans = _toy_set()
    spec = SPEC.model_copy(update={"epochs": 3, "batch_size": 4})
    with monkeypatch.context() as m:
        _crash_on(m, 20)
        with pytest.raises(SimulatedCrashError):
            _tagger(spec).fit(texts, spans, spec, checkpoint_dir=tmp_path / "ckpt")
    # Same texts and counts, the two types swapped: other targets.
    swapped = _retyped(spans, {TYPES[0]: TYPES[1], TYPES[1]: TYPES[0]})
    with pytest.raises(StudentError, match="different training config"):
        _tagger(spec).fit(texts, swapped, spec, checkpoint_dir=tmp_path / "ckpt", resume=True)
    # The types renamed: every tag id and the head's shape are the same, the meaning is not.
    renamed = EncoderTagger.build(spec, ["Restaurant", "Place"], texts)
    retyped = _retyped(spans, {TYPES[0]: "Restaurant", TYPES[1]: "Place"})
    with pytest.raises(StudentError, match="different training config"):
        renamed.fit(texts, retyped, spec, checkpoint_dir=tmp_path / "ckpt", resume=True)
    # The original data still resumes.
    result = _tagger(spec).fit(texts, spans, spec, checkpoint_dir=tmp_path / "ckpt", resume=True)
    assert result.resumed_at_step is not None


def test_load_rejects_a_head_of_another_size(tmp_path: Path) -> None:
    tagger = _tagger()
    tagger.save(tmp_path)
    meta = json.loads((tmp_path / META_FILE).read_text(encoding="utf-8"))
    meta["types"] = [*TYPES, "Dish"]
    meta["tags"] = [*meta["tags"], "B-Dish", "I-Dish"]
    (tmp_path / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(StudentError, match="head for 5 tags but lists 7"):
        EncoderTagger.load(tmp_path)
