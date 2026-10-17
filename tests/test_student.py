from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from anvil.errors import StudentError
from anvil.spec import StudentSpec
from anvil.students.encoder import EncoderClassifier, build_tiny_tokenizer
from anvil.train.loop import TrainResult, train_classifier

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["billing", "bug", "feature", "account"]
WORDS = {
    "billing": ["refund", "invoice", "charge"],
    "bug": ["crash", "error", "broken"],
    "feature": ["wish", "add", "idea"],
    "account": ["password", "login", "email"],
}
FILLER = ["please", "help", "today", "my", "the", "app", "again", "now"]
# 48 examples in batches of 7 leave a partial last batch, so the loss weighting is exercised.
SPEC = StudentSpec(model="tiny", max_length=16, epochs=8, batch_size=7, learning_rate=3e-3)


def _toy_set() -> tuple[list[str], list[int]]:
    texts, targets = [], []
    for k in range(48):
        label = LABELS[k % 4]
        words = [WORDS[label][k % 3], FILLER[k % 8], FILLER[(k * 3) % 8]]
        texts.append(" ".join(words[(k % 3) :] + words[: (k % 3)]))
        targets.append(k % 4)
    return texts, targets


def _vocab_script(corpus: list[str]) -> str:
    return (
        "import json, sys\n"
        "from anvil.students.encoder import build_tiny_tokenizer\n"
        f"tok = build_tiny_tokenizer({corpus!r})\n"
        "print(json.dumps(sorted(tok.get_vocab().items())))\n"
    )


def test_tiny_tokenizer_is_deterministic_across_processes() -> None:
    # Many frequency ties, so any hash-order dependence would show up.
    corpus = [" ".join(f"w{(i * 7 + j) % 97}" for j in range(5)) for i in range(60)]
    vocab = build_tiny_tokenizer(corpus).get_vocab()
    assert vocab == build_tiny_tokenizer(corpus).get_vocab()
    expected = sorted(vocab.items())
    # One subprocess (importing transformers takes seconds) with a hash seed unlike ours.
    seed = "1" if os.environ.get("PYTHONHASHSEED") != "1" else "2"
    out = subprocess.run(
        [sys.executable, "-c", _vocab_script(corpus)],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": seed},
        cwd=ROOT,
    ).stdout
    assert [tuple(kv) for kv in json.loads(out)] == expected


def test_tiny_tokenizer_adds_cls_and_sep() -> None:
    tok = build_tiny_tokenizer(["hello world", "hello there"])
    ids = tok("hello world")["input_ids"]
    assert ids[0] == tok.cls_token_id
    assert ids[-1] == tok.sep_token_id
    assert tok.convert_ids_to_tokens(ids) == ["[CLS]", "hello", "world", "[SEP]"]
    assert tok.convert_ids_to_tokens(tok("hello zebra")["input_ids"])[2] == "[UNK]"


def test_encode_truncates_to_max_length() -> None:
    texts, _ = _toy_set()
    student = EncoderClassifier.build(SPEC, LABELS, texts)
    assert student.encode([" ".join(["help"] * 100)])["input_ids"].shape[1] == SPEC.max_length


def test_pretrained_student_that_cannot_load_is_a_student_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    spec = StudentSpec(model="nonexistent/model")
    with pytest.raises(StudentError, match=r"student\.model: tiny"):
        EncoderClassifier.build(spec, LABELS, [])


def test_predict_proba_shape() -> None:
    texts, _ = _toy_set()
    student = EncoderClassifier.build(SPEC, LABELS, texts)
    probs = student.predict_proba(texts[:5])
    assert probs.shape == (5, 4)
    assert probs.dtype == np.float32
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, rtol=1e-5)
    assert student.predict_proba([]).shape == (0, 4)
    assert all(p in LABELS for p in student.predict(texts[:5]))


def _train(tmp_path: Path) -> tuple[EncoderClassifier, TrainResult]:
    texts, targets = _toy_set()
    student = EncoderClassifier.build(SPEC, LABELS, texts)
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = train_classifier(student, texts, targets, SPEC, tmp_path / "metrics.jsonl")
    return student, result


def test_training_learns_a_separable_task(tmp_path: Path) -> None:
    texts, targets = _toy_set()
    student, result = _train(tmp_path)
    losses = result.epoch_losses
    assert len(losses) == SPEC.epochs == result.epochs
    assert losses[-1] < losses[0] / 2
    assert result.examples == len(texts)
    assert result.steps == SPEC.epochs * 7  # ceil(48 / 7) batches per epoch
    predicted = student.predict(texts)
    acc = np.mean([p == LABELS[t] for p, t in zip(predicted, targets, strict=True)])
    assert acc >= 0.9
    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in lines]
    assert [r["epoch"] for r in records] == list(range(1, SPEC.epochs + 1))
    assert records[-1]["steps"] == result.steps


def test_training_is_reproducible_and_saves_roundtrip(tmp_path: Path) -> None:
    texts, _ = _toy_set()
    a, ra = _train(tmp_path / "a")
    b, rb = _train(tmp_path / "b")
    assert ra.epoch_losses == rb.epoch_losses
    probs = a.predict_proba(texts)
    assert np.array_equal(probs, b.predict_proba(texts))
    for pa, pb in zip(a.model.parameters(), b.model.parameters(), strict=True):
        assert torch.equal(pa, pb)

    a.meta["task"] = "toy"
    a.save(tmp_path / "saved")
    loaded = EncoderClassifier.load(tmp_path / "saved")
    assert loaded.labels == LABELS
    assert loaded.max_length == SPEC.max_length
    assert loaded.meta == {"task": "toy"}
    assert np.array_equal(loaded.predict_proba(texts), probs)


def test_load_rejects_non_student_dir(tmp_path: Path) -> None:
    with pytest.raises(StudentError, match="not a saved Anvil student"):
        EncoderClassifier.load(tmp_path)
    (tmp_path / "anvil_student.json").write_text('{"labels": ["a"]}', encoding="utf-8")
    with pytest.raises(StudentError, match="not a complete Anvil student"):
        EncoderClassifier.load(tmp_path)


def test_train_rejects_bad_inputs() -> None:
    texts, targets = _toy_set()
    student = EncoderClassifier.build(SPEC, LABELS, texts)
    with pytest.raises(StudentError, match="targets"):
        train_classifier(student, texts, targets[:-1], SPEC)
    with pytest.raises(StudentError, match="no training examples"):
        train_classifier(student, [], [], SPEC)
