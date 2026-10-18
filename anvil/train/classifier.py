"""Training an encoder classifier with the generic loop: cross-entropy on (text, label id)
pairs, with early stopping on the kappa of a validation slice carved out of the training data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from anvil.errors import StudentError
from anvil.evaluate.metrics import cohen_kappa
from anvil.log import get_logger
from anvil.spec import StudentSpec
from anvil.train.loop import EvaluateFn, TrainConfig, TrainResult, fit

if TYPE_CHECKING:
    from anvil.students.encoder import EncoderClassifier

log = get_logger(__name__)

MIN_VALIDATION_EXAMPLES = 10
"""Below this, validation kappa is too noisy to pick an epoch with (on two examples it is 0 or
1), and the few examples are worth more as training data."""


def validation_split(
    texts: Sequence[str], fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """(train indices, validation indices), from a seeded hash of each text.

    Hashing the text (not the position) keeps the split stable as labels are added and puts
    duplicates on the same side. The salt differs from the store's held-out split, whose hash
    would otherwise send no pool text to validation at the same seed.
    """
    train: list[int] = []
    val: list[int] = []
    for i, text in enumerate(texts):
        digest = hashlib.sha256(f"anvil-validation:{seed}:{text}".encode()).digest()
        u = int.from_bytes(digest[:8], "big") / 2**64
        (val if u < fraction else train).append(i)
    return train, val


def data_fingerprint(
    student: EncoderClassifier,
    texts: Sequence[str],
    targets: Sequence[int],
    val_idx: Sequence[int],
    spec: StudentSpec,
) -> str:
    """sha256 of what the student trains on: texts, targets (they carry the teacher model and
    prompt), the validation slice, the student model id and `max_length` (they decide what
    the model sees of each text) and, for the `tiny` student, the vocabulary learned from the
    pool. A resume with any of these changed would continue a run on different data."""
    # Imported lazily, like the type-only import above: the encoder pulls in transformers.
    from anvil.students.encoder import TINY

    h = hashlib.sha256()
    h.update(json.dumps([list(texts), list(targets), list(val_idx)]).encode())
    h.update(json.dumps([spec.model, spec.max_length]).encode())
    if spec.model == TINY:
        vocab = sorted(student.tokenizer.get_vocab().items())
        h.update(json.dumps(vocab, ensure_ascii=False).encode())
    return h.hexdigest()


def train_classifier(
    student: EncoderClassifier,
    texts: Sequence[str],
    targets: Sequence[int],
    spec: StudentSpec,
    metrics_path: Path | None = None,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    keep_checkpoint: bool = False,
) -> TrainResult:
    """Fit `student` on (text, label id) pairs with cross-entropy.

    With `spec.validation_fraction > 0`, a slice of the training pairs is held back and the
    epoch with the best validation kappa is kept. The slice comes out of the training data,
    never the reported held-out split: stopping on the test set would bias its kappa upward.
    """
    if len(texts) != len(targets):
        raise StudentError(f"{len(texts)} texts but {len(targets)} targets")
    if not texts:
        raise StudentError("no training examples; label some pool inputs first (anvil label)")
    config = TrainConfig.from_student(spec)
    train_idx: list[int] = list(range(len(texts)))
    val_idx: list[int] = []
    if spec.validation_fraction > 0:
        train_idx, val_idx = validation_split(texts, spec.validation_fraction, spec.seed)
        if len(val_idx) < MIN_VALIDATION_EXAMPLES or not train_idx:
            log.warning(
                "too few examples for a validation slice; training on all without early stopping",
                extra={"validation": len(val_idx), "minimum": MIN_VALIDATION_EXAMPLES},
            )
            train_idx, val_idx = list(range(len(texts))), []
            config = dataclasses.replace(config, patience=None)

    if spec.min_steps:
        per_epoch = math.ceil(math.ceil(len(train_idx) / spec.batch_size) / spec.grad_accum)
        config = dataclasses.replace(
            config, epochs=max(config.epochs, math.ceil(spec.min_steps / per_epoch))
        )
    train_texts = [texts[i] for i in train_idx]
    y = torch.tensor([targets[i] for i in train_idx], dtype=torch.long)

    def loss_fn(model: torch.nn.Module, idx: list[int]) -> torch.Tensor:
        batch = student.encode([train_texts[i] for i in idx])
        out = model(**batch, labels=y[idx].to(student.device))
        loss: torch.Tensor = out.loss
        return loss

    evaluate: EvaluateFn | None = None
    if val_idx:
        val_texts = [texts[i] for i in val_idx]
        val_labels = [student.labels[targets[i]] for i in val_idx]

        def validation_kappa(model: torch.nn.Module) -> float:
            return cohen_kappa(val_labels, student.predict(val_texts))

        evaluate = validation_kappa

    result = fit(
        student.model,
        len(train_idx),
        loss_fn,
        config,
        evaluate=evaluate,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        keep_checkpoint=keep_checkpoint,
        metrics_path=metrics_path,
        data_fingerprint=data_fingerprint(student, texts, targets, val_idx, spec),
    )
    return dataclasses.replace(result, validation_examples=len(val_idx))
