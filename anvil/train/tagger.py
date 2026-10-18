"""Training an encoder tagger with the generic loop: token cross-entropy on BIO tags, with early
stopping on the exact-match entity F1 of a validation slice carved out of the training data.

The spans are snapped to the student's tokens once, before training: the tags of each text are
computed at `max_length` truncation and padded per batch with the ignored id, and validation
scores the student against the snapped spans (what it can represent), not the raw ones, so a
boundary the vocabulary cannot express does not cap the metric below 1.
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

from anvil.align import AlignmentReport
from anvil.errors import StudentError
from anvil.evaluate.extraction import entity_scores
from anvil.log import get_logger
from anvil.spans import Span
from anvil.spec import StudentSpec
from anvil.train.classifier import MIN_VALIDATION_EXAMPLES, data_fingerprint, validation_split
from anvil.train.loop import EvaluateFn, TrainConfig, TrainResult, fit

if TYPE_CHECKING:
    from anvil.students.tagger import EncoderTagger

log = get_logger(__name__)


def _pad_labels(rows: list[list[int]], width: int, left: bool) -> torch.Tensor:
    """(batch, width) label ids, each row padded with the ignored id on the tokenizer's side."""
    from anvil.students.tagger import IGNORE

    out = torch.full((len(rows), width), IGNORE, dtype=torch.long)
    for k, row in enumerate(rows):
        if len(row) > width:
            raise StudentError(
                f"{len(row)} tag targets for a batch encoded to {width} tokens; the tokenizer"
                " truncates differently with and without padding"
            )
        if row:
            if left:
                out[k, width - len(row) :] = torch.tensor(row, dtype=torch.long)
            else:
                out[k, : len(row)] = torch.tensor(row, dtype=torch.long)
    return out


def train_tagger(
    student: EncoderTagger,
    texts: Sequence[str],
    spans: Sequence[Sequence[Span]],
    spec: StudentSpec,
    metrics_path: Path | None = None,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    keep_checkpoint: bool = False,
) -> tuple[TrainResult, AlignmentReport]:
    """Fit `student` on (text, verified spans) pairs; return the run and the snapping report.

    `spans` are `align_mentions` output (non-overlapping, offsets into the text). The report
    counts what snapping to the student's tokens cost (`no_token` past `max_length`,
    `token_conflict`, `snapped` boundaries); merge it into the character alignment's report
    and `check()` it to apply the unaligned-rate limit to both gates together.

    With `spec.validation_fraction > 0` the validation slice is the classifier's (the same
    seeded text hash), and the epoch with the best validation entity F1 is kept.
    """
    if len(texts) != len(spans):
        raise StudentError(f"{len(texts)} texts but {len(spans)} span lists")
    if not texts:
        raise StudentError("no training examples; label some pool inputs first (anvil label)")
    targets, report, snapped = student.targets(texts, spans)
    if report.unaligned:
        log.warning(
            "spans the student's tokens cannot represent were dropped",
            extra={k: v for k, v in report.dropped.items() if v},
        )

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
    train_targets = [targets[i] for i in train_idx]
    left = getattr(student.tokenizer, "padding_side", "right") == "left"

    def loss_fn(model: torch.nn.Module, idx: list[int]) -> torch.Tensor:
        batch = student.encode([train_texts[i] for i in idx])
        width = batch["input_ids"].shape[1]
        labels = _pad_labels([train_targets[i] for i in idx], width, left)
        out = model(**batch, labels=labels.to(student.device))
        loss: torch.Tensor = out.loss
        return loss

    evaluate: EvaluateFn | None = None
    if val_idx:
        val_texts = [texts[i] for i in val_idx]
        val_spans = [snapped[i] for i in val_idx]

        def validation_f1(model: torch.nn.Module) -> float:
            return entity_scores(val_spans, student.predict_spans(val_texts)).f1

        evaluate = validation_f1

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
        data_fingerprint=_fingerprint(student, texts, targets, val_idx, spec),
    )
    return dataclasses.replace(result, validation_examples=len(val_idx)), report


def _fingerprint(
    student: EncoderTagger,
    texts: Sequence[str],
    targets: list[list[int]],
    val_idx: Sequence[int],
    spec: StudentSpec,
) -> str:
    """The classifier's data fingerprint over the tag ids, plus the tag alphabet: renaming a
    type keeps every id and the head's shape but changes what the ids mean."""
    h = hashlib.sha256(data_fingerprint(student, texts, targets, val_idx, spec).encode())
    h.update(json.dumps(student.tags).encode())
    return h.hexdigest()
