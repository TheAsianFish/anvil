"""Training a LoRA generator with the generic loop: next-token cross-entropy on the answer
tokens of chat-templated sequences, with early stopping on the exact-match rate of greedy
answers on a validation slice carved out of the training data.

The loss is the mean over the answer tokens of a micro-batch (the prompt is masked), which is
what Hugging Face's causal-LM head computes from `labels`; the loop weights micro-batches by
their example count when it accumulates. Validation decodes greedily, the way the student
answers in use, rather than scoring teacher-forced token accuracy: a model can predict 95% of
a query's tokens and still get none of its queries right. Exact match is coarse, though: early
on it is 0 for several epochs, and an early stop on that tie would keep the first epoch, so the
validation loss breaks ties (`validation_score`).
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

from anvil.chat import IGNORE, ChatReport, Example, render_target
from anvil.errors import StoreError, StudentError
from anvil.log import get_logger
from anvil.outputs import decode_output
from anvil.spec import StudentSpec
from anvil.train.classifier import MIN_VALIDATION_EXAMPLES, validation_split
from anvil.train.loop import EvaluateFn, TrainConfig, TrainResult, fit

if TYPE_CHECKING:
    from anvil.students.generator import LoraGenerator

log = get_logger(__name__)


def collate(
    examples: Sequence[Example], pad_id: int, device: torch.device
) -> dict[str, torch.Tensor]:
    """A right-padded batch: ids, attention mask and labels (`IGNORE` on prompt and padding).
    Right padding is fine for training (positions count from the first real token); only
    batched generation needs the left."""
    width = max(len(e.input_ids) for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), width), dtype=torch.long)
    labels = torch.full((len(examples), width), IGNORE, dtype=torch.long)
    for row, e in enumerate(examples):
        n = len(e.input_ids)
        ids[row, :n] = torch.tensor(e.input_ids, dtype=torch.long)
        mask[row, :n] = 1
        labels[row, :n] = torch.tensor(e.labels, dtype=torch.long)
    return {
        "input_ids": ids.to(device),
        "attention_mask": mask.to(device),
        "labels": labels.to(device),
    }


def exact_match_rate(
    student: LoraGenerator, texts: Sequence[str], references: Sequence[dict[str, str]]
) -> float:
    """Share of `texts` whose greedy answer equals the reference in every field (stripped)."""
    predicted = student.predict(texts)
    hits = sum(
        p is not None and all(p[k].strip() == r[k].strip() for k in student.outputs)
        for p, r in zip(predicted, references, strict=True)
    )
    return hits / len(texts) if texts else 0.0


def validation_score(exact_match: float, loss: float, n: int) -> float:
    """Exact match, with the validation loss as a tie-break that can never outweigh it.

    The loss term lies in (0, 0.5 / n]: one more exact answer out of `n` (a step of 1 / n)
    always wins, and between epochs with the same exact-match count the lower loss does. The
    score is never below the exact match, and less than half an answer above it."""
    return exact_match + (0.5 / n) / (1.0 + loss)


@torch.no_grad()
def mean_loss(
    model: torch.nn.Module, examples: Sequence[Example], pad_id: int, batch_size: int = 16
) -> float:
    """Answer-token cross-entropy over `examples`, weighted by their answer tokens."""
    device = next(model.parameters()).device
    total, tokens = 0.0, 0
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        n = sum(e.answer_tokens for e in chunk)
        total += float(model(**collate(chunk, pad_id, device)).loss) * n
        tokens += n
    return total / tokens if tokens else 0.0


def train_generator(
    student: LoraGenerator,
    texts: Sequence[str],
    answers: Sequence[str],
    spec: StudentSpec,
    metrics_path: Path | None = None,
    *,
    input_ids: Sequence[str] | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    keep_checkpoint: bool = False,
) -> tuple[TrainResult, ChatReport]:
    """Fit `student`'s adapter on (input, stored teacher answer) pairs; return the run and the
    chat alignment report.

    Answers are decoded (`anvil.outputs`) and rendered as the student writes them
    (`chat.render_target`); examples the template or `max_length` cannot hold are dropped and
    counted, and training refuses to start when they pass `chat.MAX_DROPPED_RATE`. The
    validation slice is the classifier's (the same seeded text hash), drawn from the usable
    examples, and the epoch with the best validation exact match is kept.
    """
    if len(texts) != len(answers):
        raise StudentError(f"{len(texts)} texts but {len(answers)} answers")
    if not texts:
        raise StudentError("no training examples; label some pool inputs first (anvil label)")
    try:
        fields = [decode_output(a, student.outputs) for a in answers]
    except ValueError as exc:
        raise StoreError(
            f"a stored label is not an answer to this generate task ({exc}); were the labels"
            " made for another version of it?"
        ) from None
    targets = [render_target(f, student.outputs) for f in fields]
    built, report = student.examples(texts, targets, input_ids)
    log.info("aligned answers to the chat template", extra=report.as_dict())
    report.check()

    usable = [i for i, e in enumerate(built) if e is not None]
    examples = [e for e in built if e is not None]
    kept_texts = [texts[i] for i in usable]
    kept_fields = [fields[i] for i in usable]

    config = TrainConfig.from_student(spec)
    train_idx: list[int] = list(range(len(examples)))
    val_idx: list[int] = []
    if spec.validation_fraction > 0:
        train_idx, val_idx = validation_split(kept_texts, spec.validation_fraction, spec.seed)
        if len(val_idx) < MIN_VALIDATION_EXAMPLES or not train_idx:
            log.warning(
                "too few examples for a validation slice; training on all without early stopping",
                extra={"validation": len(val_idx), "minimum": MIN_VALIDATION_EXAMPLES},
            )
            train_idx, val_idx = list(range(len(examples))), []
            config = dataclasses.replace(config, patience=None)
    if spec.min_steps:
        per_epoch = math.ceil(math.ceil(len(train_idx) / spec.batch_size) / spec.grad_accum)
        config = dataclasses.replace(
            config, epochs=max(config.epochs, math.ceil(spec.min_steps / per_epoch))
        )
    train_examples = [examples[i] for i in train_idx]
    pad_id = int(student.tokenizer.pad_token_id)

    def loss_fn(model: torch.nn.Module, idx: list[int]) -> torch.Tensor:
        batch = collate([train_examples[i] for i in idx], pad_id, student.device)
        loss: torch.Tensor = model(**batch).loss
        return loss

    evaluate: EvaluateFn | None = None
    if val_idx:
        val_texts = [kept_texts[i] for i in val_idx]
        val_refs = [kept_fields[i] for i in val_idx]
        val_examples = [examples[i] for i in val_idx]

        def validation(model: torch.nn.Module) -> float:
            em = exact_match_rate(student, val_texts, val_refs)
            loss = mean_loss(model, val_examples, pad_id, spec.batch_size)
            log.info("validation", extra={"exact_match": em, "loss": round(loss, 6)})
            return validation_score(em, loss, len(val_texts))

        evaluate = validation

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
        data_fingerprint=_fingerprint(student, examples, val_idx, spec),
        trainable_only=True,
    )
    return dataclasses.replace(result, validation_examples=len(val_idx)), report


def _fingerprint(
    student: LoraGenerator, examples: Sequence[Example], val_idx: Sequence[int], spec: StudentSpec
) -> str:
    """sha256 of the token sequences and labels (they hold the template, the system prompt,
    the inputs and the teacher's answers), the validation slice, the base model and the
    adapter settings. A resume with any of them changed would continue on different data."""
    h = hashlib.sha256()
    h.update(json.dumps([[e.input_ids, e.labels] for e in examples]).encode())
    h.update(json.dumps([list(val_idx), spec.model, spec.lora.model_dump()]).encode())
    return h.hexdigest()
