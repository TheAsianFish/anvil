"""The training loop for encoder students.

M1 keeps it to the essentials (seeded shuffling, AdamW, per-epoch JSONL metrics); M2 adds
warmup and decay, AMP, gradient accumulation, early stopping and exact resume.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from anvil.errors import StudentError
from anvil.log import get_logger
from anvil.spec import StudentSpec
from anvil.students.encoder import EncoderClassifier

log = get_logger(__name__)


@dataclass(frozen=True)
class TrainResult:
    examples: int
    epochs: int
    steps: int
    epoch_losses: list[float]
    seconds: float


def train_classifier(
    student: EncoderClassifier,
    texts: Sequence[str],
    targets: Sequence[int],
    spec: StudentSpec,
    metrics_path: Path | None = None,
) -> TrainResult:
    """Fit `student` on (text, label id) pairs with cross-entropy."""
    if len(texts) != len(targets):
        raise StudentError(f"{len(texts)} texts but {len(targets)} targets")
    if not texts:
        raise StudentError("no training examples; label some pool inputs first (anvil label)")
    torch.manual_seed(spec.seed)
    generator = torch.Generator().manual_seed(spec.seed)
    model = student.model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec.learning_rate)
    y = torch.tensor(list(targets), dtype=torch.long)
    started = time.perf_counter()
    epoch_losses: list[float] = []
    steps = 0
    metrics = metrics_path.open("w", encoding="utf-8") if metrics_path else None
    try:
        for epoch in range(spec.epochs):
            order = torch.randperm(len(texts), generator=generator).tolist()
            total, seen = 0.0, 0
            for start in range(0, len(order), spec.batch_size):
                idx = order[start : start + spec.batch_size]
                batch = student.encode([texts[i] for i in idx])
                out = model(**batch, labels=y[idx])
                optimizer.zero_grad(set_to_none=True)
                out.loss.backward()
                optimizer.step()
                steps += 1
                total += float(out.loss.detach()) * len(idx)
                seen += len(idx)
            epoch_losses.append(total / seen)
            record = {"epoch": epoch + 1, "steps": steps, "loss": round(epoch_losses[-1], 6)}
            log.debug("epoch", extra=record)
            if metrics is not None:
                metrics.write(json.dumps(record) + "\n")
    finally:
        if metrics is not None:
            metrics.close()
    model.eval()
    result = TrainResult(
        examples=len(texts),
        epochs=spec.epochs,
        steps=steps,
        epoch_losses=epoch_losses,
        seconds=time.perf_counter() - started,
    )
    log.info("trained", extra={k: v for k, v in asdict(result).items() if k != "epoch_losses"})
    return result
