"""The generic training loop: schedule, accumulation, early stopping, AMP and exact resume."""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from anvil.errors import StudentError
from anvil.spec import StudentSpec
from anvil.students.encoder import EncoderClassifier
from anvil.train.classifier import data_fingerprint, train_classifier, validation_split
from anvil.train.loop import (
    LAST,
    TrainConfig,
    TrainResult,
    _linear_schedule,
    _warmup_steps,
    fit,
)

N = 60
BASE = TrainConfig(epochs=4, batch_size=8, learning_rate=0.05, seed=3)


class SimulatedCrashError(Exception):
    pass


def _blobs() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(11)
    centers = torch.tensor([[2.0, 0.0], [-2.0, 1.0], [0.0, -2.0]])
    y = torch.arange(N) % 3
    x = centers[y] + 0.7 * torch.randn(N, 2, generator=gen)
    return x, y


X, Y = _blobs()


def _mlp(dropout: float = 0.2) -> torch.nn.Module:
    torch.manual_seed(5)
    return torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        torch.nn.LayerNorm(32),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(32, 3),
    )


def _loss_fn(
    fail_at: int | None = None,
) -> Callable[[torch.nn.Module, list[int]], torch.Tensor]:
    """Cross-entropy over the blobs; raises on micro-batch number `fail_at` (1-based)."""
    calls = 0

    def loss_fn(model: torch.nn.Module, idx: list[int]) -> torch.Tensor:
        nonlocal calls
        calls += 1
        if fail_at is not None and calls == fail_at:
            raise SimulatedCrashError
        return torch.nn.functional.cross_entropy(model(X[idx]), Y[idx])

    return loss_fn


def _neg_loss(model: torch.nn.Module) -> float:
    with torch.no_grad():
        return -float(torch.nn.functional.cross_entropy(model(X), Y))


def _records(path: Path, event: str) -> list[dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r["event"] == event]


def _params(model: torch.nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def test_loss_decreases_and_metrics_are_written(tmp_path: Path) -> None:
    model = _mlp()
    metrics = tmp_path / "m.jsonl"
    result = fit(model, N, _loss_fn(), BASE, metrics_path=metrics)
    assert result.epochs == 4
    assert result.steps == 4 * math.ceil(N / 8)
    assert result.epoch_losses[-1] < result.epoch_losses[0] / 2
    assert result.amp == "off"
    assert result.device == "cpu"
    assert result.best_epoch is None
    assert not result.stopped_early
    assert not model.training
    steps = _records(metrics, "step")
    assert [r["step"] for r in steps] == list(range(1, result.steps + 1))
    assert set(steps[0]) == {"event", "step", "epoch", "loss", "lr", "grad_norm"}
    epochs = _records(metrics, "epoch")
    assert [r["epoch"] for r in epochs] == [1, 2, 3, 4]
    assert [r["loss"] for r in epochs] == [round(x, 6) for x in result.epoch_losses]


RESUME = dataclasses.replace(
    BASE, grad_accum=2, batch_size=4, warmup_ratio=0.2, weight_decay=0.01, max_grad_norm=1.0,
    patience=3, checkpoint_every=3,
)  # fmt: skip


def _run_resumable(tmp_path: Path, fail_at: int | None, resume: bool) -> TrainResult:
    return fit(
        _mlp(),
        N,
        _loss_fn(fail_at),
        RESUME,
        evaluate=_neg_loss,
        checkpoint_dir=tmp_path / "ckpt",
        resume=resume,
        metrics_path=tmp_path / "m.jsonl",
    )


# 8 optimizer steps of 2 micro-batches per epoch (the last has one), 60 micro-batches in all;
# checkpoints every 3 steps and at epoch ends.
@pytest.mark.parametrize("fail_at", [3, 16, 37, 46, 58])
def test_resume_is_bit_exact(tmp_path: Path, fail_at: int) -> None:
    ref_model = _mlp()
    uninterrupted = dataclasses.replace(RESUME, checkpoint_every=None)
    ref = fit(
        ref_model,
        N,
        _loss_fn(),
        uninterrupted,
        evaluate=_neg_loss,
        metrics_path=tmp_path / "ref.jsonl",
    )

    with pytest.raises(SimulatedCrashError):
        _run_resumable(tmp_path, fail_at, resume=False)
    model = _mlp()
    resumed = fit(
        model,
        N,
        _loss_fn(),
        RESUME,
        evaluate=_neg_loss,
        checkpoint_dir=tmp_path / "ckpt",
        resume=True,
        metrics_path=tmp_path / "m.jsonl",
    )
    # The checkpoint the run resumed from: steps 3, 6 (mid-epoch), 8 (epoch end), ... Before
    # the first one (fail_at 3) the run starts over.
    assert resumed.resumed_at_step == {3: None, 16: 8, 37: 18, 46: 24, 58: 30}[fail_at]
    for a, b in zip(ref_model.parameters(), model.parameters(), strict=True):
        assert torch.equal(a, b)
    assert resumed.epoch_losses == ref.epoch_losses
    assert (resumed.steps, resumed.best_epoch, resumed.best_metric) == (
        ref.steps,
        ref.best_epoch,
        ref.best_metric,
    )
    ref_text = (tmp_path / "ref.jsonl").read_text(encoding="utf-8")
    assert (tmp_path / "m.jsonl").read_text(encoding="utf-8") == ref_text
    assert not (tmp_path / "ckpt").exists()


def test_resume_with_a_different_config_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SimulatedCrashError):
        _run_resumable(tmp_path, 20, resume=False)
    changed = dataclasses.replace(RESUME, learning_rate=0.01)
    with pytest.raises(StudentError, match="different training config"):
        fit(
            _mlp(),
            N,
            _loss_fn(),
            changed,
            evaluate=_neg_loss,
            checkpoint_dir=tmp_path / "ckpt",
            resume=True,
        )
    # Only how often state is written may change between runs.
    every = dataclasses.replace(RESUME, checkpoint_every=5)
    result = fit(
        _mlp(), N, _loss_fn(), every, evaluate=_neg_loss, checkpoint_dir=tmp_path / "ckpt",
        resume=True,
    )  # fmt: skip
    assert result.resumed_at_step == 9  # the last checkpoint before micro-batch 20


def test_resume_without_a_checkpoint_starts_fresh(tmp_path: Path) -> None:
    result = fit(_mlp(), N, _loss_fn(), BASE, checkpoint_dir=tmp_path / "c", resume=True)
    assert result.resumed_at_step is None
    assert result.epochs == BASE.epochs


def test_checkpoint_dir_is_removed_unless_kept(tmp_path: Path) -> None:
    fit(_mlp(), N, _loss_fn(), BASE, evaluate=_neg_loss, checkpoint_dir=tmp_path / "gone")
    assert not (tmp_path / "gone").exists()
    kept = tmp_path / "kept"
    fit(
        _mlp(), N, _loss_fn(), BASE, evaluate=_neg_loss, checkpoint_dir=kept,
        keep_checkpoint=True,
    )  # fmt: skip
    names = sorted(p.name for p in kept.iterdir())
    assert names[-1] == LAST
    assert len(names) == 2
    assert names[0].startswith("best-epoch")


def test_accumulation_matches_a_bigger_batch() -> None:
    config = dataclasses.replace(BASE, warmup_ratio=0.25, weight_decay=0.1, max_grad_norm=0.5)
    big = _mlp(dropout=0.0)
    accumulated = _mlp(dropout=0.0)
    r1 = fit(big, N, _loss_fn(), config)
    r2 = fit(accumulated, N, _loss_fn(), dataclasses.replace(config, batch_size=4, grad_accum=2))
    assert r1.steps == r2.steps
    for a, b in zip(big.parameters(), accumulated.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)
    assert r1.epoch_losses == pytest.approx(r2.epoch_losses, abs=1e-6)


def test_lr_warms_up_then_decays(tmp_path: Path) -> None:
    config = dataclasses.replace(BASE, warmup_ratio=0.25)  # 32 steps, 8 of warmup
    fit(_mlp(), N, _loss_fn(), config, metrics_path=tmp_path / "m.jsonl")
    lrs = [float(str(r["lr"])) for r in _records(tmp_path / "m.jsonl", "step")]
    assert len(lrs) == 32
    assert lrs[0] == 0.0
    assert lrs.index(max(lrs)) == 8
    assert max(lrs) == pytest.approx(BASE.learning_rate)
    assert all(a < b for a, b in itertools.pairwise(lrs[:9]))
    assert all(a > b for a, b in itertools.pairwise(lrs[8:]))
    assert lrs[-1] == pytest.approx(BASE.learning_rate / 24, abs=1e-6)


def test_a_single_step_run_is_not_spent_warming_up(tmp_path: Path) -> None:
    # Warmup starts at LR 0, so a warmup as long as the run would leave the model untouched.
    assert _warmup_steps(0.5, 1) == 0
    assert _linear_schedule(_warmup_steps(0.5, 1), 1)(0) == 1.0
    assert _warmup_steps(0.9, 4) == 3
    config = dataclasses.replace(BASE, epochs=1, batch_size=N, warmup_ratio=0.5)
    model = _mlp()
    before = _params(model)
    fit(model, N, _loss_fn(), config, metrics_path=tmp_path / "m.jsonl")
    lrs = [float(str(r["lr"])) for r in _records(tmp_path / "m.jsonl", "step")]
    assert lrs == [pytest.approx(BASE.learning_rate)]
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters(), strict=True))


def test_early_stopping_restores_the_best_epoch(tmp_path: Path) -> None:
    scripted = iter([0.1, 0.5, 0.4, 0.5, 0.9, 0.9])
    snapshots: list[list[torch.Tensor]] = []

    def evaluate(model: torch.nn.Module) -> float:
        assert not model.training
        snapshots.append(_params(model))
        return next(scripted)

    model = _mlp()
    config = dataclasses.replace(BASE, epochs=6, patience=2)
    result = fit(model, N, _loss_fn(), config, evaluate=evaluate, metrics_path=tmp_path / "m")
    # 0.5 at epoch 4 ties the best (no strict improvement): two bad epochs after epoch 2.
    assert result.epochs == 4
    assert len(result.epoch_losses) == 4
    assert result.stopped_early
    assert (result.best_epoch, result.best_metric) == (2, 0.5)
    for got, best in zip(model.parameters(), snapshots[1], strict=True):
        assert torch.equal(got, best)
    assert [r["best_epoch"] for r in _records(tmp_path / "m", "epoch")] == [1, 2, 2, 2]


def test_patience_running_out_on_the_last_epoch_is_not_an_early_stop() -> None:
    scripted = iter([0.5, 0.4, 0.3])
    config = dataclasses.replace(BASE, epochs=3, patience=2)
    result = fit(_mlp(), N, _loss_fn(), config, evaluate=lambda m: next(scripted))
    assert result.epochs == 3
    assert not result.stopped_early
    assert result.best_epoch == 1


def test_bad_configurations_are_student_errors(tmp_path: Path) -> None:
    with pytest.raises(StudentError, match="patience needs a validation metric"):
        fit(_mlp(), N, _loss_fn(), dataclasses.replace(BASE, patience=2))
    with pytest.raises(StudentError, match="no training examples"):
        fit(_mlp(), 0, _loss_fn(), BASE)
    with pytest.raises(StudentError, match="checkpoint_dir"):
        fit(_mlp(), N, _loss_fn(), dataclasses.replace(BASE, checkpoint_every=2))
    with pytest.raises(StudentError, match="fp16 needs a CUDA device"):
        fit(_mlp(), N, _loss_fn(), dataclasses.replace(BASE, amp="fp16"))


def test_amp_bf16_on_cpu_and_auto_resolves_to_off() -> None:
    bf16 = fit(_mlp(), N, _loss_fn(), dataclasses.replace(BASE, amp="bf16"))
    assert bf16.amp == "bf16"
    assert all(math.isfinite(x) for x in bf16.epoch_losses)
    assert bf16.epoch_losses[-1] < bf16.epoch_losses[0]
    assert fit(_mlp(), N, _loss_fn(), dataclasses.replace(BASE, amp="auto")).amp == "off"


def test_from_student_turns_early_stopping_on_only_with_validation() -> None:
    spec = StudentSpec(epochs=3, grad_accum=2, patience=4)
    config = TrainConfig.from_student(spec)
    assert (config.epochs, config.grad_accum, config.patience, config.amp) == (3, 2, 4, "auto")
    no_val = TrainConfig.from_student(spec.model_copy(update={"validation_fraction": 0.0}))
    assert no_val.patience is None


def test_validation_split_is_stable_disjoint_and_sized() -> None:
    texts = [f"ticket number {i}" for i in range(1000)] + ["dup", "dup"]
    train, val = validation_split(texts, 0.2, seed=0)
    assert (train, val) == validation_split(texts, 0.2, seed=0)
    assert sorted(train + val) == list(range(len(texts)))
    assert 150 <= len(val) <= 250
    assert (1000 in val) == (1001 in val)
    assert (1000 in train) == (1001 in train)
    assert validation_split(texts, 0.2, seed=1)[1] != val
    assert validation_split(texts, 0.0, seed=0)[1] == []


LABELS = ["billing", "bug", "feature", "account"]
WORDS = {
    "billing": ["refund", "invoice", "charge"],
    "bug": ["crash", "error", "broken"],
    "feature": ["wish", "add", "idea"],
    "account": ["password", "login", "email"],
}
FILLER = ["please", "help", "today", "my", "the", "app", "again", "now"]
TINY = StudentSpec(
    model="tiny", max_length=16, epochs=4, batch_size=8, learning_rate=3e-3,
    validation_fraction=0.25, patience=2,
)  # fmt: skip


def _toy_set(n: int = 64) -> tuple[list[str], list[int]]:
    texts, targets = [], []
    for k in range(n):
        label = LABELS[k % 4]
        texts.append(f"{WORDS[label][k % 3]} {FILLER[k % 8]} {FILLER[(k * 3) % 8]} {k}")
        targets.append(k % 4)
    return texts, targets


def test_train_classifier_holds_out_validation_and_keeps_the_best_epoch(tmp_path: Path) -> None:
    texts, targets = _toy_set()
    student = EncoderClassifier.build(TINY, LABELS, texts)
    result = train_classifier(student, texts, targets, TINY, tmp_path / "m.jsonl")
    _, val = validation_split(texts, TINY.validation_fraction, TINY.seed)
    assert result.validation_examples == len(val) > 0
    assert result.examples + result.validation_examples == len(texts)
    assert result.best_epoch is not None
    assert result.best_metric is not None
    assert -1.0 <= result.best_metric <= 1.0
    metrics = [r["metric"] for r in _records(tmp_path / "m.jsonl", "epoch")]
    assert result.best_metric == max(m for m in metrics if isinstance(m, float))


def test_train_classifier_with_a_tiny_validation_slice_skips_early_stopping() -> None:
    texts, targets = _toy_set(12)
    student = EncoderClassifier.build(TINY, LABELS, texts)
    result = train_classifier(student, texts, targets, TINY)
    assert result.validation_examples == 0
    assert result.best_epoch is None
    assert result.examples == 12


def test_encoder_resume_is_bit_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    texts, targets = _toy_set()
    spec = TINY.model_copy(update={"epochs": 3, "grad_accum": 2, "batch_size": 4})
    ref = EncoderClassifier.build(spec, LABELS, texts)
    ref_result = ref.fit(texts, targets, spec)

    student = EncoderClassifier.build(spec, LABELS, texts)
    encode = EncoderClassifier.encode
    calls = 0

    def failing_encode(self: EncoderClassifier, batch: list[str]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == 15:  # mid epoch 2, after its first epoch-end checkpoint
            raise SimulatedCrashError
        return encode(self, batch)

    monkeypatch.setattr(EncoderClassifier, "encode", failing_encode)
    with pytest.raises(SimulatedCrashError):
        student.fit(texts, targets, spec, checkpoint_dir=tmp_path / "ckpt")
    monkeypatch.setattr(EncoderClassifier, "encode", encode)
    student = EncoderClassifier.build(spec, LABELS, texts)
    result = student.fit(texts, targets, spec, checkpoint_dir=tmp_path / "ckpt", resume=True)
    assert result.resumed_at_step is not None
    assert result.epoch_losses == ref_result.epoch_losses
    for a, b in zip(ref.model.parameters(), student.model.parameters(), strict=True):
        assert torch.equal(a, b)


def test_encoder_resume_on_other_data_of_the_same_size_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts, targets = _toy_set()
    spec = TINY.model_copy(update={"epochs": 3, "grad_accum": 2, "batch_size": 4})
    student = EncoderClassifier.build(spec, LABELS, texts)
    encode = EncoderClassifier.encode
    calls = 0

    def failing_encode(self: EncoderClassifier, batch: list[str]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == 15:  # after the first epoch-end checkpoint
            raise SimulatedCrashError
        return encode(self, batch)

    monkeypatch.setattr(EncoderClassifier, "encode", failing_encode)
    with pytest.raises(SimulatedCrashError):
        student.fit(texts, targets, spec, checkpoint_dir=tmp_path / "ckpt")
    monkeypatch.setattr(EncoderClassifier, "encode", encode)
    # Same texts and count, different teacher answers: the count alone would not notice.
    relabelled = [(t + 1) % 4 for t in targets]
    student = EncoderClassifier.build(spec, LABELS, texts)
    with pytest.raises(StudentError, match="different training config"):
        student.fit(texts, relabelled, spec, checkpoint_dir=tmp_path / "ckpt", resume=True)
    # The original data still resumes.
    student = EncoderClassifier.build(spec, LABELS, texts)
    result = student.fit(texts, targets, spec, checkpoint_dir=tmp_path / "ckpt", resume=True)
    assert result.resumed_at_step is not None


def test_encoder_resume_with_another_max_length_or_model_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts, targets = _toy_set()
    spec = TINY.model_copy(update={"epochs": 3, "grad_accum": 2, "batch_size": 4})
    student = EncoderClassifier.build(spec, LABELS, texts)
    encode = EncoderClassifier.encode
    calls = 0

    def failing_encode(self: EncoderClassifier, batch: list[str]) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        if calls == 15:
            raise SimulatedCrashError
        return encode(self, batch)

    monkeypatch.setattr(EncoderClassifier, "encode", failing_encode)
    with pytest.raises(SimulatedCrashError):
        student.fit(texts, targets, spec, checkpoint_dir=tmp_path / "ckpt")
    monkeypatch.setattr(EncoderClassifier, "encode", encode)
    # Same data and loop settings, but the model sees a different prefix of every text.
    shorter = spec.model_copy(update={"max_length": 4})
    student = EncoderClassifier.build(shorter, LABELS, texts)
    with pytest.raises(StudentError, match="different training config"):
        student.fit(texts, targets, shorter, checkpoint_dir=tmp_path / "ckpt", resume=True)
    # Another model id changes the fingerprint even when the architecture (and so the
    # checkpoint's tensor shapes) would match: the weights would come from another model.
    one = spec.model_copy(update={"model": "org/one-encoder"})
    other = spec.model_copy(update={"model": "org/another-encoder"})
    assert data_fingerprint(student, texts, targets, [], one) != data_fingerprint(
        student, texts, targets, [], other
    )


def test_resume_into_a_model_of_another_shape_is_a_student_error(tmp_path: Path) -> None:
    with pytest.raises(SimulatedCrashError):
        _run_resumable(tmp_path, 20, resume=False)
    torch.manual_seed(5)
    narrower = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.ReLU(), torch.nn.Linear(16, 3))
    with pytest.raises(StudentError, match="does not fit this model"):
        fit(
            narrower, N, _loss_fn(), RESUME, evaluate=_neg_loss,
            checkpoint_dir=tmp_path / "ckpt", resume=True,
        )  # fmt: skip


def test_resume_under_another_resolved_precision_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `amp: auto` resolves per machine: a checkpoint written where it meant bf16 must not
    # resume where it means fp32 (or the reverse) under the same config.
    with pytest.raises(SimulatedCrashError):
        _run_resumable(tmp_path, 20, resume=False)
    monkeypatch.setattr("anvil.train.loop._resolve_amp", lambda mode, device: torch.bfloat16)
    with pytest.raises(StudentError, match="precision"):
        _run_resumable(tmp_path, None, resume=True)


def _scripted(
    values: list[float], start: int, snapshots: dict[int, list[torch.Tensor]]
) -> Callable[[torch.nn.Module], float]:
    """A validation metric that follows a script by epoch (`start` epochs already done when
    a run resumes) and snapshots the weights it scored."""
    calls = start

    def evaluate(model: torch.nn.Module) -> float:
        nonlocal calls
        calls += 1
        snapshots.setdefault(calls, _params(model))
        return values[calls - 1]

    return evaluate


def test_best_weights_from_before_a_crash_are_restored_after_the_resume(tmp_path: Path) -> None:
    # Best at epoch 2, then two worse epochs: patience 2 stops the run after epoch 4 of 5.
    script = [0.5, 0.9, 0.6, 0.55, 0.7]
    config = dataclasses.replace(RESUME, epochs=5, patience=2, checkpoint_every=None)
    ref_model = _mlp()
    ref_snaps: dict[int, list[torch.Tensor]] = {}
    ref = fit(ref_model, N, _loss_fn(), config, evaluate=_scripted(script, 0, ref_snaps))
    assert (ref.best_epoch, ref.epochs, ref.stopped_early) == (2, 4, True)
    for a, b in zip(ref_model.parameters(), ref_snaps[2], strict=True):
        assert torch.equal(a, b)  # the kept weights are the epoch-2 ones, not the last

    # Crash in epoch 3 (micro-batch 37), after the epoch-2 checkpoint wrote best-epoch2.pt.
    with pytest.raises(SimulatedCrashError):
        fit(_mlp(), N, _loss_fn(37), config, evaluate=_scripted(script, 0, {}),
            checkpoint_dir=tmp_path / "ckpt")  # fmt: skip
    assert (tmp_path / "ckpt" / "best-epoch2.pt").exists()
    model = _mlp()
    snaps: dict[int, list[torch.Tensor]] = {}
    resumed = fit(model, N, _loss_fn(), config, evaluate=_scripted(script, 2, snaps),
                  checkpoint_dir=tmp_path / "ckpt", resume=True)  # fmt: skip
    assert resumed.resumed_at_step == 16
    assert (resumed.best_epoch, resumed.epochs, resumed.stopped_early) == (2, 4, True)
    assert resumed.best_metric == 0.9
    for a, b in zip(model.parameters(), ref_model.parameters(), strict=True):
        assert torch.equal(a, b)  # restored from best-epoch2.pt, which the resume never scored


def test_patience_below_one_is_refused() -> None:
    with pytest.raises(StudentError, match="patience must be at least 1"):
        fit(_mlp(), N, _loss_fn(), dataclasses.replace(BASE, patience=0), evaluate=_neg_loss)


def test_a_corrupt_best_file_is_a_student_error(tmp_path: Path) -> None:
    with pytest.raises(SimulatedCrashError):
        _run_resumable(tmp_path, 37, resume=False)
    best = tmp_path / "ckpt" / "best-epoch2.pt"
    assert best.exists()
    best.write_bytes(best.read_bytes()[:100])
    with pytest.raises(StudentError, match=r"unreadable.*delete the checkpoint directory"):
        _run_resumable(tmp_path, None, resume=True)


def test_lr_is_logged_with_significant_digits(tmp_path: Path) -> None:
    # 80 steps at the spec's default LR: decay moves by 7e-7 a step, below six decimals.
    config = dataclasses.replace(BASE, learning_rate=5e-5, warmup_ratio=0.1, epochs=10)
    fit(_mlp(), N, _loss_fn(), config, metrics_path=tmp_path / "m.jsonl")
    lrs = [float(str(r["lr"])) for r in _records(tmp_path / "m.jsonl", "step")]
    assert lrs[0] == 0
    assert max(lrs) == pytest.approx(5e-5, rel=1e-5)
    decay = lrs[lrs.index(max(lrs)) :]
    assert decay == sorted(decay, reverse=True)
    assert len(set(decay)) == len(decay)  # every decay step is distinct in the log
