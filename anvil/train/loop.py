"""One seeded PyTorch training loop for every student.

The loop knows nothing about tokenizers or heads: a student hands it a model, the number of
examples and a `loss_fn(model, indices)` that returns the mean loss over those examples. The
loop owns everything that has to be identical across students and across restarts: seeded
shuffling, AdamW with decoupled weight decay, linear warmup then linear decay, gradient
accumulation, AMP, gradient clipping, early stopping on a validation metric, checkpoints that
resume bit-exactly, and JSONL metrics per optimizer step and per epoch.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, TextIO

import torch

from anvil.errors import StudentError
from anvil.log import get_logger
from anvil.spec import StudentSpec

log = get_logger(__name__)

AmpMode = Literal["auto", "off", "bf16", "fp16"]
LossFn = Callable[[torch.nn.Module, list[int]], torch.Tensor]
EvaluateFn = Callable[[torch.nn.Module], float]
StateDict = dict[str, torch.Tensor]

CHECKPOINT_VERSION = 1
LAST = "last.pt"
BEST_PREFIX = "best-epoch"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    grad_accum: int = 1
    """Micro-batches per optimizer step; the effective batch is `batch_size * grad_accum`."""
    warmup_ratio: float = 0.0
    """Fraction of optimizer steps with linear warmup from 0; then linear decay to 0."""
    weight_decay: float = 0.0
    """AdamW decay, never applied to biases and norm weights (parameters with ndim < 2)."""
    max_grad_norm: float | None = None
    amp: AmpMode = "off"
    """`auto` is bf16 (or fp16 with a GradScaler) on CUDA and off elsewhere; `bf16` is allowed
    on CPU when asked for explicitly."""
    patience: int | None = None
    """Epochs without a strict improvement of `evaluate` before stopping; needs `evaluate`."""
    checkpoint_every: int | None = None
    """Optimizer steps between mid-epoch checkpoints; None checkpoints at epoch ends only."""

    @classmethod
    def from_student(cls, spec: StudentSpec) -> TrainConfig:
        """The loop settings of a student spec. Early stopping is on only when the spec holds
        out a validation slice to stop on."""
        return cls(
            epochs=spec.epochs,
            batch_size=spec.batch_size,
            learning_rate=spec.learning_rate,
            seed=spec.seed,
            grad_accum=spec.grad_accum,
            warmup_ratio=spec.warmup_ratio,
            weight_decay=spec.weight_decay,
            max_grad_norm=spec.max_grad_norm,
            amp=spec.amp,
            patience=spec.patience if spec.validation_fraction > 0 else None,
        )

    def fingerprint(self, n_examples: int, data: str = "") -> str:
        """Hash of everything that changes the trajectory. `checkpoint_every` only changes how
        often state is written, so a resume may use a different value. `data` is the caller's
        digest of the training examples: the example count alone would let a resume silently
        continue on other data of the same size."""
        payload: dict[str, object] = {
            k: v for k, v in asdict(self).items() if k != "checkpoint_every"
        }
        payload["n_examples"] = n_examples
        if data:  # absent when not given, so older checkpoints keep their fingerprint
            payload["data"] = data
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


@dataclass(frozen=True)
class TrainResult:
    examples: int
    """Examples trained on (a validation slice, if any, is counted separately)."""
    epochs: int
    """Epochs completed, fewer than configured when early stopping fired."""
    steps: int
    epoch_losses: list[float]
    seconds: float
    best_epoch: int | None = None
    best_metric: float | None = None
    stopped_early: bool = False
    resumed_at_step: int | None = None
    amp: str = "off"
    device: str = "cpu"
    validation_examples: int = 0


@dataclass
class _State:
    """Everything besides the torch objects that a checkpoint must restore."""

    epoch: int = 0  # epochs completed
    position: int = 0  # next optimizer step within the current epoch
    order: list[int] | None = None  # the current epoch's permutation, None between epochs
    loss_sum: float = 0.0
    seen: int = 0
    epoch_losses: list[float] = field(default_factory=list)
    step: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    bad_epochs: int = 0
    stopped_early: bool = False
    best_state: StateDict | None = None


def fit(
    model: torch.nn.Module,
    n_examples: int,
    loss_fn: LossFn,
    config: TrainConfig,
    *,
    evaluate: EvaluateFn | None = None,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
    keep_checkpoint: bool = False,
    metrics_path: Path | None = None,
    data_fingerprint: str = "",
) -> TrainResult:
    """Train `model` in place and return what happened.

    `loss_fn(model, indices)` returns the mean loss over those example indices, on the model's
    device. `evaluate(model)` (higher is better) runs after every epoch; the weights of the best
    epoch are restored at the end. With `checkpoint_dir`, state is written at every epoch end
    (and every `checkpoint_every` optimizer steps), and `resume=True` continues from it with the
    same result as an uninterrupted run, bit for bit. The checkpoint files are deleted when
    training completes unless `keep_checkpoint`. `data_fingerprint` (a digest of the examples
    behind `loss_fn`) is folded into the checkpoint fingerprint, so resuming on different data
    is refused instead of silently mixing two datasets.
    """
    _check(config, n_examples, evaluate, checkpoint_dir)
    started = time.perf_counter()
    device = _device_of(model)
    amp_dtype = _resolve_amp(config.amp, device)
    amp_name = "off" if amp_dtype is None else {torch.bfloat16: "bf16"}.get(amp_dtype, "fp16")
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        _param_groups(model, config.weight_decay), lr=config.learning_rate
    )
    per_step = config.batch_size * config.grad_accum
    steps_per_epoch = math.ceil(n_examples / per_step)
    total = config.epochs * steps_per_epoch
    warmup = _warmup_steps(config.warmup_ratio, total)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _linear_schedule(warmup, total))
    scaler = torch.amp.GradScaler(device.type, enabled=amp_dtype == torch.float16)
    params = [p for p in model.parameters() if p.requires_grad]
    fingerprint = config.fingerprint(n_examples, data_fingerprint)

    state = _State()
    resumed_at: int | None = None
    last = checkpoint_dir / LAST if checkpoint_dir is not None else None
    if resume and last is not None and last.exists():
        state = _load(last, fingerprint, model, optimizer, scheduler, scaler, generator, device)
        resumed_at = state.step
        log.info("resumed", extra={"checkpoint": str(last), "at_step": state.step})
    elif checkpoint_dir is not None:
        _remove_checkpoint_files(checkpoint_dir)  # a stale run must not be resumed later

    def save() -> None:
        assert checkpoint_dir is not None
        _save(checkpoint_dir, fingerprint, state, model, optimizer, scheduler, scaler, generator)

    metrics = _open_metrics(metrics_path, state if resumed_at is not None else None)
    model.train()
    try:
        while state.epoch < config.epochs and not state.stopped_early:
            if state.order is None:
                state.order = torch.randperm(n_examples, generator=generator).tolist()
                state.position, state.loss_sum, state.seen = 0, 0.0, 0
            while state.position < steps_per_epoch:
                start = state.position * per_step
                chunk = state.order[start : start + per_step]
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for mb in range(0, len(chunk), config.batch_size):
                    idx = chunk[mb : mb + config.batch_size]
                    with _autocast(device, amp_dtype):
                        loss = loss_fn(model, idx)
                    # Weighting by share of the step makes accumulation equal to a big batch.
                    weighted = scaler.scale(loss.float() * (len(idx) / len(chunk)))
                    weighted.backward()  # type: ignore[no-untyped-call]
                    value = float(loss.detach()) * len(idx)
                    step_loss += value
                    state.loss_sum += value
                    state.seen += len(idx)
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(params, config.max_grad_norm or math.inf)
                lr = float(scheduler.get_last_lr()[0])
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                state.step += 1
                state.position += 1
                grad_norm = float(norm)
                _write(
                    metrics,
                    {
                        "event": "step",
                        "step": state.step,
                        "epoch": state.epoch + 1,
                        "loss": round(step_loss / len(chunk), 6),
                        "lr": round(lr, 6),
                        "grad_norm": round(grad_norm, 6) if math.isfinite(grad_norm) else None,
                    },
                )
                mid_epoch = state.position < steps_per_epoch
                every = config.checkpoint_every
                if every is not None and state.step % every == 0 and mid_epoch:
                    save()

            state.epoch_losses.append(state.loss_sum / state.seen)
            state.epoch += 1
            state.order = None
            metric: float | None = None
            if evaluate is not None:
                model.eval()
                metric = float(evaluate(model))
                model.train()
                improved = state.best_metric is None or metric > state.best_metric
                if improved and not math.isnan(metric):  # a NaN metric never becomes the best
                    state.best_metric, state.best_epoch, state.bad_epochs = metric, state.epoch, 0
                    state.best_state = {
                        k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()
                    }
                else:
                    state.bad_epochs += 1
                    remaining = state.epoch < config.epochs  # stopping at the end is no stop
                    if config.patience is not None and state.bad_epochs >= config.patience:
                        state.stopped_early = remaining
            record = {
                "epoch": state.epoch,
                "step": state.step,
                "loss": round(state.epoch_losses[-1], 6),
                "metric": metric,
                "best_epoch": state.best_epoch,
            }
            _write(metrics, {"event": "epoch", **record})
            log.info("epoch", extra=record)
            if checkpoint_dir is not None:
                save()
    finally:
        if metrics is not None:
            metrics.close()

    if state.best_state is not None and state.best_epoch != state.epoch:
        model.load_state_dict(state.best_state)
    model.eval()
    if checkpoint_dir is not None and not keep_checkpoint:
        remove_checkpoint(checkpoint_dir)
    result = TrainResult(
        examples=n_examples,
        epochs=state.epoch,
        steps=state.step,
        epoch_losses=list(state.epoch_losses),
        seconds=time.perf_counter() - started,
        best_epoch=state.best_epoch,
        best_metric=state.best_metric,
        stopped_early=state.stopped_early,
        resumed_at_step=resumed_at,
        amp=amp_name,
        device=str(device),
    )
    log.info("trained", extra={k: v for k, v in asdict(result).items() if k != "epoch_losses"})
    return result


def _check(
    config: TrainConfig,
    n_examples: int,
    evaluate: EvaluateFn | None,
    checkpoint_dir: Path | None,
) -> None:
    if n_examples <= 0:
        raise StudentError("no training examples; label some pool inputs first (anvil label)")
    if config.epochs < 1 or config.batch_size < 1 or config.grad_accum < 1:
        raise StudentError("epochs, batch_size and grad_accum must all be at least 1")
    if not 0.0 <= config.warmup_ratio < 1.0:
        raise StudentError(f"warmup_ratio must be in [0, 1), got {config.warmup_ratio}")
    if config.patience is not None and evaluate is None:
        raise StudentError(
            "patience needs a validation metric; pass `evaluate` or set patience to None"
        )
    if config.checkpoint_every is not None and checkpoint_dir is None:
        raise StudentError("checkpoint_every needs a checkpoint_dir to write to")


def _device_of(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        raise StudentError("the model has no parameters to train") from None


def _resolve_amp(mode: AmpMode, device: torch.device) -> torch.dtype | None:
    """The autocast dtype, or None for full precision.

    bf16 needs no loss scaling, so `auto` prefers it on GPUs that support it. fp16 autocast and
    gradient scaling are only worth it (and only well supported) on CUDA.
    """
    if mode == "off":
        return None
    if mode == "bf16":
        return torch.bfloat16
    if mode == "fp16":
        if device.type != "cuda":
            raise StudentError(
                f"amp: fp16 needs a CUDA device, the model is on {device}; use bf16"
            )
        return torch.float16
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _autocast(
    device: torch.device, dtype: torch.dtype | None
) -> contextlib.AbstractContextManager[object]:
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def _param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    """Decay matrices only: decaying biases and norm gains pulls them towards 0, which is not a
    meaningful prior for them."""
    decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
    no_decay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
    groups: list[dict[str, object]] = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return [g for g in groups if g["params"]]


def _warmup_steps(ratio: float, total: int) -> int:
    """Warmup length in optimizer steps, at most `total - 1`: step 0 of a warmup runs at LR 0,
    so a warmup covering every step would train nothing (a one-step run gets no warmup)."""
    if ratio <= 0:
        return 0
    return min(max(1, round(ratio * total)), max(0, total - 1))


def _linear_schedule(warmup: int, total: int) -> Callable[[int], float]:
    """LR multiplier for optimizer step `s` (0-based): s / warmup, then linear decay that would
    reach 0 one step after the last one."""

    def factor(s: int) -> float:
        if s < warmup:
            return s / warmup
        return max(0.0, (total - s) / max(1, total - warmup))

    return factor


def _write(metrics: TextIO | None, record: dict[str, object]) -> None:
    if metrics is not None:
        metrics.write(json.dumps(record) + "\n")
        metrics.flush()


def _open_metrics(path: Path | None, resumed: _State | None) -> TextIO | None:
    """Open the JSONL metrics file. On resume, records written after the checkpoint (by the
    interrupted run) are dropped first, so the file ends up exactly as an uninterrupted run's."""
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if resumed is None or not path.exists():
        return path.open("w", encoding="utf-8")
    kept: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue  # a line cut short by the interruption
        event = record.get("event")
        before_step = event == "step" and record.get("step", 0) <= resumed.step
        before_epoch = event == "epoch" and record.get("epoch", 0) <= resumed.epoch
        if before_step or before_epoch:
            kept.append(line)
    _atomic_write_text(path, "".join(line + "\n" for line in kept))
    return path.open("a", encoding="utf-8")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _atomic_save(obj: object, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _best_name(epoch: int) -> str:
    return f"{BEST_PREFIX}{epoch}.pt"


def _checkpoint_files(directory: Path) -> Iterator[Path]:
    if not directory.is_dir():
        return
    for path in directory.iterdir():
        name = path.name.removesuffix(".tmp")
        if name == LAST or (name.startswith(BEST_PREFIX) and name.endswith(".pt")):
            yield path


def _remove_checkpoint_files(directory: Path) -> None:
    for path in list(_checkpoint_files(directory)):
        path.unlink(missing_ok=True)


def remove_checkpoint(directory: Path) -> None:
    """Delete a run's checkpoint files, and the directory if nothing else lives there."""
    _remove_checkpoint_files(directory)
    with contextlib.suppress(OSError):
        directory.rmdir()


def _save(
    directory: Path,
    fingerprint: str,
    state: _State,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
) -> None:
    """Write `last.pt` atomically.

    The best weights go to a file named after their epoch, written before `last.pt` points at
    it: with a single `best.pt`, a crash between the two writes would leave a `last.pt` whose
    best epoch no longer matches the weights on disk.
    """
    directory.mkdir(parents=True, exist_ok=True)
    best_file = _best_name(state.best_epoch) if state.best_epoch is not None else None
    if (
        best_file is not None
        and state.best_state is not None
        and not (directory / best_file).exists()
    ):
        _atomic_save(state.best_state, directory / best_file)
    device = _device_of(model)
    payload: dict[str, object] = {
        "version": CHECKPOINT_VERSION,
        "fingerprint": fingerprint,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng_cpu": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "generator": generator.get_state(),
        "epoch": state.epoch,
        "position": state.position,
        "order": torch.tensor(state.order, dtype=torch.long) if state.order is not None else None,
        "loss_sum": state.loss_sum,
        "seen": state.seen,
        "epoch_losses": list(state.epoch_losses),
        "step": state.step,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "bad_epochs": state.bad_epochs,
        "stopped_early": state.stopped_early,
        "best_file": best_file,
    }
    _atomic_save(payload, directory / LAST)
    for path in _checkpoint_files(directory):
        if path.name.startswith(BEST_PREFIX) and path.name != best_file:
            path.unlink(missing_ok=True)


def _load(
    path: Path,
    fingerprint: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    device: torch.device,
) -> _State:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, EOFError) as exc:
        raise StudentError(
            f"checkpoint {path} is unreadable ({exc}); delete it to start over"
        ) from exc
    if ckpt.get("version") != CHECKPOINT_VERSION:
        raise StudentError(
            f"checkpoint {path} has format {ckpt.get('version')}, this Anvil writes"
            f" {CHECKPOINT_VERSION}; delete it to start over"
        )
    if ckpt["fingerprint"] != fingerprint:
        raise StudentError(
            f"checkpoint {path} was written with a different training config or dataset;"
            " restore the original settings to resume, or delete the checkpoint (or resume"
            " without --resume) to start over"
        )
    try:
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if ckpt["scaler"]:
            scaler.load_state_dict(ckpt["scaler"])
    except (RuntimeError, KeyError, ValueError) as exc:
        raise StudentError(
            f"checkpoint {path} does not fit this model ({str(exc).splitlines()[0]}); the"
            " student changed since the run started, so delete the checkpoint to start over"
        ) from exc
    torch.set_rng_state(ckpt["rng_cpu"])
    if ckpt["rng_cuda"] is not None and device.type == "cuda":
        torch.cuda.set_rng_state(ckpt["rng_cuda"], device)
    generator.set_state(ckpt["generator"])
    best_state: StateDict | None = None
    if ckpt["best_file"] is not None:
        best_path = path.parent / str(ckpt["best_file"])
        if not best_path.exists():
            raise StudentError(
                f"checkpoint {path} refers to {best_path.name}, which is missing; delete the"
                " checkpoint directory to start over"
            )
        best_state = torch.load(best_path, map_location="cpu", weights_only=True)
    order = ckpt["order"]
    return _State(
        epoch=int(ckpt["epoch"]),
        position=int(ckpt["position"]),
        order=order.tolist() if order is not None else None,
        loss_sum=float(ckpt["loss_sum"]),
        seen=int(ckpt["seen"]),
        epoch_losses=[float(x) for x in ckpt["epoch_losses"]],
        step=int(ckpt["step"]),
        best_metric=ckpt["best_metric"],
        best_epoch=ckpt["best_epoch"],
        bad_epochs=int(ckpt["bad_epochs"]),
        stopped_early=bool(ckpt["stopped_early"]),
        best_state=best_state,
    )
