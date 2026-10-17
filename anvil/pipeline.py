"""The `label -> train -> eval` steps as a library API (the CLI is a thin wrapper)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from anvil.errors import StoreError, StudentError
from anvil.evaluate.metrics import Agreement, agreement
from anvil.log import get_logger
from anvil.spec import TaskSpec
from anvil.store import Split, Store, read_inputs
from anvil.students.encoder import EncoderClassifier
from anvil.teacher.base import BatchAbortedError, Failure, LabelBatch, Teacher
from anvil.train.classifier import train_classifier
from anvil.train.loop import TrainResult

log = get_logger(__name__)


@dataclass(frozen=True)
class Workspace:
    """Where one task's state lives: `<home>/<task>/`."""

    root: Path

    @classmethod
    def for_task(cls, home: Path, spec: TaskSpec) -> Workspace:
        return cls(home / spec.name)

    @property
    def db(self) -> Path:
        return self.root / "store.sqlite"

    @property
    def students(self) -> Path:
        return self.root / "students"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def open_store(self, spec: TaskSpec) -> Store:
        return Store(self.db, spec.splits.seed, spec.splits.heldout_fraction)

    def latest_student(self) -> Path:
        pointer = self.students / "LATEST"
        if not pointer.exists():
            raise StoreError(f"no trained student under {self.students}; run `anvil train` first")
        return self.students / pointer.read_text(encoding="utf-8").strip()


@dataclass
class LabelReport:
    teacher_model: str
    inputs_added: int
    heldout_labelled: int
    pool_labelled: int
    failures: list[Failure] = field(default_factory=list)
    spent_usd: float = 0.0
    cache_hits: int = 0
    total_labels: int = 0
    total_label_cost_usd: float = 0.0


def label(
    spec: TaskSpec,
    ws: Workspace,
    teacher: Teacher,
    inputs: Path | None,
    pool_n: int,
    heldout_max: int | None = None,
) -> LabelReport:
    """Ingest inputs, label the held-out split (up to `heldout_max`), then `pool_n` more pool
    inputs in the store's fixed pseudo-random order (the random-sampling baseline)."""
    with ws.open_store(spec) as store:
        added = store.add_inputs(read_inputs(inputs)) if inputs is not None else 0
        done = store.labels(teacher.model_id, spec.prompt_hash)

        heldout = [it for it in store.inputs(Split.HELDOUT) if it.id not in done]
        if heldout_max is not None:
            already = sum(1 for it in store.inputs(Split.HELDOUT) if it.id in done)
            heldout = heldout[: max(0, heldout_max - already)]
        pool = [it for it in store.inputs(Split.POOL) if it.id not in done][:pool_n]

        report = LabelReport(teacher.model_id, added, 0, 0)

        def persist(batch: LabelBatch) -> None:
            store.append_labels(batch.labels)
            store.append_failures(batch.failures, teacher.model_id, spec.prompt_hash)

        for name, items in (("heldout", heldout), ("pool", pool)):
            if not items:
                continue
            try:
                batch = teacher.label(items)
            except BatchAbortedError as exc:
                # Keep every answer that was paid for before the run stopped.
                persist(exc.batch)
                log.warning(
                    "label run aborted",
                    extra={"kept": len(exc.batch.labels), "spent_usd": exc.batch.spent_usd},
                )
                raise exc.cause from None
            persist(batch)
            report.failures += batch.failures
            report.spent_usd += batch.spent_usd
            report.cache_hits += batch.cache_hits
            if name == "heldout":
                report.heldout_labelled = len(batch.labels)
            else:
                report.pool_labelled = len(batch.labels)
        stats = store.stats(teacher.model_id, spec.prompt_hash)
        report.total_labels = stats.labels
        report.total_label_cost_usd = stats.cost_usd
    return report


@dataclass(frozen=True)
class TrainReport:
    run_id: str
    path: Path
    teacher_model: str
    result: TrainResult


def train(
    spec: TaskSpec, ws: Workspace, teacher_model: str, resume: str | None = None
) -> TrainReport:
    """Train a fresh student on every labelled pool input under (teacher model, prompt hash).

    The loop checkpoints at every epoch under `<run>/checkpoint/`; `resume` names a run that
    was interrupted and continues it exactly where it stopped. The checkpoint is removed once
    the run completes.
    """
    with ws.open_store(spec) as store:
        pairs = store.labelled(teacher_model, spec.prompt_hash, Split.POOL)
        labelled_by = store.teacher_models(spec.prompt_hash)
        corpus = [it.text for it in store.inputs(Split.POOL)]
    if not pairs:
        others = sorted(set(labelled_by) - {teacher_model})
        hint = (
            f"; the store has labels from {', '.join(others)} (pass --teacher to use them)"
            if others
            else "; run `anvil label` first"
        )
        raise StoreError(f"no pool labels from {teacher_model} for this prompt{hint}")
    texts = [it.text for it, _ in pairs]
    index = {name: i for i, name in enumerate(spec.label_names)}
    targets = [index[lab] for _, lab in pairs]

    if resume is not None:
        run_id = resume
        out = ws.students / run_id
        if not (out / "checkpoint").is_dir():
            raise StoreError(
                f"no checkpoint to resume under {out}; it finished or never started"
                " (run `anvil train` without --resume)"
            )
    else:
        base = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-n{len(pairs)}"
        run_id, k = base, 1
        while (ws.students / run_id).exists():
            k += 1
            run_id = f"{base}-{k}"
        out = ws.students / run_id
        out.mkdir(parents=True)
    # The tiny student learns its vocabulary from pool texts only, never the held-out split.
    student = EncoderClassifier.build(spec.student, spec.label_names, corpus)
    result = train_classifier(
        student,
        texts,
        targets,
        spec.student,
        out / "metrics.jsonl",
        checkpoint_dir=out / "checkpoint",
        resume=resume is not None,
    )
    student.meta.update(
        {
            "task": spec.name,
            "teacher_model": teacher_model,
            "prompt_hash": spec.prompt_hash,
            "splits": spec.splits.model_dump(),
            "student_model": spec.student.model,
            "train_labels": [lab for _, lab in pairs],
            "train": {k: v for k, v in asdict(result).items()},
        }
    )
    student.save(out)
    (ws.students / "LATEST").write_text(run_id + "\n", encoding="utf-8")
    return TrainReport(run_id, out, teacher_model, result)


@dataclass(frozen=True)
class EvalReport:
    student: str
    teacher_model: str
    agreement: Agreement
    report_path: Path


def evaluate(
    spec: TaskSpec,
    ws: Workspace,
    student_dir: Path | None = None,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> EvalReport:
    """Agreement of a trained student with the teacher on the teacher-labelled held-out split.

    `n_resamples` and `seed` drive the kappa bootstrap CI; an eval passes its own so the
    headline interval follows the same settings as every other interval in its report.
    """
    path = student_dir or ws.latest_student()
    student = EncoderClassifier.load(path)
    try:
        task = str(student.meta["task"])
        teacher_model = str(student.meta["teacher_model"])
        prompt_hash = str(student.meta["prompt_hash"])
    except KeyError as exc:
        raise StudentError(
            f"student {path} has no {exc} in its metadata; it was not trained by `anvil train`"
        ) from None
    if task != spec.name:
        raise StoreError(
            f"student {path.name} was trained for task {task!r}, the spec is {spec.name!r};"
            " pass the matching --spec or train a student for this task"
        )
    if prompt_hash != spec.prompt_hash:
        raise StoreError(
            f"student {path.name} was trained for prompt {prompt_hash}, the spec is now"
            f" {spec.prompt_hash}; relabel and retrain"
        )
    if student.meta.get("splits") != spec.splits.model_dump():
        raise StoreError(
            f"student {path.name} was trained with splits {student.meta.get('splits')}, the spec"
            f" now has {spec.splits.model_dump()}; the held-out set would include training"
            " inputs, so retrain before evaluating"
        )
    with ws.open_store(spec) as store:
        pairs = store.labelled(teacher_model, prompt_hash, Split.HELDOUT)
    if not pairs:
        raise StoreError("no teacher labels on the held-out split; run `anvil label` first")
    texts = [it.text for it, _ in pairs]
    teacher = [lab for _, lab in pairs]
    probs = student.predict_proba(texts)
    predicted = [student.labels[i] for i in probs.argmax(axis=1)]
    raw_train = student.meta.get("train_labels", [])
    train_labels = [str(x) for x in raw_train] if isinstance(raw_train, list) else []
    result = agreement(
        teacher,
        predicted,
        spec.label_names,
        train_labels,
        confidence=probs.max(axis=1).tolist(),
        lengths=[len(text.split()) for text in texts],
        n_resamples=n_resamples,
        seed=seed,
    )

    ws.reports.mkdir(parents=True, exist_ok=True)
    report_path = ws.reports / f"eval-{path.name}.json"
    payload = {
        "task": spec.name,
        "student": path.name,
        "student_model": student.meta.get("student_model"),
        "teacher_model": teacher_model,
        "prompt_hash": prompt_hash,
        "labels": spec.label_names,
        "train_examples": len(train_labels),
        **asdict(result),
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log.info("evaluated", extra={"student": path.name, "n": result.n, "kappa": result.kappa})
    return EvalReport(path.name, teacher_model, result, report_path)
