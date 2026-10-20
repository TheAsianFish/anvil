"""The trained classify student the M5 studies share (calibration, export, bench).

`make eval-classify` trains its student in the eval workspace (`.anvil/eval/<task>-<sample>/`)
and keeps it there. The calibration, export and bench studies measure that same student, so
they look it up rather than train their own: a different student would make their numbers
incomparable with the parity table. When it is missing (a fresh clone, a cleaned workspace),
`trained_student` labels and trains it exactly as `eval.classify` does, minus the baselines
and the ceiling, so a clean run reproduces it; on CPU that is the training time quoted in the
parity table (about an hour for Banking77 with ModernBERT-base).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from anvil.errors import ConfigError
from anvil.log import get_logger
from anvil.pipeline import Workspace, label, train
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.store import Split
from anvil.students.encoder import META_FILE
from anvil.teacher.base import Item
from eval.classify import (
    CONFIG_DIR,
    EvalConfig,
    TeacherKind,
    load_config,
    make_teacher,
    require_complete,
    resolve_teacher_kind,
    workspace,
)
from eval.data import Sample, load_rows, sample

logger = get_logger("anvil.eval.student")


@dataclass(frozen=True)
class TrainedStudent:
    config: EvalConfig
    data: Sample
    workspace: Workspace
    kind: TeacherKind
    teacher_model: str
    path: Path
    heldout: list[tuple[Item, str]]
    """Held-out inputs with their teacher labels, in store order."""
    trained_here: bool
    """True when this call trained the student (it was not in the workspace)."""
    spent_usd: float = 0.0
    """Dollars this call spent labelling (only when it had to label and train)."""

    @property
    def spec(self) -> TaskSpec:
        return self.config.spec


def _matches(
    path: Path, spec: TaskSpec, teacher_model: str, pool_labels: int
) -> Literal["yes", "no", "unknown"]:
    """Whether the student in `path` is the one `spec` trains: "unknown" when everything
    recorded matches but the student predates `student_spec` in its metadata, so its seed and
    epochs cannot be checked."""
    meta_path = path / META_FILE
    if not meta_path.exists():
        return "no"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    train_labels = meta.get("train_labels")
    same = (
        meta.get("teacher_model") == teacher_model
        and meta.get("prompt_hash") == spec.prompt_hash
        and meta.get("student_model") == spec.student.model
        and meta.get("splits") == spec.splits.model_dump()
        and isinstance(train_labels, list)
        and len(train_labels) == pool_labels
    )
    if not same:
        return "no"
    if "student_spec" not in meta:
        return "unknown"
    return "yes" if meta["student_spec"] == spec.student.model_dump() else "no"


def find_student(
    ws: Workspace, spec: TaskSpec, teacher_model: str, pool_labels: int
) -> Path | None:
    """The student in `ws` trained on `pool_labels` labels from this teacher, prompt, split and
    student settings (seed and epochs included); None when there is none.

    A student saved before the settings were recorded is accepted only when it is the one
    candidate: a seed repeat shares the workspace, and picking the newest of two would measure
    the wrong one."""
    if not ws.students.exists():
        return None
    runs = sorted((p for p in ws.students.iterdir() if p.is_dir()), reverse=True)
    verdicts = {p: _matches(p, spec, teacher_model, pool_labels) for p in runs}
    exact = [p for p, v in verdicts.items() if v == "yes"]
    if exact:
        return exact[0]
    unknown = [p for p, v in verdicts.items() if v == "unknown"]
    if len(unknown) > 1:
        raise ConfigError(
            f"{len(unknown)} students in {ws.students} match this eval but record no seed:"
            f" {', '.join(p.name for p in unknown)}; move the ones that are not the"
            f" seed-{spec.student.seed} run aside"
        )
    return unknown[0] if unknown else None


def trained_student(name: str, teacher: TeacherKind, settings: Settings) -> TrainedStudent:
    """The `eval.classify` student for dataset `name`, trained now if the workspace lacks it."""
    config = load_config(CONFIG_DIR / f"{name}.yaml")
    data = sample(
        name, *load_rows(config.dataset.ref()), n=config.sample.size, seed=config.sample.seed
    )
    kind = resolve_teacher_kind(teacher, config)
    spec = config.spec
    ws = workspace(settings, spec, data)
    t = make_teacher(kind, config, spec, data, settings)
    with ws.open_store(spec) as store:
        pool_done = len(store.labelled(t.model_id, spec.prompt_hash, Split.POOL))
    found = (
        find_student(ws, spec, t.model_id, config.sample.pool_labels)
        if pool_done >= config.sample.pool_labels
        else None
    )
    trained_here = found is None
    spent = 0.0
    if found is None:
        inputs = ws.root / "inputs.jsonl"
        from eval.data import write_inputs

        write_inputs(inputs, data.items)
        labelled = label(
            spec,
            ws,
            t,
            inputs,
            max(0, config.sample.pool_labels - pool_done),
            config.sample.heldout_labels,
        )
        spent = labelled.spent_usd
        require_complete(kind, labelled.failures, config, spec)
        logger.info("training the eval student", extra={"dataset": name, "teacher": t.model_id})
        found = train(spec, ws, t.model_id).path
    with ws.open_store(spec) as store:
        heldout = store.labelled(t.model_id, spec.prompt_hash, Split.HELDOUT)
    return TrainedStudent(config, data, ws, kind, t.model_id, found, heldout, trained_here, spent)
