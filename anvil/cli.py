"""Command-line interface: `anvil label | train | eval`."""

from __future__ import annotations

import sys
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from anvil import __version__, log
from anvil.errors import AnvilError
from anvil.settings import Settings
from anvil.spec import TaskSpec, load_spec

app = typer.Typer(
    name="anvil",
    help="Distil an LLM call into a small, fast task model, with measured parity.",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
)


class TeacherChoice(StrEnum):
    claude = "claude"
    fake = "fake"


SpecOpt = Annotated[Path, typer.Option("--spec", "-s", help="Task spec (anvil.yaml).")]
TeacherOpt = Annotated[
    TeacherChoice | None,
    typer.Option("--teacher", help="Override teacher.provider (fake runs offline)."),
]


def _run[T](fn: Callable[[], T]) -> T:
    """Run a command body, mapping Anvil errors to a one-line message and an exit code."""
    try:
        return fn()
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise typer.Exit(exc.exit_code) from None


def _setup(spec_path: Path) -> tuple[TaskSpec, Settings]:
    settings = Settings.from_env()
    log.configure(settings.log_level)
    return load_spec(spec_path), settings


def _teacher_model(spec: TaskSpec, teacher: TeacherChoice | None) -> str:
    from anvil.teacher.fake import FakeTeacher

    provider = teacher.value if teacher else spec.teacher.provider
    return FakeTeacher.model_id_for(spec) if provider == "fake" else spec.teacher.model


@app.callback(invoke_without_command=True)
def main(
    version: Annotated[bool, typer.Option("--version", help="Print the version.")] = False,
) -> None:
    if version:
        print(__version__)
        raise typer.Exit(0)


@app.command("label")
def label_cmd(
    spec: SpecOpt = Path("anvil.yaml"),
    inputs: Annotated[
        Path | None, typer.Option("--inputs", "-i", help="JSONL of {text, id?} to add first.")
    ] = None,
    n: Annotated[int, typer.Option("--n", "-n", min=0, help="Pool inputs to label.")] = 100,
    heldout_max: Annotated[
        int | None, typer.Option(help="Cap on labelled held-out inputs (default: all).")
    ] = None,
    teacher: TeacherOpt = None,
) -> None:
    """Label the held-out split and N more pool inputs with the teacher."""

    def body() -> None:
        from anvil.pipeline import Workspace, label
        from anvil.teacher import build_teacher

        task, settings = _setup(spec)
        t = build_teacher(task, settings, provider=teacher.value if teacher else None)
        r = label(task, Workspace.for_task(settings.home, task), t, inputs, n, heldout_max)
        print(
            f"labelled {r.heldout_labelled} held-out + {r.pool_labelled} pool inputs with"
            f" {r.teacher_model} ({r.cache_hits} from cache, ${r.spent_usd:.4f} spent);"
            f" {r.total_labels} labels in store worth ${r.total_label_cost_usd:.4f}"
        )
        for f in r.failures[:10]:
            print(f"  failed {f.input_id}: {f.reason}", file=sys.stderr)
        if r.failures:
            print(f"{len(r.failures)} inputs failed; re-run to retry them", file=sys.stderr)
            raise typer.Exit(1)

    _run(body)


@app.command("train")
def train_cmd(spec: SpecOpt = Path("anvil.yaml"), teacher: TeacherOpt = None) -> None:
    """Train a fresh student on every labelled pool input."""

    def body() -> None:
        from anvil.pipeline import Workspace, train

        task, settings = _setup(spec)
        r = train(task, Workspace.for_task(settings.home, task), _teacher_model(task, teacher))
        losses = r.result.epoch_losses
        print(
            f"trained {r.run_id} on {r.result.examples} {r.teacher_model} labels:"
            f" {r.result.epochs} epochs, loss {losses[0]:.3f} -> {losses[-1]:.3f},"
            f" {r.result.seconds:.1f}s"
        )

    _run(body)


@app.command("eval")
def eval_cmd(
    spec: SpecOpt = Path("anvil.yaml"),
    student: Annotated[
        Path | None, typer.Option(help="Student directory (default: the latest).")
    ] = None,
) -> None:
    """Measure the student's agreement with the teacher on the held-out split."""

    def body() -> None:
        from anvil.pipeline import Workspace, evaluate

        task, settings = _setup(spec)
        r = evaluate(task, Workspace.for_task(settings.home, task), student)
        a = r.agreement
        print(
            f"student {r.student} vs {r.teacher_model} on {a.n} held-out inputs:"
            f" kappa {a.kappa:.3f}, accuracy {a.accuracy:.3f}"
            f" (majority '{a.majority_label}' accuracy {a.majority_accuracy:.3f});"
            f" report {r.report_path}"
        )

    _run(body)


if __name__ == "__main__":
    app()
