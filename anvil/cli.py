"""Command-line interface: `anvil label | train | eval | loop`."""

from __future__ import annotations

import sys
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from anvil import __version__, log
from anvil.errors import AnvilError, ConfigError
from anvil.select import STRATEGIES
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
def train_cmd(
    spec: SpecOpt = Path("anvil.yaml"),
    teacher: TeacherOpt = None,
    resume: Annotated[
        str | None,
        typer.Option(help="Continue an interrupted run (its run id) from its checkpoint."),
    ] = None,
) -> None:
    """Train a fresh student on every labelled pool input."""

    def body() -> None:
        from anvil.pipeline import Workspace, train

        task, settings = _setup(spec)
        r = train(
            task, Workspace.for_task(settings.home, task), _teacher_model(task, teacher), resume
        )
        res = r.result
        losses = res.epoch_losses
        metric = "entity F1" if task.kind == "extract" else "kappa"
        stopping = (
            f" ({res.validation_examples} held back for early stopping, best epoch"
            f" {res.best_epoch}, {metric} {res.best_metric:.3f})"
            if res.validation_examples and res.best_metric is not None
            else ""
        )
        if r.alignment is not None:
            a = r.alignment
            repairs = ", ".join(f"{k} {v}" for k, v in a.matched.items() if v and k != "exact")
            print(
                f"aligned {a.mentions} teacher mentions: {a.matched['exact']} at their offset,"
                f" {a.repaired} repaired ({repairs or 'none'}), {a.unaligned} dropped"
                f" ({a.unaligned_rate:.1%}), {a.snapped} snapped to token boundaries"
            )
        print(
            f"trained {r.run_id} on {res.examples + res.validation_examples} {r.teacher_model}"
            f" labels{stopping}: {res.epochs} epochs, loss {losses[0]:.3f} -> {losses[-1]:.3f},"
            f" {res.seconds:.1f}s"
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
        from anvil.evaluate.extraction import ExtractionAgreement
        from anvil.pipeline import Workspace, evaluate

        task, settings = _setup(spec)
        r = evaluate(task, Workspace.for_task(settings.home, task), student)
        a = r.agreement
        if isinstance(a, ExtractionAgreement):
            e, f1, tk = a.entity, a.entity_f1_ci, a.token_kappa_ci
            ha = r.heldout_alignment
            aligned = (
                f" (held-out teacher mentions: {ha.repaired} repaired, {ha.unaligned} dropped,"
                f" {ha.unaligned_rate:.1%})"
                if ha is not None
                else ""
            )
            print(
                f"student {r.student} vs {r.teacher_model} on {a.n} held-out inputs:"
                f" entity F1 {e.f1:.3f} [{f1.low:.3f}, {f1.high:.3f}]"
                f" (precision {e.precision:.3f}, recall {e.recall:.3f}),"
                f" token kappa {a.token_kappa:.3f} [{tk.low:.3f}, {tk.high:.3f}]{aligned};"
                f" report {r.report_path}"
            )
            return
        print(
            f"student {r.student} vs {r.teacher_model} on {a.n} held-out inputs:"
            f" kappa {a.kappa:.3f} [{a.kappa_ci.low:.3f}, {a.kappa_ci.high:.3f}],"
            f" accuracy {a.accuracy:.3f}"
            f" (majority '{a.majority_label}' accuracy {a.majority_accuracy:.3f});"
            f" report {r.report_path}"
        )

    _run(body)


def _budgets(raw: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in raw.split(",") if x.strip())
    except ValueError:
        raise ConfigError(f"--budgets {raw!r} must be comma-separated label counts") from None


@app.command("loop")
def loop_cmd(
    spec: SpecOpt = Path("anvil.yaml"),
    inputs: Annotated[
        Path | None, typer.Option("--inputs", "-i", help="JSONL of {text, id?} to add first.")
    ] = None,
    strategy: Annotated[
        str, typer.Option(help=f"Selection strategy: {', '.join(STRATEGIES)}.")
    ] = "random",
    budgets: Annotated[
        str, typer.Option(help="Cumulative pool labels after each round, comma-separated.")
    ] = "100,200,400,800,1600",
    max_usd: Annotated[
        float | None, typer.Option(help="Dollar budget for every label the loop uses.")
    ] = None,
    target_kappa: Annotated[
        float | None,
        typer.Option(help="Stop once the lower end of the kappa interval reaches this."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Seeds selection and the student.")] = 0,
    heldout_max: Annotated[
        int | None, typer.Option(help="Cap on labelled held-out inputs (default: all).")
    ] = None,
    loop_id: Annotated[
        str | None,
        typer.Option("--id", help="Loop id; an existing id resumes that loop."),
    ] = None,
    teacher: TeacherOpt = None,
) -> None:
    """Grow the training set round by round: select, label, retrain from scratch, evaluate."""

    def body() -> None:
        from anvil.loop import LoopConfig, run_loop
        from anvil.pipeline import Workspace
        from anvil.teacher import build_teacher

        task, settings = _setup(spec)
        config = LoopConfig(
            strategy=strategy,
            budgets=_budgets(budgets),
            seed=seed,
            max_usd=max_usd,
            target_kappa=target_kappa,
            heldout_max=heldout_max,
        )
        t = build_teacher(task, settings, provider=teacher.value if teacher else None)
        r = run_loop(
            task,
            Workspace.for_task(settings.home, task),
            t,
            config,
            loop_id=loop_id,
            inputs=inputs,
        )
        print(f"loop {r.loop_id} ({strategy}, seed {seed}) vs {t.model_id}:")
        print("  round  labels  kappa [95% CI]          label cost  spent")
        for row in r.rows:
            print(
                f"  {row['round']:>5}  {row['labels']:>6}  {row['kappa']:.3f}"
                f" [{row['kappa_low']:.3f}, {row['kappa_high']:.3f}]"
                f"  ${row['label_cost_usd']:>9.4f}  ${row['spent_usd']:.4f}"
            )
        print(f"stopped: {r.stopped or 'interrupted'}; frontier {r.frontier}; student {r.student}")

    _run(body)


if __name__ == "__main__":
    app()
