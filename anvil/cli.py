"""Command-line interface:
`anvil label | train | eval | loop | calibrate | export | serve | push`.

Top-level imports stay torch-free: `anvil serve` runs in an image without torch, so every
command imports its heavy dependencies inside its body."""

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
    ollama = "ollama"
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
        metric = {
            "extract": "entity F1",
            "generate": "validation score (exact match, loss as tie-break)",
        }.get(task.kind, "kappa")
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
        if r.chat is not None:
            c = r.chat
            dropped = ", ".join(f"{k} {v}" for k, v in c.dropped.items() if v)
            print(
                f"aligned {c.examples} answers to the chat template: {c.unusable} dropped"
                f" ({dropped or 'none'}, {c.dropped_rate:.1%}); longest sequence {c.longest}"
                " tokens"
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
        from anvil.evaluate.generation import GenerationAgreement
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
        if isinstance(a, GenerationAgreement):
            em = a.exact_match
            fields = ", ".join(f"{f.name} {f.match_rate:.3f}" for f in a.field_match)
            print(
                f"student {r.student} vs {r.teacher_model} on {a.n} held-out inputs: exact match"
                f" {em.estimate:.3f} [{em.low:.3f}, {em.high:.3f}] (per field: {fields});"
                f" {a.invalid} invalid answers; report {r.report_path}"
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


class Rule(StrEnum):
    lower_bound = "lower"  # not `lower`: that would shadow str.lower
    point = "point"


@app.command("calibrate")
def calibrate_cmd(
    spec: SpecOpt = Path("anvil.yaml"),
    student: Annotated[
        Path | None, typer.Option(help="Student directory (default: the latest).")
    ] = None,
    target_kappa: Annotated[
        float,
        typer.Option(help="Kappa with the teacher the student must keep on what it answers."),
    ] = 0.9,
    rule: Annotated[
        Rule,
        typer.Option(
            help="lower: the 95% bootstrap lower bound must reach the target; point:"
            " the estimate must."
        ),
    ] = Rule.lower_bound,
) -> None:
    """Fit a temperature and the escalation threshold on the held-out teacher labels."""

    def body() -> None:
        from anvil.pipeline import Workspace, calibrate_student

        task, settings = _setup(spec)
        r = calibrate_student(
            task,
            Workspace.for_task(settings.home, task),
            student,
            target_kappa=target_kappa,
            rule=rule.value,
        )
        c = r.calibration
        print(
            f"calibrated {r.student} on {c.n} held-out {r.teacher_model} labels: temperature"
            f" {c.temperature:.3f}, ECE {c.ece_before:.3f} -> {c.ece_after:.3f}"
        )
        if c.threshold is None:
            print(
                f"no threshold keeps kappa {target_kappa} ({rule.value} rule) on at least 30"
                " held-out inputs: the cascade would escalate everything; label more or lower"
                " the target"
            )
        else:
            low = f", lower bound {c.kappa_low:.3f}" if c.kappa_low is not None else ""
            print(
                f"threshold {c.threshold:.4f}: the student answers {c.coverage:.1%} of held-out"
                f" inputs at kappa {c.kappa:.3f}{low} (target {target_kappa}); written to {r.path}"
            )

    _run(body)


@app.command("export")
def export_cmd(
    spec: SpecOpt = Path("anvil.yaml"),
    student: Annotated[
        Path | None, typer.Option(help="Student directory (default: the latest).")
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Bundle directory (default: <task>/exports/<student>)."),
    ] = None,
    int8: Annotated[
        bool, typer.Option("--int8", help="Also export a dynamic int8 model.")
    ] = False,
    probes: Annotated[
        int,
        typer.Option(
            min=1, help="Held-out inputs the parity check runs on (generators: at most 64)."
        ),
    ] = 512,
) -> None:
    """Export the student and check it before it can be served: an ONNX bundle for a
    classifier, the adapter merged into its base model for a generator."""

    def body() -> None:
        from anvil.pipeline import Workspace, export_bundle, export_merged

        task, settings = _setup(spec)
        if task.kind == "generate":
            if int8:
                raise ConfigError("--int8 is for ONNX classifier bundles, not merged generators")
            m = export_merged(
                task, Workspace.for_task(settings.home, task), student, out, probes=min(probes, 64)
            )
            mp = m.parity
            print(
                f"merged: passed parity vs the adapter on {mp.probes} held-out inputs:"
                f" {mp.same_completions}/{mp.probes} identical completions, max |logit diff|"
                f" {mp.max_abs_logit_diff:.2e} on {mp.logit_probes}; {m.size_mb:.1f} MB"
                f" ({mp.gate})"
            )
            print(f"merged model {m.out} (a plain transformers checkpoint)")
            return
        r = export_bundle(
            task, Workspace.for_task(settings.home, task), student, out, int8=int8, probes=probes
        )
        for p in (r.fp32, r.int8):
            if p is None:
                continue
            kappa = (
                f", kappa vs teacher {p.kappa:.3f} (drop {p.kappa_drop:+.3f})"
                if p.kappa is not None and p.kappa_drop is not None
                else ""
            )
            answered = (
                f" ({p.agreement_answered:.3f} on the {p.answered} probe runs either graph"
                " answers)"
                if p.agreement_answered is not None and p.answered is not None
                else ""
            )
            print(
                f"{p.variant}: {'passed' if p.passed else 'FAILED'} parity vs {p.reference} on"
                f" {p.n} inputs: max |logit diff| {p.max_abs_diff:.2e}, agreement"
                f" {p.agreement:.3f}{answered}{kappa}; {p.size_mb:.1f} MB ({p.gate})"
            )
        print(f"bundle {r.bundle}; serve it with `anvil serve {r.bundle}`")
        if r.int8 is not None and not r.int8.passed:
            print("the int8 model failed its parity gate and was not kept", file=sys.stderr)
            raise typer.Exit(1)

    _run(body)


class ServeTeacher(StrEnum):
    auto = "auto"
    claude = "claude"
    ollama = "ollama"
    fake = "fake"
    none = "none"


class Variant(StrEnum):
    fp32 = "fp32"
    int8 = "int8"


@app.command("serve")
def serve_cmd(
    bundle: Annotated[Path, typer.Argument(help="Export bundle directory (`anvil export`).")],
    host: Annotated[str, typer.Option(help="Interface to bind.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port to listen on.")] = 8000,
    variant: Annotated[Variant, typer.Option(help="Model variant in the bundle.")] = Variant.fp32,
    threshold: Annotated[
        float | None,
        typer.Option(help="Escalate inputs below this confidence (default: the bundle's)."),
    ] = None,
    teacher: Annotated[
        ServeTeacher,
        typer.Option(
            help="Escalation teacher; auto uses the spec's local model when it names one,"
            " else Claude when credentials exist."
        ),
    ] = ServeTeacher.auto,
) -> None:
    """Serve a student over HTTP, escalating inputs it is unsure of to the teacher."""

    def body() -> None:
        import uvicorn

        from anvil.serve import load_app

        settings = Settings.from_env()
        log.configure(settings.log_level)
        api = load_app(
            bundle,
            variant=variant.value,
            teacher=teacher.value,
            threshold=threshold,
            settings=settings,
        )
        # No uvicorn log config or access log: our JSON `predict` line covers each request.
        uvicorn.run(api, host=host, port=port, log_config=None, access_log=False)

    _run(body)


@app.command("push")
def push_cmd(
    artifact: Annotated[Path, typer.Argument(help="Exported student directory (`anvil export`).")],
    repo: Annotated[str, typer.Option("--repo", help="Hugging Face model repo, namespace/name.")],
    public: Annotated[
        bool, typer.Option("--public", help="Create the repo public (default: private).")
    ] = False,
    report: Annotated[
        Path | None,
        typer.Option(help="Eval report for the card (default: found next to the export)."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Write the model card and list the files; no upload."),
    ] = False,
) -> None:
    """Publish an exported student to the Hugging Face Hub with a model card rendered from
    its measured results (needs HF_TOKEN)."""

    def body() -> None:
        from anvil.hub import push

        settings = Settings.from_env()
        log.configure(settings.log_level)
        r = push(
            artifact,
            repo,
            home=settings.home,
            token=settings.hf_token,
            private=not public,
            dry_run=dry_run,
            report=report,
        )
        print(f"model card {r.card}")
        if r.dry_run:
            print(f"dry run: {len(r.files)} files would be uploaded to {r.repo_id}:")
            for f in r.files:
                print(f"  {f}")
        else:
            print(f"pushed {len(r.files)} files to {r.url}")

    _run(body)


if __name__ == "__main__":
    app()
