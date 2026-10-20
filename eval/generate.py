"""`make eval-generate`: student-vs-teacher parity on a text-to-SQL task, by execution match.

One run: draw a seeded sample of WikiSQL questions (each with its table), label the held-out
split and the pool with the teacher (one SQLite query per question), label a subset of the
held-out split again under a paraphrased prompt (the self-agreement ceiling), fine-tune a
LoRA adapter on the decoder student, and score its greedy queries against the teacher's on
the held-out split.

The headline is **execution match**: both queries run in a sandboxed in-memory SQLite copy of
the question's own table (`anvil.evaluate.sql`) and match when they return the same rows.
It is objective and forgiving of style (aliases, quoting, keyword case), but it can be fooled
by a small table: two different queries that both return nothing, or happen to return the
same row, match. The report therefore also gives the strict normalised-SQL match, execution
match on the questions whose reference returns at least one row, and an LLM judge's verdict
on a seeded sample of 100 pairs, compared with execution match item by item (the judge needs
a key; without one it is reported as skipped). Baseline: the same base model with no
adapter, prompted the same way (what fine-tuning adds). Slices: aggregate or not, number of
conditions, input length. Finally the adapter is merged into the base weights and the
merged model must answer the probe questions exactly as the adapter did.

Teachers are as in `eval.classify` (`claude`, `replay`, `gold`, `auto`); with `gold` the
dataset's own queries are the teacher (model id `gold`). Raw output goes to
`eval/results/generate-<dataset>-<teacher>/` (`report.json`, `meta.json`,
`train-metrics.jsonl`); `python -m eval.render` builds the RESULTS.md table. Overrides
(`--student`, `--epochs`, `--size`, `--max-length`, `--resamples`, `--no-baseline`) make a
smoke run under
`...-smoke/`.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import shutil
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from anvil import __version__, log
from anvil.errors import AnvilError, ConfigError
from anvil.evaluate.generation import mean_ci, paired_difference_ci
from anvil.evaluate.judge import (
    ClaudeJudge,
    Judge,
    JudgeItem,
    RecordedThenLiveJudge,
    Verdict,
    VerdictFileJudge,
    judge_sample,
    judge_vs_metric,
    rubric_hash,
    spent_usd,
    write_verdict_file,
)
from anvil.evaluate.sql import ExecMatch, Table, execution_match, extract_sql, normalise_sql
from anvil.generation import answers, student_system_prompt
from anvil.merge import export_generator
from anvil.pipeline import Workspace, label, train
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.store import Split
from anvil.students.generator import LoraGenerator
from anvil.teacher.base import Item
from anvil.teacher.cache import DiskCache
from eval.classify import (
    DEFAULT_RESAMPLES,
    LABELS_DIR,
    CeilingCfg,
    TeacherKind,
    _shown,
    _versions,
    git_provenance,
    label_file,
    make_teacher,
    record_labels,
    require_complete,
    resolve_teacher_kind,
)
from eval.data import Sample, near_duplicates, write_inputs
from eval.render import SMOKE_SUFFIX
from eval.wikisql import SqlExample, WikiSqlRef, load_examples, sample_examples

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "eval" / "generate"
RESULTS_DIR = ROOT / "eval" / "results"

logger = log.get_logger("anvil.eval.generate")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetCfg(_Strict):
    name: str
    repo: str
    revision: str
    filename: str
    license: str = ""

    def ref(self) -> WikiSqlRef:
        return WikiSqlRef(self.repo, self.revision, self.filename, self.license)


class SampleCfg(_Strict):
    size: int = Field(ge=10)
    pool_labels: int = Field(ge=1)
    heldout_labels: int = Field(ge=1)
    seed: int = 0
    max_columns: int | None = Field(default=None, ge=1)
    """Leave out tables wider than this: every column is in the prompt, and a 44-column table
    is a prompt three times the median's length."""


class JudgeCfg(_Strict):
    n: int = Field(default=100, ge=0)
    model: str = "claude-haiku-5-5"
    seed: int = 0


class BaselineCfg(_Strict):
    zero_shot: bool = True
    """Score the base model with no adapter on the same prompts (what fine-tuning adds)."""


class ExportCfg(_Strict):
    probes: int = Field(default=64, ge=0)
    """Held-out questions the merged model must answer exactly as the adapter (0 skips)."""


class EvalConfig(_Strict):
    dataset: DatasetCfg
    sample: SampleCfg
    ceiling: CeilingCfg
    judge: JudgeCfg = JudgeCfg()
    baseline: BaselineCfg = BaselineCfg()
    export: ExportCfg = ExportCfg()
    spec: TaskSpec

    def paraphrase_spec(self) -> TaskSpec:
        return self.spec.model_copy(update={"instructions": self.ceiling.instructions})


def load_config(path: Path) -> EvalConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"eval config {path} not found") from exc
    config = EvalConfig.model_validate(raw)
    TaskSpec.model_validate(config.spec.model_dump())
    if config.spec.kind != "generate" or config.spec.output_names != ["sql"]:
        raise ConfigError(f"{path} must hold a generate task with one output field, `sql`")
    return config


# --- scoring ----------------------------------------------------------------------------


def doubled(table: Table) -> Table:
    """The table with every row twice. A query that differs from the reference only in a way
    one copy of the rows hides (a missing `MAX` over one matching row, `COUNT` against a
    column that happens to hold the count, a missing `DISTINCT`) returns something else here,
    while two equivalent queries still agree: a cheap second database in the spirit of
    test-suite accuracy (Zhong et al., 2020), with no model involved."""
    return Table(table.header, table.types, [*table.rows, *table.rows])


def match_all(
    tables: Sequence[Table], references: Sequence[str], candidates: Sequence[str]
) -> list[ExecMatch]:
    return [
        execution_match(t, r, c) for t, r, c in zip(tables, references, candidates, strict=True)
    ]


def _rate(flags: Sequence[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def _error_kind(error: str | None) -> str | None:
    """The category `anvil.evaluate.sql.execute` puts before the colon of its error text."""
    return None if error is None else error.split(":", 1)[0]


def score_pair(
    tables: Sequence[Table],
    references: Sequence[str],
    candidates: Sequence[str],
    n_resamples: int,
    seed: int,
) -> tuple[dict[str, object], list[ExecMatch]]:
    """Execution match of `candidates` against `references` on each item's table, with its
    interval; the same on the table and its `doubled` copy (both must match); the strict
    normalised-SQL match; the match on non-empty references; and the candidates' execution
    errors by kind."""
    matches = match_all(tables, references, candidates)
    flags = [m.match for m in matches]
    again = match_all([doubled(t) for t in tables], references, candidates)
    strict = [m.match and d.match for m, d in zip(matches, again, strict=True)]
    nonempty = [m.match for m in matches if m.reference.error is None and not m.reference_empty]
    exact = [
        normalise_sql(r) == normalise_sql(c) for r, c in zip(references, candidates, strict=True)
    ]
    errors: dict[str, int] = {}
    for m in matches:
        kind = _error_kind(m.candidate.error)
        if kind is not None:
            errors[kind] = errors.get(kind, 0) + 1
    stats: dict[str, object] = {
        "n": len(flags),
        "execution_match": asdict(mean_ci(flags, n_resamples=n_resamples, seed=seed)),
        "execution_match_doubled": asdict(mean_ci(strict, n_resamples=n_resamples, seed=seed)),
        "coincidental_matches": sum(f and not st for f, st in zip(flags, strict, strict=True)),
        "exact_match": sum(exact) / len(exact),
        "nonempty_reference": len(nonempty),
        "execution_match_nonempty": _rate(nonempty),
        "reference_empty": sum(m.reference_empty for m in matches),
        "reference_errors": sum(m.reference.error is not None for m in matches),
        "candidate_errors": errors,
        "exact_but_not_executed_match": sum(
            e and not m.match for e, m in zip(exact, matches, strict=True)
        ),
        "executed_match_not_exact": sum(
            m.match and not e for e, m in zip(exact, matches, strict=True)
        ),
    }
    return stats, matches


def _slice(
    name: str, keys: Sequence[str], flags: Sequence[bool], order: Sequence[str]
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key in order:
        hits = [f for k, f in zip(keys, flags, strict=True) if k == key]
        if hits:
            rows.append({name: key, "n": len(hits), "execution_match": sum(hits) / len(hits)})
    return rows


def slices(
    examples: Sequence[SqlExample], texts: Sequence[str], flags: Sequence[bool]
) -> dict[str, list[dict[str, object]]]:
    """Execution match by the gold query's shape (aggregate, number of conditions) and by the
    input's length in words (quartile bins). The shape comes from the dataset's query, not the
    teacher's, so a slice means the same thing whichever teacher made the labels."""
    agg = ["aggregate" if ex.agg else "none" for ex in examples]
    conds = [str(min(ex.n_conds, 3)) if ex.n_conds < 3 else "3+" for ex in examples]
    words = np.array([len(t.split()) for t in texts])
    edges = sorted({int(q) for q in np.quantile(words, [0.25, 0.5, 0.75])}) if len(words) else []
    bins = [f"<= {edges[0]}" if edges else "all"]
    bins += [f"{a + 1}-{b}" for a, b in itertools.pairwise(edges)]
    if edges:
        bins.append(f"> {edges[-1]}")

    def length_bin(n: int) -> str:
        for k, edge in enumerate(edges):
            if n <= edge:
                return bins[k]
        return bins[-1]

    lengths = [length_bin(int(n)) for n in words]
    return {
        "aggregate": _slice("aggregate", agg, flags, ["none", "aggregate"]),
        "conditions": _slice("conditions", conds, flags, ["0", "1", "2", "3+"]),
        "length": _slice("words", lengths, flags, bins),
    }


def sql_of(spec: TaskSpec, pairs: Sequence[tuple[Item, str]]) -> list[str]:
    return [a["sql"] for a in answers(spec, pairs)]


# --- ceiling, baseline, judge, export ---------------------------------------------------


def _ceiling(
    kind: TeacherKind,
    config: EvalConfig,
    data: Sample,
    heldout: Sequence[tuple[Item, str]],
    tables: Mapping[str, Table],
    ws: Workspace,
    settings: Settings,
    n_resamples: int,
    seed: int,
) -> tuple[dict[str, object], float, int]:
    """Execution match between the teacher's queries under two prompts on `ceiling.n`
    held-out questions; (stats, spent, failures)."""
    if kind == "gold":
        return {"skipped": "gold queries are fixed, so the teacher agrees with itself"}, 0.0, 0
    if config.ceiling.n == 0:
        return {"skipped": "ceiling.n is 0"}, 0.0, 0
    spec, para = config.spec, config.paraphrase_spec()
    subset = list(heldout[: config.ceiling.n])
    second = make_teacher(kind, config, para, data, settings)
    spent, failures = 0.0, 0
    with ws.open_store(para) as store:
        done = store.labels(second.model_id, para.prompt_hash)
        todo = [it for it, _ in subset if it.id not in done]
        if todo:
            batch = second.label(todo)
            spent, failures = batch.spent_usd, len(batch.failures)
            require_complete(kind, batch.failures, config, para)
            store.append_labels(batch.labels)
            store.append_failures(batch.failures, second.model_id, para.prompt_hash)
        repeated = store.labels(second.model_id, para.prompt_hash)
    if kind == "claude":
        _record(config, para, ws)
    both = [(it, lab) for it, lab in subset if it.id in repeated]
    if not both:
        raise ConfigError("the paraphrase labels cover none of the ceiling inputs; re-label")
    first = sql_of(spec, both)
    again = sql_of(para, [(it, repeated[it.id]) for it, _ in both])
    stats, _ = score_pair([tables[it.id] for it, _ in both], first, again, n_resamples, seed)
    return {"paraphrase_prompt_hash": para.prompt_hash, **stats}, spent, failures


def _zero_shot(config: EvalConfig, texts: Sequence[str]) -> tuple[list[str], list[str], float]:
    """The base model's completions with no adapter (a fresh LoRA has B = 0, so it is the
    base model exactly), their extracted SQL, and the seconds it took."""
    started = time.perf_counter()
    base = LoraGenerator.build(
        config.spec.student, config.spec.output_names, student_system_prompt(config.spec)
    )
    completions = base.generate(texts)
    del base
    return completions, [extract_sql(c) for c in completions], time.perf_counter() - started


def judge_file(dataset: str, model: str) -> Path:
    return LABELS_DIR / dataset / f"judge-{model}-{rubric_hash()}.jsonl"


def _make_judge(config: EvalConfig, settings: Settings) -> Judge | str:
    """The judge to use, or why it is skipped: recorded verdicts replay for free, a key pays
    only for the items they do not cover, and with neither the judge is skipped (never
    imitated)."""
    path = judge_file(config.dataset.name, config.judge.model)
    recorded = VerdictFileJudge.from_file(path, config.judge.model) if path.exists() else None
    if os.environ.get("ANTHROPIC_API_KEY"):
        from anvil.teacher.transport import AnthropicTransport

        live = ClaudeJudge(
            AnthropicTransport(),
            model=config.judge.model,
            cache=DiskCache(settings.home / "cache" / "judge"),
            workers=settings.teacher_workers,
        )
        return live if recorded is None else RecordedThenLiveJudge(recorded, live)
    if recorded is not None:
        return recorded
    return f"no recorded verdicts at {_shown(path)} and no ANTHROPIC_API_KEY"


def _judge(
    config: EvalConfig,
    settings: Settings,
    heldout: Sequence[tuple[Item, str]],
    references: Sequence[str],
    candidates: Sequence[str],
    matches: Sequence[ExecMatch],
) -> tuple[dict[str, object], float]:
    """The judge's verdicts on a seeded sample of (teacher, student) pairs, and how they line
    up with execution match on the same items; (stats, spent)."""
    if config.judge.n == 0:
        return {"skipped": "judge.n is 0"}, 0.0
    judge = _make_judge(config, settings)
    ids = [it.id for it, _ in heldout]
    chosen = set(judge_sample(ids, config.judge.n, config.judge.seed))
    index = [i for i, it_id in enumerate(ids) if it_id in chosen]
    if isinstance(judge, str):
        return {"skipped": judge, "sample": [ids[i] for i in index]}, 0.0
    items = [JudgeItem(ids[i], heldout[i][0].text, references[i], candidates[i]) for i in index]
    verdicts: list[Verdict] = judge.judge(items)
    spent = spent_usd(verdicts)
    if isinstance(judge, ClaudeJudge | RecordedThenLiveJudge):
        write_verdict_file(
            judge_file(config.dataset.name, config.judge.model),
            verdicts,
            config.judge.model,
            rubric_hash(),
        )
    agreement = judge_vs_metric(verdicts, {ids[i]: matches[i].match for i in index})
    return {
        "model": judge.model_id,
        "rubric_hash": rubric_hash(),
        "vs_execution_match": asdict(agreement),
        "verdicts": [asdict(v) for v in verdicts],
        "spent_usd": spent,
    }, spent


def _export(
    config: EvalConfig, student_dir: Path, ws: Workspace, texts: Sequence[str]
) -> dict[str, object]:
    if config.export.probes == 0:
        return {"skipped": "export.probes is 0"}
    out = ws.root / "exports" / f"{student_dir.name}-merged"
    report = export_generator(
        student_dir, out, list(texts[: config.export.probes]), spec=config.spec.model_dump()
    )
    return {
        "parity": asdict(report.parity),
        "size_mb": report.size_mb,
        "seconds": report.seconds,
        "adapter_parameters": report.adapter_parameters,
    }


# --- the run ----------------------------------------------------------------------------


def workspace(settings: Settings, spec: TaskSpec, data: Sample) -> Workspace:
    return Workspace(settings.home / "eval" / f"{spec.name}-{data.sha256[:12]}")


def _record(config: EvalConfig, spec: TaskSpec, ws: Workspace) -> None:
    with ws.open_store(spec) as store:
        record_labels(
            label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash),
            store.label_records(spec.teacher.model, spec.prompt_hash),
            store.failure_records(spec.teacher.model, spec.prompt_hash),
            (spec.teacher.model, spec.prompt_hash),
        )


def _timed[T](fn: Callable[[], T]) -> tuple[T, float]:
    started = time.perf_counter()
    out = fn()
    return out, time.perf_counter() - started


def run(
    config: EvalConfig,
    data: Sample,
    examples: Mapping[str, SqlExample],
    kind: TeacherKind,
    out_dir: Path,
    settings: Settings,
    n_resamples: int = DEFAULT_RESAMPLES,
    command: str = "",
    provenance: Mapping[str, object] | None = None,
    load_report: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """One full eval; writes `report.json`, `meta.json` and `train-metrics.jsonl`."""
    started = time.perf_counter()
    prov = dict(provenance) if provenance is not None else git_provenance()
    kind = resolve_teacher_kind(kind, config)
    spec = config.spec
    seed = spec.student.seed
    ws = workspace(settings, spec, data)
    inputs = ws.root / "inputs.jsonl"
    write_inputs(inputs, data.items)
    tables = {it_id: ex.table() for it_id, ex in examples.items()}

    teacher = make_teacher(kind, config, spec, data, settings)
    with ws.open_store(spec) as store:
        pool_done = len(store.labelled(teacher.model_id, spec.prompt_hash, Split.POOL))
    labelled = label(
        spec,
        ws,
        teacher,
        inputs,
        max(0, config.sample.pool_labels - pool_done),
        config.sample.heldout_labels,
    )
    spent, failures = labelled.spent_usd, len(labelled.failures)
    require_complete(kind, labelled.failures, config, spec)
    if kind == "claude":
        _record(config, spec, ws)
    with ws.open_store(spec) as store:
        heldout = store.labelled(teacher.model_id, spec.prompt_hash, Split.HELDOUT)
        pool = store.labelled(teacher.model_id, spec.prompt_hash, Split.POOL)
        stats = store.stats(teacher.model_id, spec.prompt_hash)
    if not heldout:
        raise ConfigError(
            f"the sample of {len(data.items)} questions put none in the held-out split; raise"
            " --size (or sample.size in the config)"
        )

    held_tables = [tables[it.id] for it, _ in heldout]
    held_examples = [examples[it.id] for it, _ in heldout]
    texts = [it.text for it, _ in heldout]
    teacher_sql = sql_of(spec, heldout)

    ceiling, c_spent, c_failures = _ceiling(
        kind, config, data, heldout, tables, ws, settings, n_resamples, seed
    )
    spent += c_spent
    failures += c_failures

    trained = train(spec, ws, teacher.model_id)
    student = LoraGenerator.load(trained.path)
    completions, gen_seconds = _timed(lambda: student.generate(texts))
    student_sql = [extract_sql(c) for c in completions]
    agreement, matches = score_pair(held_tables, teacher_sql, student_sql, n_resamples, seed)
    flags = [m.match for m in matches]

    baseline: dict[str, object]
    base_sql: list[str] = []
    if config.baseline.zero_shot:
        base_completions, base_sql, base_seconds = _zero_shot(config, texts)
        base_stats, base_matches = score_pair(
            held_tables, teacher_sql, base_sql, n_resamples, seed
        )
        baseline = {
            "zero_shot": {
                **base_stats,
                "seconds": base_seconds,
                "fenced_completions": sum("```" in c for c in base_completions),
                "student_minus_zero_shot": asdict(
                    paired_difference_ci(
                        flags,
                        [m.match for m in base_matches],
                        n_resamples=n_resamples,
                        seed=seed,
                    )
                ),
            }
        }
    else:
        baseline = {"zero_shot": {"skipped": "baseline.zero_shot is false"}}

    gold_sql = [ex.sql for ex in held_examples]
    gold: dict[str, object] = {
        "student_vs_gold": score_pair(held_tables, gold_sql, student_sql, n_resamples, seed)[0]
    }
    if kind != "gold":
        gold["teacher_vs_gold"] = score_pair(
            held_tables, gold_sql, teacher_sql, n_resamples, seed
        )[0]

    judged, j_spent = _judge(config, settings, heldout, teacher_sql, student_sql, matches)
    spent += j_spent
    export = _export(config, trained.path, ws, texts)

    wall = time.perf_counter() - started
    report: dict[str, object] = {
        "dataset": {
            "name": config.dataset.name,
            "repo": config.dataset.repo,
            "revision": config.dataset.revision,
            "file": config.dataset.filename,
            "license": config.dataset.license,
            "source_rows": data.source_rows,
            "load": dict(load_report or {}),
            "sampled": len(data.items),
            "sample_seed": config.sample.seed,
            "max_columns": config.sample.max_columns,
            "sample_sha256": data.sha256,
            "near_duplicate_heldout": near_duplicates(texts, [it.text for it, _ in pool]),
        },
        "teacher": {
            "kind": kind,
            "model": teacher.model_id,
            "prompt_hash": spec.prompt_hash,
            "effort": spec.teacher.effort if kind != "gold" else None,
            "pool_labels": len(pool),
            "heldout_labels": len(heldout),
            "failures": failures,
            "label_cost_usd": stats.cost_usd,
        },
        "student": {
            "model": spec.student.model,
            "run": trained.run_id,
            "settings": spec.student.model_dump(),
            "train": asdict(trained.result),
            "chat": trained.chat.as_dict() if trained.chat is not None else None,
            "generate_seconds": gen_seconds,
            "generate_ms_per_input": 1000 * gen_seconds / len(texts),
        },
        "agreement": agreement,
        "slices": slices(held_examples, texts, flags),
        "baselines": baseline,
        "ceiling": ceiling,
        "gold": gold,
        "judge": judged,
        "export": export,
        "predictions": {
            "input_id": [it.id for it, _ in heldout],
            "teacher": teacher_sql,
            "student": student_sql,
            "zero_shot": base_sql,
            "execution_match": flags,
            "student_error": [m.candidate.error for m in matches],
        },
        "spent_usd": spent,
        "wall_seconds": wall,
    }
    meta = {
        "command": command,
        "git_sha": prov["git_sha"],
        "git_dirty": prov["git_dirty"],
        "anvil": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "device": "cuda" if _cuda() else "cpu",
        "versions": _versions(),
        "seed": seed,
        "n_resamples": n_resamples,
        "teacher_model": teacher.model_id,
        "student_model": spec.student.model,
        "judge_model": config.judge.model,
        "wall_seconds": round(wall, 1),
        "spent_usd": spent,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    metrics = trained.path / "metrics.jsonl"
    if metrics.exists():
        shutil.copyfile(metrics, out_dir / "train-metrics.jsonl")
    logger.info("eval finished", extra={"dataset": spec.name, "wall_s": round(wall, 1)})
    return report


def _cuda() -> bool:
    import torch

    return bool(torch.cuda.is_available())


def results_dir(dataset: str, teacher_model: str, smoke: bool = False) -> Path:
    return RESULTS_DIR / f"generate-{dataset}-{teacher_model}{SMOKE_SUFFIX if smoke else ''}"


def _override(
    config: EvalConfig,
    student: str | None,
    epochs: int | None,
    size: int | None,
    baseline: bool,
    max_length: int | None = None,
) -> EvalConfig:
    s_update: dict[str, object] = {}
    if student is not None:
        s_update["model"] = student
    if epochs is not None:
        s_update["epochs"] = epochs
    if max_length is not None:
        s_update["max_length"] = max_length
    spec = config.spec.model_copy(
        update={"student": config.spec.student.model_copy(update=s_update)}
    )
    smp = config.sample
    if size is not None:
        ratio = size / smp.size
        smp = smp.model_copy(
            update={
                "size": size,
                "pool_labels": max(1, round(smp.pool_labels * ratio)),
                "heldout_labels": max(1, round(smp.heldout_labels * ratio)),
            }
        )
    update: dict[str, object] = {"spec": spec, "sample": smp}
    if not baseline:
        update["baseline"] = BaselineCfg(zero_shot=False)
    # `model_copy` does not validate, so the result goes through the validators again.
    try:
        return EvalConfig.model_validate(config.model_copy(update=update).model_dump())
    except ValidationError as exc:
        raise ConfigError(f"invalid override: {exc}") from exc


def summary_line(report: Mapping[str, object], out: Path) -> str:
    ag, teacher, dataset = report["agreement"], report["teacher"], report["dataset"]
    assert isinstance(ag, dict)
    assert isinstance(teacher, dict)
    assert isinstance(dataset, dict)
    em = ag["execution_match"]
    return (
        f"{dataset['name']}: {teacher['model']} -> student: execution match"
        f" {em['estimate']:.3f} [{em['low']:.3f}, {em['high']:.3f}], exact SQL"
        f" {ag['exact_match']:.3f} on {ag['n']} held-out, {teacher['pool_labels']} train labels;"
        f" raw output {out}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.generate", description=__doc__)
    parser.add_argument("datasets", nargs="+", help="config names under eval/generate/")
    parser.add_argument("--teacher", choices=["auto", "claude", "replay", "gold"], default="auto")
    parser.add_argument("--student", help="override student.model (e.g. tiny for a smoke run)")
    parser.add_argument("--epochs", type=int, help="override student.epochs")
    parser.add_argument("--size", type=int, help="override sample.size (smoke runs)")
    parser.add_argument(
        "--max-length",
        type=int,
        help="override student.max_length (the tiny student counts bytes, about 4x as many)",
    )
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument(
        "--no-baseline", action="store_true", help="skip the zero-shot base model (smoke runs)"
    )
    parser.add_argument("--out", type=Path, help="results directory (one dataset only)")
    args = parser.parse_args(argv)
    if args.out is not None and len(args.datasets) > 1:
        parser.error("--out takes one dataset; several would overwrite each other")

    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.generate " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    smoke = (
        any(v is not None for v in (args.student, args.epochs, args.size, args.max_length))
        or args.resamples != DEFAULT_RESAMPLES
        or args.no_baseline
    )
    if smoke and args.out is not None and not args.out.name.endswith(SMOKE_SUFFIX):
        parser.error(f"--out with overrides (a smoke run) must end in {SMOKE_SUFFIX}")
    try:
        for name in args.datasets:
            config = _override(
                load_config(CONFIG_DIR / f"{name}.yaml"),
                args.student,
                args.epochs,
                args.size,
                baseline=not args.no_baseline,
                max_length=args.max_length,
            )
            loaded, load_report = load_examples(config.dataset.ref())
            data, examples = sample_examples(
                loaded,
                config.sample.size,
                config.sample.seed,
                max_columns=config.sample.max_columns,
            )
            kind = resolve_teacher_kind(args.teacher, config)
            model = "gold" if kind == "gold" else config.spec.teacher.model
            out = args.out or results_dir(name, model, smoke=smoke)
            report = run(
                config,
                data,
                examples,
                kind,
                out,
                settings,
                args.resamples,
                command,
                provenance,
                load_report=asdict(load_report),
            )
            print(summary_line(report, out))  # noqa: T201 - the final summary
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
