"""`make eval-frontier`: held-out kappa against teacher labels, per selection strategy.

For each dataset in `eval/frontier.yaml`, every implemented strategy runs one active-learning
loop (`anvil.loop`) per seed over the same label budgets: round 0 is a random draw shared by
every strategy of that seed, later rounds pick by the strategy, and each round retrains a
fresh student and scores it against the teacher on the same held-out inputs. Strategies that
are declared but not written yet are recorded as such, so the table shows what is missing.

Datasets, samples, prompts and teachers come from `eval/classify/<dataset>.yaml`, so this eval
and `make eval-classify` share recorded labels: a label bought once serves both, and every
strategy and seed of the frontier reuses it. The teacher kinds are those of `eval.classify`
(`auto`, `claude`, `replay`, `gold`).

Raw output: `eval/results/frontier-<dataset>-<teacher>/` with `frontier.jsonl` (one row per
loop round), `report.json` (per strategy and budget: kappa mean and range over seeds, label
cost, classes covered; labels needed to reach each target kappa) and `meta.json`. A run with
overrides (`--seeds`, `--budgets`, `--strategies`, `--size`, `--student`, `--epochs`,
`--resamples`) is a smoke run and writes to `...-smoke/`, which the renderer ignores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from anvil import __version__, log
from anvil.errors import AnvilError, ConfigError
from anvil.loop import LoopConfig, run_loop
from anvil.pipeline import Workspace
from anvil.select import STRATEGIES, implemented
from anvil.settings import Settings
from anvil.spec import StudentSpec, TaskSpec
from anvil.store import Split
from anvil.teacher import Teacher
from anvil.teacher.base import Item, LabelBatch
from anvil.teacher.labelfile import GOLD, NOT_RECORDED
from eval.classify import (
    CONFIG_DIR,
    RESULTS_DIR,
    EvalConfig,
    TeacherKind,
    _shown,
    _versions,
    git_provenance,
    label_file,
    load_config,
    make_teacher,
    record_labels,
)
from eval.data import Sample, load_rows, sample, write_inputs
from eval.render import SMOKE_SUFFIX

ROOT = Path(__file__).resolve().parent.parent
FRONTIER_CONFIG = ROOT / "eval" / "frontier.yaml"
DEFAULT_RESAMPLES = 1000

logger = log.get_logger("anvil.eval.frontier")


class FrontierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    datasets: list[str]
    strategies: list[str]
    seeds: list[int] = Field(min_length=1)
    budgets: list[int] = Field(min_length=1)
    targets: list[float]
    heldout_labels: int = Field(ge=1)
    student: StudentSpec
    n_resamples: int = DEFAULT_RESAMPLES


def load_frontier_config(path: Path = FRONTIER_CONFIG) -> FrontierConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"frontier config {path} not found") from exc
    config = FrontierConfig.model_validate(raw)
    unknown = sorted(set(config.strategies) - set(STRATEGIES))
    if unknown:
        raise ConfigError(f"{path} names unknown strategies {unknown}; known: {STRATEGIES}")
    return config


def resolve_kind(kind: TeacherKind, config: EvalConfig) -> TeacherKind:
    """`auto`: replay the recorded labels when they exist, else Claude when a key is set."""
    if kind != "auto":
        return kind
    spec = config.spec
    path = label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash)
    if path.exists():
        return "replay"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    raise ConfigError(
        f"no recorded labels at {_shown(path)} and no ANTHROPIC_API_KEY; set a key to label"
        " with Claude, or pass --teacher gold to run offline with the dataset's own labels"
    )


class CompleteRecording:
    """Wraps a replayed teacher so an input missing from the recording stops the run: a loop
    would otherwise train on fewer labels than its budget and report it as a result."""

    def __init__(self, inner: Teacher, path: Path) -> None:
        self._inner = inner
        self._path = path
        self.model_id = inner.model_id

    def label(self, items: Sequence[Item]) -> LabelBatch:
        batch = self._inner.label(items)
        missing = [f for f in batch.failures if f.reason == NOT_RECORDED]
        if missing:
            raise ConfigError(
                f"{len(missing)} inputs have no recorded label in {_shown(self._path)}; the"
                " recording is incomplete, so label them with --teacher claude (only the"
                " missing inputs are paid for) or run with --teacher gold"
            )
        return batch


def summarise(
    rows: Sequence[dict[str, object]], config: FrontierConfig
) -> dict[str, dict[str, object]]:
    """Per strategy: kappa over seeds at each budget, and the labels each target needs.

    Two readings of "labels to reach kappa t": the smallest budget whose mean kappa over the
    seeds reaches t, and the stricter smallest budget at which every seed's 95% interval lies
    above t. Neither interpolates between budgets; `None` means not reached within them.
    """
    out: dict[str, dict[str, object]] = {}
    for strategy in config.strategies:
        mine = [r for r in rows if r["strategy"] == strategy]
        if not mine:
            continue
        points = []
        for budget in config.budgets:
            at = [r for r in mine if r["budget"] == budget]
            if not at:
                continue
            kappas = [float(str(r["kappa"])) for r in at]
            points.append(
                {
                    "budget": budget,
                    "seeds": len(at),
                    "labels_mean": _mean([float(str(r["labels"])) for r in at]),
                    "kappa_mean": _mean(kappas),
                    "kappa_min": min(kappas),
                    "kappa_max": max(kappas),
                    "kappa_low_min": min(float(str(r["kappa_low"])) for r in at),
                    "label_cost_usd_mean": _mean([float(str(r["label_cost_usd"])) for r in at]),
                    "classes_covered_mean": _mean([float(str(r["classes_covered"])) for r in at]),
                    "train_seconds_mean": _mean([float(str(r["train_seconds"])) for r in at]),
                }
            )
        complete = [p for p in points if p["seeds"] == len(config.seeds)]
        to_target: dict[str, dict[str, int | None]] = {}
        for t in config.targets:
            to_target[f"{t:g}"] = {
                "mean_kappa": next(
                    (int(str(p["budget"])) for p in complete if float(str(p["kappa_mean"])) >= t),
                    None,
                ),
                "every_seed_ci": next(
                    (
                        int(str(p["budget"]))
                        for p in complete
                        if float(str(p["kappa_low_min"])) >= t
                    ),
                    None,
                ),
            }
        out[strategy] = {"points": points, "labels_to_target": to_target}
    return out


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def loop_id(loop: LoopConfig, spec: TaskSpec, teacher_model: str) -> str:
    """`<strategy>-seed<seed>-<digest of the loop's settings>`. A dataset's workspace is shared
    by every run on the same sample, so a smoke run with other budgets, another student or
    another teacher must start its own loops rather than collide with the full run's."""
    key = {
        **loop.to_json(),
        "teacher_model": teacher_model,
        "prompt_hash": spec.prompt_hash,
        "student": spec.student.model_dump(),
        "splits": spec.splits.model_dump(),
    }
    digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:10]
    return f"{loop.strategy}-seed{loop.seed}-{digest}"


def run_dataset(
    name: str,
    frontier: FrontierConfig,
    data: Sample,
    config: EvalConfig,
    kind: TeacherKind,
    out_dir: Path,
    settings: Settings,
    command: str = "",
    provenance: dict[str, object] | None = None,
) -> dict[str, object]:
    """Every loop of one dataset; writes `frontier.jsonl`, `report.json` and `meta.json`."""
    started = time.perf_counter()
    prov = provenance if provenance is not None else git_provenance()
    kind = resolve_kind(kind, config)
    spec: TaskSpec = config.spec.model_copy(update={"student": frontier.student})
    teacher = make_teacher(kind, config, spec, data, settings)
    path = label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash)
    if kind in ("replay", "gold"):
        teacher = CompleteRecording(teacher, path)
    ws = Workspace(settings.home / "eval" / f"frontier-{name}-{data.sha256[:12]}")
    inputs = ws.root / "inputs.jsonl"
    write_inputs(inputs, data.items)

    rows: list[dict[str, object]] = []
    resumed_rows = 0
    status: dict[str, str] = {}
    for strategy in frontier.strategies:
        if not implemented(strategy):
            status[strategy] = "not implemented"
            logger.warning("strategy not implemented; skipped", extra={"strategy": strategy})
            continue
        status[strategy] = "measured"
        for seed in frontier.seeds:
            loop = LoopConfig(
                strategy=strategy,
                budgets=tuple(frontier.budgets),
                seed=seed,
                heldout_max=frontier.heldout_labels,
                n_resamples=frontier.n_resamples,
            )
            result = run_loop(
                spec,
                ws,
                teacher,
                loop,
                loop_id=loop_id(loop, spec, teacher.model_id),
                inputs=inputs,
            )
            rows += result.rows
            resumed_rows += result.resumed_rows
            logger.info(
                "loop finished",
                extra={"dataset": name, "strategy": strategy, "seed": seed, "rounds": len(rows)},
            )

    with ws.open_store(spec) as store:
        if kind == "claude":
            key = (spec.teacher.model, spec.prompt_hash)
            record_labels(path, store.label_records(*key), store.failure_records(*key), key)
        stats = store.stats(teacher.model_id, spec.prompt_hash)
        pool_size = len(store.inputs(Split.POOL))
        heldout_size = len(store.inputs(Split.HELDOUT))
    spent = sum(float(str(r["spent_usd"])) for r in rows)
    wall = time.perf_counter() - started
    # Wall is this invocation only; the rows' own timings cover a run resumed across several.
    compute = sum(float(str(r["train_seconds"])) + float(str(r["select_seconds"])) for r in rows)
    report: dict[str, object] = {
        "dataset": {
            "name": config.dataset.name,
            "repo": config.dataset.repo,
            "revision": config.dataset.revision,
            "license": config.dataset.license,
            "sampled": len(data.items),
            "sample_seed": config.sample.seed,
            "sample_sha256": data.sha256,
            "pool": pool_size,
            "heldout_inputs": heldout_size,
        },
        "teacher": {
            "kind": kind,
            "model": teacher.model_id,
            "prompt_hash": spec.prompt_hash,
            "labels_in_store": stats.labels,
            "label_cost_usd": stats.cost_usd,
        },
        "student": frontier.student.model_dump(),
        "settings": {
            "strategies": frontier.strategies,
            "seeds": frontier.seeds,
            "budgets": frontier.budgets,
            "targets": frontier.targets,
            "heldout_labels": frontier.heldout_labels,
            "n_resamples": frontier.n_resamples,
        },
        "status": status,
        "strategies": summarise(rows, frontier),
        "spent_usd": spent,
        "wall_seconds": wall,
        "compute_seconds": compute,
        "resumed_rows": resumed_rows,
    }
    meta = {
        "command": command,
        "git_sha": prov["git_sha"],
        "git_dirty": prov["git_dirty"],
        "anvil": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "versions": _versions(),
        "seeds": frontier.seeds,
        "n_resamples": frontier.n_resamples,
        "teacher_model": teacher.model_id,
        "student_model": frontier.student.model,
        "wall_seconds": round(wall, 1),
        "compute_seconds": round(compute, 1),
        "resumed_rows": resumed_rows,
        "spent_usd": spent,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows, key=lambda r: (str(r["strategy"]), int(str(r["seed"])), int(str(r["round"])))
    )
    (out_dir / "frontier.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in ordered), encoding="utf-8"
    )
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return report


def results_dir(dataset: str, teacher_model: str, smoke: bool = False) -> Path:
    return RESULTS_DIR / f"frontier-{dataset}-{teacher_model}{SMOKE_SUFFIX if smoke else ''}"


def _ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _strs(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.frontier", description=__doc__)
    parser.add_argument("datasets", nargs="*", help="default: every dataset in frontier.yaml")
    parser.add_argument("--teacher", choices=["auto", "claude", "replay", "gold"], default="auto")
    parser.add_argument("--strategies", type=_strs, help="smoke: a subset of strategies")
    parser.add_argument("--seeds", type=_ints, help="smoke: other seeds")
    parser.add_argument("--budgets", type=_ints, help="smoke: other label budgets")
    parser.add_argument("--student", help="smoke: another student model (e.g. tiny)")
    parser.add_argument("--epochs", type=int, help="smoke: another epoch count")
    parser.add_argument(
        "--size",
        type=int,
        help="smoke: a smaller sample (pair it with --budgets that fit the smaller pool)",
    )
    parser.add_argument("--resamples", type=int, help="smoke: other bootstrap resamples")
    parser.add_argument("--out", type=Path, help="results directory (one dataset; smoke only)")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.frontier " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    overrides = (
        args.strategies,
        args.seeds,
        args.budgets,
        args.student,
        args.epochs,
        args.size,
        args.resamples,
    )
    smoke = any(v is not None for v in overrides)
    if args.out is not None and (not smoke or len(args.datasets) != 1):
        parser.error("--out is for a smoke run of one dataset")
    try:
        frontier = load_frontier_config()
        update: dict[str, object] = {}
        for key in ("strategies", "seeds", "budgets"):
            if getattr(args, key) is not None:
                update[key] = getattr(args, key)
        if args.resamples is not None:
            update["n_resamples"] = args.resamples
        student: dict[str, object] = {}
        if args.student is not None:
            student["model"] = args.student
        if args.epochs is not None:
            student["epochs"] = args.epochs
        if student:
            update["student"] = frontier.student.model_copy(update=student)
        frontier = FrontierConfig.model_validate({**frontier.model_dump(), **update})
        for name in args.datasets or frontier.datasets:
            config = load_config(CONFIG_DIR / f"{name}.yaml")
            if args.size is not None:
                ratio = args.size / config.sample.size
                smp = config.sample.model_copy(
                    update={
                        "size": args.size,
                        "pool_labels": max(1, round(config.sample.pool_labels * ratio)),
                        "heldout_labels": max(1, round(config.sample.heldout_labels * ratio)),
                    }
                )
                config = config.model_copy(update={"sample": smp})
            data = sample(
                name,
                *load_rows(config.dataset.ref()),
                n=config.sample.size,
                seed=config.sample.seed,
            )
            kind = resolve_kind(args.teacher, config)
            model = GOLD if kind == "gold" else config.spec.teacher.model
            out = args.out or results_dir(name, model, smoke=smoke)
            report = run_dataset(
                name, frontier, data, config, kind, out, settings, command, provenance
            )
            print(summary_line(report, out))  # noqa: T201 - the final summary
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    return 0


def summary_line(report: dict[str, object], out: Path) -> str:
    dataset, strategies = report["dataset"], report["strategies"]
    assert isinstance(dataset, dict)
    assert isinstance(strategies, dict)
    parts = []
    for name, s in strategies.items():
        last = s["points"][-1]
        parts.append(f"{name} {last['kappa_mean']:.3f} at {last['budget']}")
    return f"{dataset['name']}: {', '.join(parts) or 'no strategy ran'}; raw output {out}"


if __name__ == "__main__":
    raise SystemExit(main())
