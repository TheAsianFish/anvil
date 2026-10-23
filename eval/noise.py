"""`make eval-noise`: what kind of noise is the local teacher's?

A student trained on a fallible teacher's labels can end up closer to the truth than its
teacher (when the teacher's mistakes look random, training averages them away) or as far from
it (when the mistakes are systematic, the student learns them as if they were the task). This
control trains the frontier's CPU student (MiniLM-L6, `eval/frontier.yaml`) on the same pool
inputs four times per seed and scores each against gold on the same held-out inputs:

- `gold`: the dataset's own labels (the clean reference);
- `teacher`: the local teacher's labels, as recorded by `make eval-classify TEACHER=ollama`;
- `uniform`: gold with exactly as many labels wrong as the teacher got wrong on the pool, each
  moved to a uniformly random other class;
- `pair`: gold with the teacher's own confusions, class by class: for each gold class, the
  wrong labels the teacher gave that class's inputs are dealt to as many randomly chosen
  inputs of the class. The confusion matrix is the teacher's exactly; only which inputs carry
  the errors is random.

If the teacher-trained student scores like `uniform`, the teacher's errors behave as random
noise; like `pair`, as fixed confusions between classes; below both, its errors follow the
inputs themselves (the same kind of input is mislabelled the same way), which no class-level
noise model reproduces. How often each student repeats the teacher's held-out mistakes
(`copied`) separates the three: input-tied errors are the only ones a student can learn to
reproduce on unseen inputs, so they show up as copying well above what `pair` produces.

Each seed draws its own corruption and its own student initialisation and data order; `gold`
and `teacher` vary only the student. Raw output: `eval/results/noise-<dataset>-<teacher>/`
(`report.json` with every run's held-out predictions, `meta.json`). The labels come from
`eval/labels/` (replayed, no Ollama server needed); about ten CPU minutes per student.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from anvil import __version__, log
from anvil.errors import AnvilError, ConfigError
from anvil.evaluate.bootstrap import bootstrap_difference_ci
from anvil.pipeline import Workspace, label, train
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.store import Split
from anvil.students.encoder import EncoderClassifier
from anvil.teacher.base import Item, Label
from anvil.teacher.labelfile import LabelFileTeacher
from eval.classify import (
    CONFIG_DIR,
    RESULTS_DIR,
    EvalConfig,
    _pair_stats,
    _versions,
    file_safe,
    git_provenance,
    label_file,
    load_config,
    require_complete,
    teacher_errors,
    workspace,
)
from eval.data import Sample, load_rows, sample, write_inputs
from eval.frontier import load_frontier_config
from eval.render import SMOKE_SUFFIX
from eval.student import find_student

DATASETS = ("banking77", "ag_news")
SEEDS = (0, 1, 2)
CONDITIONS = ("gold", "teacher", "uniform", "pair")
DEFAULT_RESAMPLES = 1000

logger = log.get_logger("anvil.eval.noise")


def corrupt_uniform(
    gold: Sequence[str], classes: Sequence[str], n_wrong: int, seed: int
) -> list[str]:
    """`gold` with exactly `n_wrong` labels, chosen at random, moved to a uniformly random
    other class."""
    if not 0 <= n_wrong <= len(gold):
        raise ValueError(f"cannot corrupt {n_wrong} of {len(gold)} labels")
    rng = np.random.default_rng(seed)
    out = list(gold)
    for i in rng.choice(len(gold), size=n_wrong, replace=False).tolist():
        others = [c for c in classes if c != gold[i]]
        out[i] = others[int(rng.integers(len(others)))]
    return out


def corrupt_pairs(gold: Sequence[str], teacher: Sequence[str], seed: int) -> list[str]:
    """`gold` with the teacher's confusions moved to random inputs of the same gold class.

    For each gold class, the wrong labels the teacher gave inputs of that class (a multiset)
    are dealt to that many inputs of the class chosen at random, so every (gold, label) cell
    of the confusion matrix matches the teacher's while the inputs carrying the errors do not.
    """
    if len(gold) != len(teacher):
        raise ValueError("gold and teacher labels must have the same length")
    rng = np.random.default_rng(seed)
    by_class: dict[str, list[int]] = defaultdict(list)
    for i, g in enumerate(gold):
        by_class[g].append(i)
    out = list(gold)
    for g in sorted(by_class):
        members = by_class[g]
        wrong = sorted(teacher[i] for i in members if teacher[i] != g)
        if not wrong:
            continue
        chosen = rng.choice(len(members), size=len(wrong), replace=False).tolist()
        for slot, label_ in zip(chosen, rng.permutation(wrong).tolist(), strict=True):
            out[members[slot]] = str(label_)
    return out


def _labels_for(
    condition: str, gold: list[str], teacher: list[str], classes: list[str], seed: int
) -> list[str]:
    if condition == "gold":
        return list(gold)
    if condition == "teacher":
        return list(teacher)
    n_wrong = sum(g != t for g, t in zip(gold, teacher, strict=True))
    if condition == "uniform":
        return corrupt_uniform(gold, classes, n_wrong, seed)
    if condition == "pair":
        return corrupt_pairs(gold, teacher, seed)
    raise ValueError(f"unknown condition {condition!r}")


def _model_id(condition: str, seed: int) -> str:
    """The store key the condition's training labels go under: the clean and the real labels
    are one set for every seed, a synthetic corruption is a set per seed."""
    return f"noise-{condition}" + (f"-seed{seed}" if condition in ("uniform", "pair") else "")


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {"mean": float(np.mean(values)), "min": min(values), "max": max(values)}


@dataclass(frozen=True)
class Pool:
    """The M8 sample's pool and held-out inputs with the teacher's recorded labels and gold."""

    config: EvalConfig
    spec: TaskSpec
    data: Sample
    teacher_model: str
    pool: list[tuple[Item, str]]
    """(input, teacher label) in store order: the order every corruption is drawn in."""
    heldout: list[tuple[Item, str]]

    @property
    def pool_gold(self) -> list[str]:
        return [self.data.gold[it.id] for it, _ in self.pool]

    @property
    def pool_teacher(self) -> list[str]:
        return [lab for _, lab in self.pool]


def load_pool(name: str, settings: Settings, size: int | None = None) -> Pool:
    """The local teacher's labels for dataset `name`, replayed from `eval/labels/` into the
    classify eval's workspace (never labelled live), with the gold labels next to them."""
    config = load_config(CONFIG_DIR / f"{name}.yaml").with_local_teacher()
    n = size or config.sample.size
    data = sample(name, *load_rows(config.dataset.ref()), n=n, seed=config.sample.seed)
    spec = config.spec
    pool_labels = config.sample.pool_labels if size is None else round(0.8 * n)
    heldout_labels = config.sample.heldout_labels if size is None else n
    ws = workspace(settings, spec, data)
    inputs = ws.root / "inputs.jsonl"
    write_inputs(inputs, data.items)
    # Replay only: a gap in the recording is an error here, never labelled live, so the
    # control always trains on exactly the labels `make eval-classify TEACHER=ollama` scored.
    recording = label_file(name, spec.teacher.model, spec.prompt_hash)
    if not recording.exists():
        raise ConfigError(
            f"no recorded {spec.teacher.model} labels for {name} at {recording}; run"
            " `make eval-classify TEACHER=ollama` first"
        )
    recorded = LabelFileTeacher.from_file(recording, spec.teacher.model, spec.prompt_hash)
    with ws.open_store(spec) as store:
        pool_done = len(store.labelled(recorded.model_id, spec.prompt_hash, Split.POOL))
    labelled = label(spec, ws, recorded, inputs, max(0, pool_labels - pool_done), heldout_labels)
    require_complete("replay", labelled.failures, config, spec)
    with ws.open_store(spec) as store:
        pool = store.labelled(recorded.model_id, spec.prompt_hash, Split.POOL)
        heldout = store.labelled(recorded.model_id, spec.prompt_hash, Split.HELDOUT)
    if not pool or not heldout:
        raise ConfigError(f"no recorded {recorded.model_id} labels for {name}")
    return Pool(config, spec, data, recorded.model_id, pool, heldout)


def run_dataset(
    name: str,
    settings: Settings,
    seeds: Sequence[int],
    conditions: Sequence[str],
    n_resamples: int,
    size: int | None = None,
    epochs: int | None = None,
) -> dict[str, object]:
    loaded = load_pool(name, settings, size)
    config, spec, data, pool, heldout = (
        loaded.config,
        loaded.spec,
        loaded.data,
        loaded.pool,
        loaded.heldout,
    )
    pool_gold = loaded.pool_gold
    pool_teacher = loaded.pool_teacher
    h_texts = [it.text for it, _ in heldout]
    h_gold = [data.gold[it.id] for it, _ in heldout]
    h_teacher = [lab for _, lab in heldout]
    classes = spec.label_names
    n_wrong = sum(g != t for g, t in zip(pool_gold, pool_teacher, strict=True))

    student_spec = load_frontier_config().student
    if epochs is not None:
        student_spec = student_spec.model_copy(update={"epochs": epochs, "min_steps": 0})
    noise_ws = Workspace(settings.home / "eval" / f"noise-{spec.name}-{data.sha256[:12]}")
    with noise_ws.open_store(spec) as store:
        store.add_inputs([it for it, _ in pool])

    runs: list[dict[str, object]] = []
    for seed in seeds:
        for condition in conditions:
            model_id = _model_id(condition, seed)
            labels = _labels_for(condition, pool_gold, pool_teacher, classes, seed)
            with noise_ws.open_store(spec) as store:
                stored = store.labels(model_id, spec.prompt_hash)
                if not stored:
                    store.append_labels(
                        [
                            Label(it.id, lab, model_id, spec.prompt_hash)
                            for (it, _), lab in zip(pool, labels, strict=True)
                        ]
                    )
                elif [stored[it.id] for it, _ in pool] != labels:
                    raise ConfigError(
                        f"the store in {noise_ws.root} holds other {model_id} labels than this"
                        " run draws (the pool or the corruption changed); delete it and re-run"
                    )
            run_spec = spec.model_copy(
                update={"student": student_spec.model_copy(update={"seed": seed})}
            )
            path = find_student(noise_ws, run_spec, model_id, len(pool))
            started = time.perf_counter()
            if path is None:
                logger.info(
                    "training a noise-control student",
                    extra={"dataset": name, "condition": condition, "seed": seed},
                )
                path = train(run_spec, noise_ws, model_id).path
            train_seconds = time.perf_counter() - started
            student = EncoderClassifier.load(path)
            probs = student.predict_proba(h_texts)
            predicted = [student.labels[i] for i in probs.argmax(axis=1)]
            runs.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "student_run": path.name,
                    "train_labels_wrong": sum(
                        g != lab for g, lab in zip(pool_gold, labels, strict=True)
                    ),
                    "train_seconds": round(train_seconds, 1),
                    "vs_gold": _pair_stats(h_gold, predicted, n_resamples, seed),
                    "vs_teacher": _pair_stats(h_teacher, predicted, n_resamples, seed),
                    "errors": teacher_errors(h_gold, h_teacher, predicted, n_resamples, seed),
                    "predicted": predicted,
                }
            )
    return {
        "dataset": {
            "name": name,
            "repo": config.dataset.repo,
            "revision": config.dataset.revision,
            "sample_sha256": data.sha256,
        },
        "teacher": {
            "model": loaded.teacher_model,
            "digest": spec.teacher.digest,
            "prompt_hash": spec.prompt_hash,
        },
        "pool": {
            "n": len(pool),
            "teacher_wrong": n_wrong,
            "teacher_error_rate": n_wrong / len(pool),
            "teacher_vs_gold": _pair_stats(pool_gold, pool_teacher, n_resamples, 0),
        },
        "heldout": {
            "n": len(heldout),
            "input_id": [it.id for it, _ in heldout],
            "gold": h_gold,
            "teacher": h_teacher,
            "teacher_vs_gold": _pair_stats(h_gold, h_teacher, n_resamples, 0),
        },
        "student": student_spec.model_dump(),
        "seeds": list(seeds),
        "conditions": list(conditions),
        "runs": runs,
        "summary": summarise(runs, h_gold, n_resamples),
    }


def summarise(
    runs: Sequence[dict[str, object]], gold: Sequence[str], n_resamples: int
) -> dict[str, object]:
    """Per condition, the mean and range over seeds of each score; per seed, the paired
    bootstrap interval of the teacher-trained student's kappa with gold minus each synthetic
    condition's (same held-out items, same seed)."""

    def score(run: dict[str, object], key: str) -> float:
        stats = run[key]
        assert isinstance(stats, dict)
        return float(stats["kappa"])

    def copied(run: dict[str, object]) -> float:
        errors = run["errors"]
        assert isinstance(errors, dict)
        on = errors["on_teacher_errors"]
        return on["copied"] / on["n"] if on["n"] else 0.0

    out: dict[str, object] = {}
    by_condition: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in runs:
        by_condition[str(run["condition"])].append(run)
    for condition, rows in by_condition.items():
        out[condition] = {
            "seeds": len(rows),
            "kappa_vs_gold": _summary([score(r, "vs_gold") for r in rows]),
            "kappa_vs_teacher": _summary([score(r, "vs_teacher") for r in rows]),
            "copied_rate": _summary([copied(r) for r in rows]),
        }
    paired: dict[str, list[dict[str, object]]] = {}
    for other in ("uniform", "pair", "gold"):
        cells = []
        for run in by_condition.get("teacher", []):
            match = [r for r in by_condition.get(other, []) if r["seed"] == run["seed"]]
            if not match:
                continue
            a, b = run["predicted"], match[0]["predicted"]
            assert isinstance(a, list)
            assert isinstance(b, list)
            seed = int(str(run["seed"]))
            diff = bootstrap_difference_ci(gold, a, b, "kappa", n_resamples=n_resamples, seed=seed)
            cells.append({"seed": seed, **asdict(diff)})
        if cells:
            paired[f"teacher_minus_{other}"] = cells
    out["paired_vs_gold"] = paired
    return out


def results_dir(dataset: str, teacher_model: str, smoke: bool) -> Path:
    return RESULTS_DIR / (
        f"noise-{dataset}-{file_safe(teacher_model)}" + (SMOKE_SUFFIX if smoke else "")
    )


def _ints(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.noise", description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--seeds", type=_ints, help="comma-separated (default 0,1,2)")
    parser.add_argument("--size", type=int, help="smaller sample (smoke run)")
    parser.add_argument("--epochs", type=int, help="override the student's epochs (smoke run)")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    log.configure(settings.log_level)
    args_shown = " ".join(argv if argv is not None else sys.argv[1:])
    command = f"python -m eval.noise {args_shown}".rstrip()
    provenance = git_provenance()
    seeds = tuple(args.seeds) if args.seeds else SEEDS
    smoke = (
        args.seeds is not None
        or args.size is not None
        or args.epochs is not None
        or args.resamples != DEFAULT_RESAMPLES
    )
    try:
        for name in args.datasets:
            started = time.perf_counter()
            report = run_dataset(
                name, settings, seeds, CONDITIONS, args.resamples, args.size, args.epochs
            )
            wall = time.perf_counter() - started
            teacher = report["teacher"]
            assert isinstance(teacher, dict)
            out = results_dir(name, str(teacher["model"]), smoke)
            out.mkdir(parents=True, exist_ok=True)
            meta = {
                "command": command,
                **provenance,
                "anvil": __version__,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "cpus": os.cpu_count(),
                "versions": _versions(),
                "seeds": list(seeds),
                "n_resamples": args.resamples,
                "teacher_model": teacher["model"],
                "teacher_digest": teacher["digest"],
                "student_model": load_frontier_config().student.model,
                "wall_seconds": round(wall, 1),
                "spent_usd": 0.0,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
            print(summary_line(report, out))  # noqa: T201 - the final summary
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    return 0


def summary_line(report: dict[str, object], out: Path) -> str:
    summary = report["summary"]
    dataset = report["dataset"]
    assert isinstance(summary, dict)
    assert isinstance(dataset, dict)
    cells = []
    for condition in CONDITIONS:
        entry = summary.get(condition)
        if isinstance(entry, dict):
            cells.append(f"{condition} {entry['kappa_vs_gold']['mean']:.3f}")
    return (
        f"{dataset['name']}: student kappa vs gold by training labels: {', '.join(cells)}; {out}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
