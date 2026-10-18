"""`make eval-select`: the acquisition strategies on synthetic data with a cheap student.

The text frontier (a fine-tuned encoder per point, real teacher labels) is expensive; this
study isolates the selection mechanism first. A logistic regression on Gaussian blobs stands in
for the student and the true labels stand in for the teacher, so a full run takes seconds, costs
nothing and runs offline.

Per setting and seed: label `initial` pool points chosen at random (the same points for every
strategy, so strategies differ only from the first selected batch on), then repeatedly train a
fresh student on the labelled points, score the unlabelled pool (`predict_proba`, the raw
features as embeddings), ask `anvil.select.select_batch` for the next `batch` points and label
them, up to `max_labels`. Test accuracy is recorded at every label count.

Settings:
- `blobs`: every pool point is distinct.
- `blobs-duplicated`: `pool_size / copies` distinct points, each copied `copies` times with tiny
  jitter, so a batch picked on uncertainty alone can be several copies of one point.

Raw output goes to `eval/results/select-synthetic/`: `report.json` (settings, data seed,
full-pool accuracy, per-strategy curves for every seed, summaries and labels to target) and
`meta.json` (command, git SHA, seeds, wall time, versions). Strategies that are declared but not
written yet are recorded as `{"status": "not implemented"}`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from anvil import __version__, log
from anvil.select import STRATEGIES, Pool, implemented, select_batch
from anvil.settings import Settings

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results" / "select-synthetic"
SETTINGS = ("blobs", "blobs-duplicated")

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
# Any: the report is arbitrary JSON (the same alias as eval.render.Report).
Report = dict[str, Any]

logger = log.get_logger("eval.select_study")


@dataclass(frozen=True)
class StudyConfig:
    """Every knob of the study; the defaults are the run RESULTS.md reports."""

    seeds: int = 20
    data_seed: int = 0
    classes: int = 10
    features: int = 12
    cluster_std: float = 2.3
    center_box: float = 5.0
    pool_size: int = 2000
    test_size: int = 1000
    copies: int = 5
    """`blobs-duplicated`: each distinct pool point appears this many times."""
    jitter: float = 1e-3
    """Standard deviation of the noise added to each copy."""
    initial: int = 20
    batch: int = 10
    max_labels: int = 400
    target_accuracy: float = 0.95
    target_relative: float = 0.95
    """The second target: this fraction of the full-pool accuracy."""
    strategies: tuple[str, ...] = field(default=STRATEGIES)


@dataclass(frozen=True)
class Data:
    pool_x: FloatArray
    pool_y: IntArray
    test_x: FloatArray
    test_y: IntArray
    distinct: int
    """How many of the pool points are not copies of another."""


def make_data(config: StudyConfig) -> dict[str, Data]:
    """Both settings from one draw: the same test set, and the duplicated pool built from the
    first `pool_size / copies` points of the distinct pool."""
    x, y = make_blobs(
        n_samples=config.pool_size + config.test_size,
        n_features=config.features,
        centers=config.classes,
        cluster_std=config.cluster_std,
        center_box=(-config.center_box, config.center_box),
        random_state=config.data_seed,
    )
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    pool_x, pool_y = x[: config.pool_size], y[: config.pool_size]
    test_x, test_y = x[config.pool_size :], y[config.pool_size :]
    distinct = config.pool_size // config.copies
    rng = np.random.default_rng([config.data_seed, 1])
    dup_x = np.repeat(pool_x[:distinct], config.copies, axis=0)
    dup_x = dup_x + rng.normal(scale=config.jitter, size=dup_x.shape)
    dup_y = np.repeat(pool_y[:distinct], config.copies)
    return {
        "blobs": Data(pool_x, pool_y, test_x, test_y, config.pool_size),
        "blobs-duplicated": Data(dup_x, dup_y, test_x, test_y, distinct),
    }


class Student:
    """Logistic regression with probabilities over every class, seen in training or not."""

    def __init__(self, classes: int) -> None:
        self.classes = classes
        self.model: LogisticRegression | None = None
        self.only: int | None = None

    def fit(self, x: FloatArray, y: IntArray) -> Student:
        seen = np.unique(y)
        if len(seen) == 1:
            self.model, self.only = None, int(seen[0])
            return self
        self.model = LogisticRegression(max_iter=1000).fit(x, y)  # lbfgs, multinomial
        self.only = None
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        probs = np.zeros((len(x), self.classes))
        if self.model is None:
            probs[:, self.only] = 1.0
            return probs
        columns = np.asarray(self.model.classes_, dtype=np.int64)
        probs[:, columns] = np.asarray(self.model.predict_proba(x), dtype=np.float64)
        return probs

    def accuracy(self, x: FloatArray, y: IntArray) -> float:
        return float((self.predict_proba(x).argmax(axis=1) == y).mean())


def _seed(*parts: int) -> int:
    return int(np.random.SeedSequence(list(parts)).generate_state(1)[0])


def label_counts(config: StudyConfig) -> list[int]:
    return list(range(config.initial, config.max_labels + 1, config.batch))


def initial_set(config: StudyConfig, pool_size: int, seed: int) -> list[int]:
    """The cold start of one seed, shared by every strategy."""
    rng = np.random.default_rng([config.data_seed, seed, 0])
    return [int(i) for i in rng.choice(pool_size, size=config.initial, replace=False)]


def curve(config: StudyConfig, data: Data, strategy: str, seed: int) -> list[float]:
    """Test accuracy at each of `label_counts(config)` for one strategy and seed."""
    labelled = initial_set(config, len(data.pool_y), seed)
    unlabelled = np.setdiff1d(np.arange(len(data.pool_y)), labelled)
    accuracies = []
    for round_, count in enumerate(label_counts(config)):
        assert len(labelled) == count
        student = Student(config.classes).fit(data.pool_x[labelled], data.pool_y[labelled])
        accuracies.append(round(student.accuracy(data.test_x, data.test_y), 4))
        if count == config.max_labels:
            break
        pool = Pool(
            probs=student.predict_proba(data.pool_x[unlabelled]),
            embeddings=data.pool_x[unlabelled],
        )
        k = min(config.batch, config.max_labels - count)
        picked = unlabelled[select_batch(pool, k, strategy, _seed(seed, round_))]
        labelled += [int(i) for i in picked]
        unlabelled = np.setdiff1d(unlabelled, picked)
    return accuracies


def labels_to(counts: list[int], accuracies: list[float], target: float) -> int | None:
    """The smallest label count whose test accuracy reaches `target`, or None."""
    return next((c for c, a in zip(counts, accuracies, strict=True) if a >= target), None)


def _spread(values: list[int]) -> dict[str, float | int | None]:
    return {
        "mean": round(float(np.mean(values)), 1) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
    }


def summarise(
    counts: list[int], curves: dict[str, list[float]], targets: dict[str, float]
) -> dict[str, object]:
    matrix = np.array(list(curves.values()))
    to_target: dict[str, object] = {}
    for name, target in targets.items():
        per_seed = {s: labels_to(counts, c, target) for s, c in curves.items()}
        reached = [v for v in per_seed.values() if v is not None]
        to_target[name] = {
            "target": round(target, 4),
            "per_seed": per_seed,
            **_spread(reached),
            "never": len(per_seed) - len(reached),
        }
    return {
        "status": "ok",
        "label_counts": counts,
        "accuracy": {
            "mean": [round(float(v), 4) for v in matrix.mean(axis=0)],
            "min": [round(float(v), 4) for v in matrix.min(axis=0)],
            "max": [round(float(v), 4) for v in matrix.max(axis=0)],
        },
        "labels_to_target": to_target,
        "curves": curves,
    }


def run_study(config: StudyConfig) -> Report:
    """The whole study as one JSON-ready report. BLAS runs single-threaded: the matrices are
    tiny, and a thread pool per fit made each fit several times slower on a 4-CPU runner."""
    with threadpool_limits(limits=1):
        return _run_study(config)


def _run_study(config: StudyConfig) -> Report:
    started = time.perf_counter()
    counts = label_counts(config)
    settings: dict[str, object] = {}
    for name, data in make_data(config).items():
        full = Student(config.classes).fit(data.pool_x, data.pool_y)
        full_accuracy = round(full.accuracy(data.test_x, data.test_y), 4)
        targets = {
            "accuracy": config.target_accuracy,
            "relative": config.target_relative * full_accuracy,
        }
        strategies: dict[str, object] = {}
        for strategy in config.strategies:
            if not implemented(strategy):
                strategies[strategy] = {"status": "not implemented"}
                logger.info("strategy skipped", extra={"setting": name, "strategy": strategy})
                continue
            t0 = time.perf_counter()
            curves = {str(s): curve(config, data, strategy, s) for s in range(config.seeds)}
            strategies[strategy] = summarise(counts, curves, targets)
            logger.info(
                "strategy done",
                extra={
                    "setting": name,
                    "strategy": strategy,
                    "seconds": round(time.perf_counter() - t0, 1),
                },
            )
        settings[name] = {
            "pool_size": len(data.pool_y),
            "distinct_pool_points": data.distinct,
            "test_size": len(data.test_y),
            "full_pool_accuracy": full_accuracy,
            "targets": {k: round(v, 4) for k, v in targets.items()},
            "strategies": strategies,
        }
    return {
        "study": "select-synthetic",
        "config": asdict(config),
        "data": {
            "generator": "sklearn.datasets.make_blobs",
            "data_seed": config.data_seed,
            "classes": config.classes,
            "features": config.features,
            "cluster_std": config.cluster_std,
            "center_box": [-config.center_box, config.center_box],
        },
        "student": "sklearn.linear_model.LogisticRegression(max_iter=1000), lbfgs, multinomial",
        "settings": settings,
        "spent_usd": 0.0,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def git_provenance() -> dict[str, object]:
    """The commit and whether the tree differs from it (same rule as `eval.classify`)."""
    return {"git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}


def _versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for pkg in ("numpy", "scikit-learn"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def write(report: Report, config: StudyConfig, out_dir: Path, command: str, wall: float) -> None:
    provenance = git_provenance()
    meta = {
        "command": command,
        "git_sha": provenance["git_sha"],
        "git_dirty": provenance["git_dirty"],
        "anvil": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "versions": _versions(),
        "seeds": list(range(config.seeds)),
        "data_seed": config.data_seed,
        "wall_seconds": round(wall, 1),
        "spent_usd": 0.0,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def summary_line(report: Report, out_dir: Path) -> str:
    parts = []
    for name, setting in report["settings"].items():
        random = setting["strategies"].get("random", {})
        reach = random.get("labels_to_target", {}).get("accuracy", {}).get("mean")
        parts.append(
            f"{name}: full pool {setting['full_pool_accuracy']:.3f},"
            f" random reaches 0.95 at {reach} labels (mean)"
        )
    return f"select study -> {out_dir}: " + "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.select_study", description=__doc__)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR, help="results directory")
    args = parser.parse_args(argv)
    log.configure(Settings.from_env().log_level)
    command = "python -m eval.select_study " + " ".join(argv if argv is not None else sys.argv[1:])
    config = StudyConfig()
    started = time.perf_counter()
    report = run_study(config)
    write(report, config, args.out, command.strip(), time.perf_counter() - started)
    print(summary_line(report, args.out))  # noqa: T201 - the final summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
