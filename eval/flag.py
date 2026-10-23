"""`make eval-flag`: can the student flag its teacher's mistakes?

M8 left a fallible teacher whose labels a student copies faithfully, and ended with "telling
consistency from correctness needs gold labels or a better teacher". This tests a third option,
the student itself, with the self-confidence score of confident learning (Northcutt et al.,
2021): train the student out of fold on the teacher's labels, then flag a training label when
the student, never having seen it, gives it a low probability. That works when wrong labels
are scattered noise the rest of the data argues against; the question is whether it still works
when the teacher's errors follow the inputs, so that the other folds teach the same mistake.

For each M8 dataset, the frontier's MiniLM student (`eval/frontier.yaml`) is trained 3-fold out
of fold on the 2,000 pool labels three ways, the `eval/noise.py` conditions with their seed-0
corruptions:

- `teacher`: the local teacher's recorded labels (`eval/labels/`, replayed);
- `uniform`: gold with as many labels wrong as the teacher got, each to a random other class;
- `pair`: gold with the teacher's own confusions dealt to random inputs of each class.

The folds are one random split of the pool, shared by every condition, so the conditions differ
only in their labels. Each condition reports, against the labels that really are wrong (given
label != gold): the AUROC of the score, the precision and recall of flagging the lowest-scoring
20%, and the largest recall at precision 0.7 or better. Then the student is retrained (seed 0)
on the 80% that was not flagged and scored against gold on the held-out split, paired with the
committed seed-0 `eval-noise` student trained on the same labels unfiltered. Two more trainings
make that comparison readable for the real teacher: the same student on a random 80% (is any
change the filtering or just less data?) and the unfiltered seed-0 student again (does the
committed reference reproduce on this machine?).

Raw output: `eval/results/flag-<dataset>-<teacher>/` (`report.json`, `meta.json`, and
`oof.npz` with every out-of-fold probability, so `--oof` re-runs the flag analysis without a
model). The labels come from `eval/labels/` (no Ollama server); about six CPU minutes per
student, 14 students per dataset. Every trained student's predictions are cached under
`$ANVIL_HOME/eval/flag-*`, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt

from anvil import __version__, log
from anvil.errors import AnvilError, ConfigError
from anvil.evaluate.bootstrap import bootstrap_difference_ci, resample_indices
from anvil.pipeline import fit_student
from anvil.settings import Settings
from anvil.spec import StudentSpec
from anvil.teacher.base import Item
from eval.classify import RESULTS_DIR, _pair_stats, _versions, file_safe, git_provenance
from eval.frontier import load_frontier_config
from eval.noise import Pool, _labels_for, load_pool
from eval.noise import results_dir as noise_results_dir
from eval.render import SMOKE_SUFFIX
from eval.shift import auroc

DATASETS = ("banking77", "ag_news")
CONDITIONS = ("teacher", "uniform", "pair")
FOLDS = 3
FLAG_FRACTION = 0.2
TARGET_PRECISION = 0.7
SEED = 0
"""The fold split, the corruptions (`eval/noise.py` seed 0) and every student's seed."""
DEFAULT_RESAMPLES = 1000

logger = log.get_logger("anvil.eval.flag")

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


class Predictor(Protocol):
    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float32]: ...


Fit = Callable[[Sequence[str], Sequence[str]], Predictor]
"""(training texts, their labels) -> a student whose probability columns follow the class
order the caller passes to `out_of_fold_proba`."""


# --- flagging ---------------------------------------------------------------------------------


def fold_ids(n: int, folds: int, seed: int) -> IntArray:
    """Fold of each of `n` items: a random permutation dealt round-robin, so the folds differ
    in size by at most one and do not depend on the labels."""
    if not 2 <= folds <= n:
        raise ValueError(f"need 2 <= folds <= n, got {folds} folds for {n} items")
    out = np.empty(n, dtype=np.int64)
    out[np.random.default_rng(seed).permutation(n)] = np.arange(n) % folds
    return out


def out_of_fold_proba(
    texts: Sequence[str], labels: Sequence[str], folds: IntArray, n_classes: int, fit: Fit
) -> FloatArray:
    """Each item's class probabilities from a student trained on the other folds only."""
    if not len(texts) == len(labels) == len(folds):
        raise ValueError("texts, labels and folds must have the same length")
    out = np.full((len(texts), n_classes), np.nan)
    for fold in np.unique(folds).tolist():
        held = folds == fold
        student = fit(
            [t for t, h in zip(texts, held, strict=True) if not h],
            [lab for lab, h in zip(labels, held, strict=True) if not h],
        )
        proba = np.asarray(
            student.predict_proba([t for t, h in zip(texts, held, strict=True) if h])
        )
        if proba.shape != (int(held.sum()), n_classes):
            raise ValueError(
                f"the student returned probabilities of shape {proba.shape} for"
                f" {int(held.sum())} items and {n_classes} classes"
            )
        out[held] = proba
    return out


def self_confidence(proba: FloatArray, given: IntArray) -> FloatArray:
    """The out-of-fold probability of each item's given label: low means suspect."""
    out: FloatArray = proba[np.arange(len(given)), given].astype(np.float64)
    return out


def flag_lowest(score: FloatArray, fraction: float) -> BoolArray:
    """The `round(fraction * n)` items with the lowest score (ties broken by position)."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be in [0, 1], got {fraction}")
    flagged = np.zeros(len(score), dtype=bool)
    flagged[np.argsort(score, kind="stable")[: round(fraction * len(score))]] = True
    return flagged


def precision_recall(flagged: BoolArray, wrong: BoolArray) -> dict[str, float | int]:
    hits = int((flagged & wrong).sum())
    n_flagged, n_wrong = int(flagged.sum()), int(wrong.sum())
    return {
        "flagged": n_flagged,
        "hits": hits,
        "precision": hits / n_flagged if n_flagged else float("nan"),
        "recall": hits / n_wrong if n_wrong else float("nan"),
    }


def recall_at_precision(score: FloatArray, wrong: BoolArray, precision: float) -> dict[str, float]:
    """The largest recall of a flag "score <= cut" whose precision is at least `precision`,
    over every cut between distinct scores (a cut never splits tied items). Recall 0 (and
    no cut) when no cut reaches the precision."""
    order = np.argsort(score, kind="stable")
    s, w = score[order], wrong[order]
    hits = np.cumsum(w)
    flagged = np.arange(1, len(s) + 1)
    ends = np.r_[s[1:] != s[:-1], True]  # the last item of each run of tied scores
    ok = ends & (hits >= precision * flagged - 1e-12)
    n_wrong = int(wrong.sum())
    if not ok.any() or n_wrong == 0:
        return {
            "recall": 0.0,
            "precision": float("nan"),
            "flagged_share": 0.0,
            "cut": float("nan"),
        }
    best = int(np.flatnonzero(ok)[np.argmax(hits[ok])])
    return {
        "recall": float(hits[best] / n_wrong),
        "precision": float(hits[best] / flagged[best]),
        "flagged_share": float(flagged[best] / len(s)),
        "cut": float(s[best]),
    }


def auroc_ci(
    suspicion: FloatArray, wrong: BoolArray, n_resamples: int, seed: int
) -> dict[str, float]:
    """AUROC of `suspicion` for "this label is wrong" with a 95% percentile interval over
    item resamples (a resample with only one kind of item is skipped)."""
    estimate = auroc(suspicion[wrong], suspicion[~wrong])
    reps = []
    for idx in resample_indices(len(wrong), n_resamples, seed):
        w = wrong[idx]
        if w.any() and not w.all():
            reps.append(auroc(suspicion[idx][w], suspicion[idx][~w]))
    low, high = np.percentile(reps, [2.5, 97.5]) if reps else (np.nan, np.nan)
    return {"auroc": estimate, "low": float(low), "high": float(high)}


def analyse_flags(
    proba: FloatArray,
    given: IntArray,
    gold: IntArray,
    fraction: float = FLAG_FRACTION,
    precision: float = TARGET_PRECISION,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = SEED,
) -> dict[str, object]:
    """How well the out-of-fold self-confidence finds the given labels that differ from gold."""
    wrong = given != gold
    score = self_confidence(proba, given)
    flagged = flag_lowest(score, fraction)
    argmax = proba.argmax(axis=1)
    caught = flagged & wrong
    out: dict[str, object] = {
        "n": len(given),
        "wrong": int(wrong.sum()),
        "error_rate": float(wrong.mean()),
        "out_of_fold_accuracy_vs_given": float((argmax == given).mean()),
        "out_of_fold_accuracy_vs_gold": float((argmax == gold).mean()),
        "auroc": auroc_ci(-score, wrong, n_resamples, seed)
        if 0 < wrong.sum() < len(wrong)
        else None,
        "fraction": fraction,
        "at_fraction": precision_recall(flagged, wrong),
        "best_possible_recall": min(1.0, round(fraction * len(given)) / max(1, int(wrong.sum()))),
        "target_precision": precision,
        "at_precision": recall_at_precision(score, wrong, precision),
        # Of the wrong labels the flag catches, how often the student's own top class is gold:
        # how much relabelling, not just dropping, could recover.
        "caught_argmax_is_gold": float((argmax[caught] == gold[caught]).mean())
        if caught.any()
        else None,
    }
    return out


# --- the run ----------------------------------------------------------------------------------


def _key(*parts: object) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode("utf-8")).hexdigest()[:20]


class _Trainer:
    """Trains the eval student on (text, label) pairs and caches what it predicts, keyed by
    everything the prediction depends on, so a re-run or a resumed run trains nothing twice."""

    def __init__(
        self,
        pool: Pool,
        student: StudentSpec,
        cache: Path,
        train: Callable[..., tuple[Predictor, object]] | None = None,
    ) -> None:
        self.pool = pool
        self.spec = pool.spec.model_copy(
            update={"student": student.model_copy(update={"seed": SEED})}
        )
        self.cache = cache
        self.corpus = [it.text for it, _ in pool.pool]
        self.train = train if train is not None else fit_student
        self.trained = 0

    def predict(
        self, items: Sequence[Item], labels: Sequence[str], targets: Sequence[str], tag: str
    ) -> FloatArray:
        key = _key(
            self.spec.student.model_dump(),
            [it.id for it in items],
            list(labels),
            [hashlib.sha256(t.encode("utf-8")).hexdigest() for t in targets],
        )
        path = self.cache / f"{tag}-{key}.npy"
        if path.exists():
            cached: FloatArray = np.load(path).astype(np.float64)
            return cached
        logger.info(
            "training a flag student",
            extra={"dataset": self.pool.spec.name, "run": tag, "n": len(items)},
        )
        started = time.perf_counter()
        student, _ = self.train(
            self.spec, self.pool.teacher_model, list(zip(items, labels, strict=True)), self.corpus
        )
        proba = np.asarray(student.predict_proba(list(targets)), dtype=np.float32)
        self.trained += 1
        logger.info(
            "trained a flag student",
            extra={"run": tag, "seconds": round(time.perf_counter() - started, 1)},
        )
        self.cache.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, proba)
        tmp.replace(path)
        return proba.astype(np.float64)

    def fit(self, tag: str) -> Fit:
        by_text = {it.text: it for it, _ in self.pool.pool}

        class _Fitted:
            def __init__(self, trainer: _Trainer, items: list[Item], labels: list[str]) -> None:
                self.trainer, self.items, self.labels = trainer, items, labels

            def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
                p = self.trainer.predict(self.items, self.labels, texts, tag)
                return p.astype(np.float32)

        def fit(texts: Sequence[str], labels: Sequence[str]) -> Predictor:
            return _Fitted(self, [by_text[t] for t in texts], list(labels))

        return fit


def _reference(pool: Pool, heldout_ids: list[str]) -> dict[str, list[str]] | None:
    """The committed seed-0 `eval-noise` students' held-out predictions by condition, or None
    when that report is absent or scored other held-out inputs."""
    path = noise_results_dir(pool.spec.name, pool.teacher_model, smoke=False) / "report.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["heldout"]["input_id"] != heldout_ids:
        logger.warning("the committed noise report scored other held-out inputs; not pairing")
        return None
    return {
        str(r["condition"]): list(r["predicted"]) for r in report["runs"] if int(r["seed"]) == SEED
    }


def _scored(
    proba: FloatArray, classes: list[str], gold: list[str], reference: list[str] | None, n: int
) -> dict[str, object]:
    predicted = [classes[i] for i in proba.argmax(axis=1)]
    out: dict[str, object] = {
        "vs_gold": _pair_stats(gold, predicted, n, SEED),
        "predicted": predicted,
    }
    if reference is not None:
        diff = bootstrap_difference_ci(
            gold, predicted, reference, "kappa", n_resamples=n, seed=SEED
        )
        out["minus_unfiltered"] = asdict(diff)
    return out


def run_dataset(
    name: str,
    settings: Settings,
    conditions: Sequence[str],
    n_resamples: int,
    size: int | None = None,
    epochs: int | None = None,
) -> tuple[dict[str, object], dict[str, npt.NDArray[np.generic]]]:
    pool = load_pool(name, settings, size)
    spec, classes = pool.spec, pool.spec.label_names
    index = {c: i for i, c in enumerate(classes)}
    items = [it for it, _ in pool.pool]
    texts = [it.text for it in items]
    pool_gold = pool.pool_gold
    gold_codes = np.array([index[g] for g in pool_gold], dtype=np.int64)
    h_items = [it for it, _ in pool.heldout]
    h_texts = [it.text for it in h_items]
    h_gold = [pool.data.gold[it.id] for it in h_items]
    h_ids = [it.id for it in h_items]
    reference = _reference(pool, h_ids) if size is None else None

    student = load_frontier_config().student
    if epochs is not None:
        student = student.model_copy(update={"epochs": epochs, "min_steps": 0})
    trainer = _Trainer(
        pool, student, settings.home / "eval" / f"flag-{spec.name}-{pool.data.sha256[:12]}"
    )
    folds = fold_ids(len(items), FOLDS, SEED)
    arrays: dict[str, npt.NDArray[np.generic]] = {
        "input_ids": np.array([it.id for it in items]),
        "classes": np.array(classes),
        "gold": gold_codes,
        "folds": folds,
    }
    by_condition: dict[str, object] = {}
    for condition in conditions:
        labels = _labels_for(condition, pool_gold, pool.pool_teacher, classes, SEED)
        given = np.array([index[lab] for lab in labels], dtype=np.int64)
        proba = out_of_fold_proba(
            texts, labels, folds, len(classes), trainer.fit(f"oof-{condition}")
        )
        flags = analyse_flags(proba, given, gold_codes, n_resamples=n_resamples)
        arrays[f"given_{condition}"] = given
        arrays[f"proba_{condition}"] = proba.astype(np.float32)

        def retrain(
            keep: BoolArray,
            tag: str,
            labels: list[str] = labels,
            given: IntArray = given,
            condition: str = condition,
        ) -> dict[str, object]:
            kept = [it for it, k in zip(items, keep, strict=True) if k]
            kept_labels = [lab for lab, k in zip(labels, keep, strict=True) if k]
            p = trainer.predict(kept, kept_labels, h_texts, tag)
            ref = reference.get(condition) if reference is not None else None
            return {
                "n_train": len(kept),
                "train_labels_wrong": int((given != gold_codes)[keep].sum()),
                **_scored(p, classes, h_gold, ref, n_resamples),
            }

        flagged = flag_lowest(self_confidence(proba, given), FLAG_FRACTION)
        retrains: dict[str, object] = {"filtered": retrain(~flagged, f"filtered-{condition}")}
        if condition == "teacher":
            rng = np.random.default_rng(SEED)
            random_keep = np.ones(len(items), dtype=bool)
            random_keep[rng.choice(len(items), size=int(flagged.sum()), replace=False)] = False
            retrains["random"] = retrain(random_keep, "random-teacher")
            retrains["unfiltered"] = retrain(np.ones(len(items), dtype=bool), "unfiltered-teacher")
            a = retrains["filtered"]
            b = retrains["random"]
            assert isinstance(a, dict)
            assert isinstance(b, dict)
            retrains["filtered_minus_random"] = asdict(
                bootstrap_difference_ci(
                    h_gold,
                    a["predicted"],
                    b["predicted"],
                    "kappa",
                    n_resamples=n_resamples,
                    seed=SEED,
                )
            )
            if reference is not None and "teacher" in reference:
                unf = retrains["unfiltered"]
                assert isinstance(unf, dict)
                retrains["reproduces_reference"] = {
                    "same_predictions": sum(
                        x == y for x, y in zip(unf["predicted"], reference["teacher"], strict=True)
                    ),
                    "n": len(h_ids),
                }
        if reference is not None and condition in reference:
            retrains["reference_vs_gold"] = _pair_stats(
                h_gold, reference[condition], n_resamples, SEED
            )
        by_condition[condition] = {
            "train_labels_wrong": int((given != gold_codes).sum()),
            "flags": flags,
            "retrain": retrains,
        }
    report: dict[str, object] = {
        "dataset": {
            "name": name,
            "repo": pool.config.dataset.repo,
            "revision": pool.config.dataset.revision,
            "sample_sha256": pool.data.sha256,
        },
        "teacher": {
            "model": pool.teacher_model,
            "digest": spec.teacher.digest,
            "prompt_hash": spec.prompt_hash,
        },
        "pool": {"n": len(items), "folds": FOLDS, "fold_seed": SEED},
        "heldout": {"n": len(h_ids), "input_id": h_ids, "gold": h_gold},
        "student": student.model_copy(update={"seed": SEED}).model_dump(),
        "flag_fraction": FLAG_FRACTION,
        "target_precision": TARGET_PRECISION,
        "reference": (
            str(noise_results_dir(name, pool.teacher_model, smoke=False) / "report.json")
            if reference is not None
            else None
        ),
        "students_trained": trainer.trained,
        "conditions": by_condition,
    }
    return report, arrays


def reanalyse(oof: Path, n_resamples: int) -> dict[str, object]:
    """The report next to `oof` with every condition's flag analysis recomputed from the
    committed out-of-fold probabilities (the retrains are kept as they are)."""
    previous = oof.parent / "report.json"
    if not previous.exists():
        raise ConfigError(f"{previous} not found next to {oof}")
    report: dict[str, object] = json.loads(previous.read_text(encoding="utf-8"))
    saved = np.load(oof)
    gold = saved["gold"].astype(np.int64)
    conditions = report["conditions"]
    assert isinstance(conditions, dict)
    for condition, entry in conditions.items():
        if f"proba_{condition}" not in saved:
            raise ConfigError(f"{oof} has no out-of-fold probabilities for {condition!r}")
        entry["flags"] = analyse_flags(
            saved[f"proba_{condition}"].astype(np.float64),
            saved[f"given_{condition}"].astype(np.int64),
            gold,
            fraction=float(str(report["flag_fraction"])),
            precision=float(str(report["target_precision"])),
            n_resamples=n_resamples,
        )
    return report


def results_dir(dataset: str, teacher_model: str, smoke: bool) -> Path:
    return RESULTS_DIR / (
        f"flag-{dataset}-{file_safe(teacher_model)}" + (SMOKE_SUFFIX if smoke else "")
    )


def summary_line(report: Mapping[str, object], out: Path) -> str:
    dataset = report["dataset"]
    conditions = report["conditions"]
    assert isinstance(dataset, dict)
    assert isinstance(conditions, dict)
    cells = []
    for condition, entry in conditions.items():
        auc = entry["flags"]["auroc"]
        at = entry["flags"]["at_precision"]
        cells.append(
            f"{condition} AUROC {auc['auroc']:.3f}, recall {at['recall']:.2f} at precision"
            f" {report['target_precision']}"
            if auc
            else condition
        )
    return f"{dataset['name']}: {'; '.join(cells)}; {out}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.flag", description=__doc__)
    parser.add_argument("datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--size", type=int, help="smaller sample (smoke run)")
    parser.add_argument("--epochs", type=int, help="override the student's epochs (smoke run)")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument(
        "--oof",
        type=Path,
        help="re-run the flag analysis from a committed oof.npz (no model, no dataset download)",
    )
    args = parser.parse_args(argv)
    if args.resamples < 1:
        parser.error("--resamples must be at least 1")
    if args.size is not None and args.size < 10:
        parser.error("--size must be at least 10")
    if args.oof is not None and args.datasets != list(DATASETS) and len(args.datasets) != 1:
        parser.error("--oof re-runs one report; pass no dataset")
    settings = Settings.from_env()
    log.configure(settings.log_level)
    args_shown = " ".join(argv if argv is not None else sys.argv[1:])
    command = f"python -m eval.flag {args_shown}".rstrip()
    provenance = git_provenance()
    smoke = args.size is not None or args.epochs is not None or args.resamples != DEFAULT_RESAMPLES
    try:
        if args.oof is not None:
            started = time.perf_counter()
            report = reanalyse(args.oof, args.resamples)
            out = args.oof.parent
            if smoke:
                out = out.with_name(out.name + SMOKE_SUFFIX)
                out.mkdir(parents=True, exist_ok=True)
            meta_path = args.oof.parent / "meta.json"
            meta = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else {}
            meta.update(
                {
                    "reanalysed": {
                        "command": command,
                        **provenance,
                        "n_resamples": args.resamples,
                        "wall_seconds": round(time.perf_counter() - started, 1),
                    }
                }
            )
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
            print(summary_line(report, out))  # noqa: T201 - the final summary
            return 0
        for name in args.datasets:
            started = time.perf_counter()
            report, arrays = run_dataset(
                name, settings, CONDITIONS, args.resamples, args.size, args.epochs
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
                "seed": SEED,
                "n_resamples": args.resamples,
                "teacher_model": teacher["model"],
                "teacher_digest": teacher["digest"],
                "student_model": load_frontier_config().student.model,
                "wall_seconds": round(wall, 1),
                "spent_usd": 0.0,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            np.savez_compressed(out / "oof.npz", **arrays)  # type: ignore[arg-type]
            (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
            (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
            print(summary_line(report, out))  # noqa: T201 - the final summary
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
