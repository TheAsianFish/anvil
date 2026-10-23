"""The tagger in the cascade (M12): does the threshold rule port from kappa over labels to
entity F1 over sentences, and does the tagger's confidence see the errors it makes?

`python -m eval.extract --predictions eval/results/extract-mit_restaurant-gold/report.json`
(`make eval-extract-cascade`) reads the held-out predictions an `eval.extract` run committed:
the teacher's and the student's spans for every held-out sentence and the student's confidence
in it, the tagger's minimum token probability (`EncoderTagger.predict_with_confidence`, rounded
to 4 decimals in the report). No model and no teacher are needed, so the study re-runs in
seconds. A cascade answers a sentence itself when that confidence reaches a threshold and sends
it to the teacher otherwise; agreement is micro entity F1 against the teacher on the answered
sentences (`evaluate.extraction.entity_f1_statistic`).

1. `curve`: F1 on the answered sentences at a grid of coverage levels.
2. `thresholds`: the threshold `calibrate.choose_statistic_threshold` picks at F1 0.9 and 0.95
   under the `point` rule (the estimate reaches the target), `lower` (its one-sided 95%
   percentile bound, sentences resampled) and `bca`, on the whole held-out split, with the
   error kinds of M4's matcher above and below each threshold (`errors_by_input`, so the two
   sides add up to the committed report's breakdown) and the cascade's F1 over every sentence.
3. `split_study`: the held-out split cut in half at random; the threshold is chosen on one half
   and the other half says whether the answered sentences really keep the target, as in M5's
   classification split study.
4. `detection`: how well low confidence picks out a sentence holding each kind of error, as the
   AUROC against the same clean sentences (no false positive, no false negative) and, as the
   pilot measured it, against every other sentence; and the difference between the boundary
   and the type AUROC with a sentence-resampled interval.

Raw output: `eval/results/cascade-extract-<dataset>-<teacher>/` (`report.json`, `meta.json`).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from anvil import __version__
from anvil.calibrate import Rule, choose_statistic_threshold, statistic_curve
from anvil.errors import ConfigError
from anvil.evaluate.binomial import clopper_pearson_lower
from anvil.evaluate.bootstrap import resample_indices
from anvil.evaluate.extraction import (
    ErrorBreakdown,
    entity_f1_statistic,
    entity_scores,
    error_breakdown,
    errors_by_input,
)
from anvil.spans import Span
from eval.shift import auroc

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
PREFIX = "cascade-extract-"
TARGETS = (0.9, 0.95)
RULES: tuple[Rule, ...] = ("point", "lower", "bca")
SERVED_RULE: Rule = "lower"
"""The rule whose thresholds the error-kind table splits at: the bound rule the hypotheses
name (the count rules need a binomial count, which F1 does not have)."""
COVERAGE_GRID = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2)
MIN_ACCEPTED = 30
DEFAULT_SPLITS = 200
DEFAULT_RESAMPLES = 1000
KINDS = ("wrong_boundary", "wrong_type", "wrong_both", "missed", "spurious")
"""M4's error kinds, in the order of the committed breakdown."""

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
Report = dict[str, Any]  # parsed JSON


@dataclass(frozen=True)
class Predictions:
    """The held-out sentences of one `eval.extract` run, as its report stores them."""

    ids: list[str]
    teacher: list[list[Span]]
    student: list[list[Span]]
    confidence: FloatArray


def _spans(rows: object, name: str) -> list[list[Span]]:
    if not isinstance(rows, list):
        raise ConfigError(f"predictions.{name} must be a list of span lists")
    try:
        return [[Span(int(s), int(e), str(t)) for s, e, t in row] for row in rows]
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"predictions.{name} holds a malformed span: {exc}") from exc


def load_predictions(report: Mapping[str, Any]) -> Predictions:
    """The `predictions` block of an `eval.extract` report; a `ConfigError` names what is
    missing (reports from before M4 stored no confidences)."""
    p = report.get("predictions")
    if not isinstance(p, dict):
        raise ConfigError("the report has no predictions block; re-run `make eval-extract`")
    missing = [k for k in ("input_id", "teacher", "student", "student_confidence") if k not in p]
    if missing:
        raise ConfigError(f"the report's predictions lack {', '.join(missing)}")
    ids = [str(x) for x in p["input_id"]]
    teacher, student = _spans(p["teacher"], "teacher"), _spans(p["student"], "student")
    confidence = np.asarray(p["student_confidence"], dtype=np.float64)
    if not (len(ids) == len(teacher) == len(student) == confidence.size) or not ids:
        raise ConfigError(
            "predictions need one teacher list, student list and confidence per input id"
        )
    if not np.isfinite(confidence).all():
        raise ConfigError("predictions.student_confidence holds NaN or infinite values")
    return Predictions(ids, teacher, student, confidence)


def _take(rows: Sequence[list[Span]], mask: BoolArray) -> list[list[Span]]:
    return [r for r, keep in zip(rows, mask, strict=True) if keep]


def _f1(teacher: Sequence[list[Span]], student: Sequence[list[Span]]) -> float:
    return entity_scores(teacher, student).f1


def _quantiles(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=np.float64)
    p5, p50, p95 = np.percentile(a, [5, 50, 95])
    return {"mean": float(a.mean()), "p5": float(p5), "median": float(p50), "p95": float(p95)}


def exact_interval(successes: int, trials: int) -> list[float]:
    """The two-sided 95% Clopper-Pearson interval on a share of splits (each one-sided 97.5%)."""
    low = float(clopper_pearson_lower(successes, trials, 0.975)[0])
    high = 1.0 - float(clopper_pearson_lower(trials - successes, trials, 0.975)[0])
    return [low, high]


def system_f1(pred: Predictions, accepted: BoolArray) -> float:
    """F1 of the cascade's output against the teacher over every sentence: the student's spans
    where it answers, the teacher's (so a perfect match) where it escalates."""
    out = [s if a else t for t, s, a in zip(pred.teacher, pred.student, accepted, strict=True)]
    return _f1(pred.teacher, out)


def side(pred: Predictions, mask: BoolArray) -> dict[str, object]:
    """F1 and the error kinds on the sentences in `mask`."""
    teacher, student = _take(pred.teacher, mask), _take(pred.student, mask)
    out: dict[str, object] = {"sentences": int(mask.sum())}
    if not teacher:
        return out
    scores = entity_scores(teacher, student)
    out.update(
        teacher_entities=scores.tp + scores.fn,
        student_entities=scores.tp + scores.fp,
        f1=scores.f1,
        errors=asdict(error_breakdown(teacher, student)),
    )
    return out


def curve(pred: Predictions) -> list[dict[str, object]]:
    """The F1-vs-coverage curve at the `COVERAGE_GRID` levels: for each, the point with the
    largest coverage not above it (tied confidences are answered together)."""
    stat = entity_f1_statistic(pred.teacher, pred.student)
    points = statistic_curve(pred.confidence, stat)
    rows: list[dict[str, object]] = []
    for level in COVERAGE_GRID:
        below = [p for p in points if p.coverage <= level + 1e-12]
        if below:
            p = below[-1]
            accepted = pred.confidence >= p.threshold
            rows.append({"level": level, **asdict(p), "system_f1": system_f1(pred, accepted)})
    return rows


def thresholds(pred: Predictions, n_resamples: int, seed: int) -> dict[str, dict[str, object]]:
    """Every rule's threshold at every target on the whole held-out split, and what it splits."""
    stat = entity_f1_statistic(pred.teacher, pred.student)
    out: dict[str, dict[str, object]] = {}
    for target in TARGETS:
        out[str(target)] = {}
        for rule in RULES:
            choice = choose_statistic_threshold(
                pred.confidence,
                stat,
                target,
                rule=rule,
                min_accepted=MIN_ACCEPTED,
                n_resamples=n_resamples,
                seed=seed,
            )
            if choice is None:
                out[str(target)][rule] = None
                continue
            accepted = pred.confidence >= choice.threshold
            out[str(target)][rule] = {
                **asdict(choice),
                "system_f1": system_f1(pred, accepted),
                "above": side(pred, accepted),
                "below": side(pred, ~accepted),
            }
    return out


def split_study(
    pred: Predictions, n_splits: int, n_resamples: int, seed: int
) -> dict[str, object]:
    """Fit the threshold on a random half, score the answered sentences of the other half."""
    n = len(pred.ids)
    if n < 2 * MIN_ACCEPTED:
        raise ConfigError(f"the split study needs at least {2 * MIN_ACCEPTED} sentences, got {n}")
    rng = np.random.default_rng(seed)
    outcomes: dict[str, dict[str, list[dict[str, float | bool]]]] = {
        str(t): {rule: [] for rule in RULES} for t in TARGETS
    }
    for split in range(n_splits):
        order = rng.permutation(n)
        fit, test = np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)
        fit[order[: n // 2]] = True
        test[order[n // 2 :]] = True
        stat = entity_f1_statistic(_take(pred.teacher, fit), _take(pred.student, fit))
        conf_fit, conf_test = pred.confidence[fit], pred.confidence[test]
        t_test, s_test = _take(pred.teacher, test), _take(pred.student, test)
        for target in TARGETS:
            for rule in RULES:
                choice = choose_statistic_threshold(
                    conf_fit,
                    stat,
                    target,
                    rule=rule,
                    min_accepted=MIN_ACCEPTED,
                    n_resamples=n_resamples,
                    seed=seed + split,
                )
                row: dict[str, float | bool] = {"found": choice is not None}
                if choice is not None:
                    accepted = conf_test >= choice.threshold
                    row["fit_coverage"] = choice.coverage
                    row["coverage"] = float(accepted.mean())
                    if accepted.any():
                        row["f1"] = _f1(_take(t_test, accepted), _take(s_test, accepted))
                        row["met"] = row["f1"] >= target
                outcomes[str(target)][rule].append(row)
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for key, by_rule in outcomes.items():
        summary[key] = {}
        for name, rows in by_rule.items():
            found = [r for r in rows if r["found"]]
            scored = [r for r in found if "f1" in r]
            met = sum(bool(r.get("met")) for r in found)
            entry: dict[str, object] = {
                "found": len(found) / len(rows),
                # One that answers nothing on the unseen half has kept the target on no
                # answered sentence, so it is not met (as in eval.calibration).
                "met": met / len(found) if found else None,
                "missed_share": (len(found) - met) / len(rows),
                "found_splits": len(found),
                "met_splits": met,
                "met_interval": exact_interval(met, len(found)) if found else None,
            }
            if scored:
                entry["coverage"] = _quantiles([float(r["coverage"]) for r in found])
                entry["fit_coverage"] = _quantiles([float(r["fit_coverage"]) for r in found])
                entry["f1"] = _quantiles([float(r["f1"]) for r in scored])
            summary[key][name] = entry
    return {
        "splits": n_splits,
        "fit_size": n // 2,
        "test_size": n - n // 2,
        "seed": seed,
        "n_resamples": n_resamples,
        "min_accepted": MIN_ACCEPTED,
        "targets": summary,
    }


def sentence_kinds(pred: Predictions) -> tuple[dict[str, BoolArray], BoolArray]:
    """For each error kind, the sentences holding at least one; and the clean sentences, those
    with no false positive and no false negative at all. A sentence can hold several kinds,
    and one can hold none yet not be clean: a student span that overlaps a teacher span
    another student span matches exactly is a false positive of no kind."""
    rows: list[ErrorBreakdown] = errors_by_input(pred.teacher, pred.student)
    kinds = {k: np.array([getattr(r, k) > 0 for r in rows], dtype=bool) for k in KINDS}
    clean = np.array(
        [set(t) == set(s) for t, s in zip(pred.teacher, pred.student, strict=True)], dtype=bool
    )
    return kinds, clean


def _interval(point: float, reps: FloatArray) -> dict[str, float]:
    ok = reps[~np.isnan(reps)]
    low, high = np.percentile(ok, [2.5, 97.5])
    return {"estimate": point, "low": float(low), "high": float(high), "resamples": int(ok.size)}


def _auroc_or_nan(score: FloatArray, positive: BoolArray, negative: BoolArray) -> float:
    if not positive.any() or not negative.any():
        return float("nan")
    return auroc(score[positive], score[negative])


COMPARISONS = ("clean", "rest")
"""What each kind's sentences are ranked against: the clean sentences (the plan's comparison,
the same negatives for every kind) or every other sentence (the pilot's, where the negatives
include sentences holding other errors)."""


def detection(pred: Predictions, n_resamples: int, seed: int) -> dict[str, object]:
    """AUROC of low confidence for "this sentence holds an error of kind k", against the clean
    sentences and against every other sentence, each with a 95% percentile interval, and the
    boundary-minus-type difference paired on the same resamples. Sentences are resampled, so a
    sentence's kinds and its confidence move together; a resample without a positive or a
    negative is skipped."""
    kinds, clean = sentence_kinds(pred)
    score = -pred.confidence  # higher means "more likely wrong"
    groups = {"any_error": ~clean, **kinds}
    idx = resample_indices(len(pred.ids), n_resamples, seed)

    def negatives(mask: BoolArray, against: str) -> BoolArray:
        return clean if against == "clean" else ~mask

    out: dict[str, object] = {
        "clean_sentences": int(clean.sum()),
        "boundary_and_type": int((kinds["wrong_boundary"] & kinds["wrong_type"]).sum()),
    }
    for against in COMPARISONS:
        by_kind: dict[str, object] = {}
        points: dict[str, float] = {}
        reps: dict[str, FloatArray] = {}
        for name, mask in groups.items():
            if name == "any_error" and against == "rest":
                continue  # the rest of the error sentences are the clean ones
            neg = negatives(mask, against)
            if not mask.any() or not neg.any():
                by_kind[name] = {"sentences": int(mask.sum())}
                continue
            points[name] = auroc(score[mask], score[neg])
            reps[name] = np.array([_auroc_or_nan(score[r], mask[r], neg[r]) for r in idx])
            by_kind[name] = {
                "sentences": int(mask.sum()),
                "auroc": _interval(points[name], reps[name]),
            }
        entry: dict[str, object] = {"by_kind": by_kind}
        if "wrong_boundary" in points and "wrong_type" in points:
            entry["boundary_minus_type"] = _interval(
                points["wrong_boundary"] - points["wrong_type"],
                reps["wrong_boundary"] - reps["wrong_type"],
            )
        out[against] = entry
    return out


def analyse(pred: Predictions, n_splits: int, n_resamples: int, seed: int) -> dict[str, object]:
    """Every number of the study from the predictions: a pure function of its inputs."""
    scores = entity_scores(pred.teacher, pred.student)
    return {
        "n": len(pred.ids),
        "f1": scores.f1,
        "errors": asdict(error_breakdown(pred.teacher, pred.student)),
        "confidence": _quantiles(pred.confidence.tolist()),
        "distinct_confidences": int(np.unique(pred.confidence).size),
        "curve": curve(pred),
        "thresholds": thresholds(pred, n_resamples, seed),
        "split_study": split_study(pred, n_splits, n_resamples, seed),
        "detection": detection(pred, n_resamples, seed),
    }


def check_against(report: Mapping[str, Any], analysis: Mapping[str, Any]) -> None:
    """The predictions must give back the report's own F1 and error breakdown: a mismatch
    means the predictions and the scores were not written by the same run."""
    agreement = report.get("agreement")
    if not isinstance(agreement, dict):
        return
    if "errors" in agreement and agreement["errors"] != analysis["errors"]:
        raise ConfigError(
            f"the predictions give error counts {analysis['errors']}, the report"
            f" {agreement['errors']}; the report's predictions block is out of date"
        )
    f1 = (agreement.get("entity") or {}).get("f1")
    if isinstance(f1, float) and abs(f1 - float(analysis["f1"])) > 1e-12:
        raise ConfigError(f"the predictions give F1 {analysis['f1']}, the report {f1}")


def results_dir(report: Mapping[str, Any]) -> Path:
    dataset = (report.get("dataset") or {}).get("name", "unknown")
    teacher = (report.get("teacher") or {}).get("model", "unknown")
    return RESULTS_DIR / f"{PREFIX}{dataset}-{str(teacher).replace(':', '-')}"


def run(
    report_path: Path,
    out_dir: Path,
    *,
    n_splits: int = DEFAULT_SPLITS,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
    command: str = "",
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Analyse the predictions in `report_path` and write `report.json` and `meta.json`."""
    if n_splits < 1 or n_resamples < 1:
        raise ConfigError(
            f"--splits and --resamples must be positive, got {n_splits} and {n_resamples}"
        )
    started = time.perf_counter()
    try:
        raw = report_path.read_bytes()
    except FileNotFoundError as exc:
        raise ConfigError(f"no extraction report at {report_path}") from exc
    report = json.loads(raw)
    pred = load_predictions(report)
    analysis = analyse(pred, n_splits, n_resamples, seed)
    check_against(report, analysis)
    wall = time.perf_counter() - started
    try:
        source_path = str(report_path.resolve().relative_to(ROOT))
    except ValueError:
        source_path = str(report_path)
    out: dict[str, object] = {
        "source": {
            "report": source_path,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dataset": (report.get("dataset") or {}).get("name"),
            "teacher_model": (report.get("teacher") or {}).get("model"),
            "student_model": (report.get("student") or {}).get("model"),
            "student_run": (report.get("student") or {}).get("run"),
            "confidence": "minimum token probability, rounded to 4 decimals",
        },
        "targets": list(TARGETS),
        "rules": list(RULES),
        "served_rule": SERVED_RULE,
        **analysis,
    }
    prov = dict(provenance) if provenance is not None else {}
    meta = {
        "command": command,
        "git_sha": prov.get("git_sha"),
        "git_dirty": prov.get("git_dirty"),
        "anvil": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "seed": seed,
        "n_resamples": n_resamples,
        "splits": n_splits,
        "teacher_model": out["source"]["teacher_model"],  # type: ignore[index]
        "wall_seconds": round(wall, 1),
        "spent_usd": 0.0,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out


def summary_line(report: Mapping[str, Any], out: Path) -> str:
    parts = []
    for target in TARGETS:
        choice = report["thresholds"][str(target)].get(SERVED_RULE)
        cov = "no threshold" if choice is None else f"coverage {choice['coverage']:.1%}"
        parts.append(f"F1 {target}: {cov}")
    diff = report["detection"]["clean"].get("boundary_minus_type")
    det = (
        ""
        if diff is None
        else f"; boundary - type AUROC {diff['estimate']:+.3f} [{diff['low']:+.3f},"
        f" {diff['high']:+.3f}]"
    )
    dataset = report["source"]["dataset"]
    return f"{dataset} cascade ({SERVED_RULE}): {', '.join(parts)}{det}; raw output {out}"
