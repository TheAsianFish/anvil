"""`make eval-shift`: what does the served cascade do with inputs it was never meant to see?

`make eval-calibration` chose the Banking77 bundle's threshold on held-out inputs drawn from
the same 2,500-row training sample, and every coverage number so far is measured there. This
study scores the exported bundle through `anvil/runtime.py`, exactly as `anvil serve` does,
on the sets in `eval/shift.yaml`, each read at a pinned revision:

1. In distribution, Banking77's own test split (no Anvil run has labelled or trained on it):
   coverage at the served threshold, kappa with gold on the inputs the student answers, and
   ECE before and after the bundle's temperature.
2. Out of distribution, CLINC150's out-of-scope utterances and AG News items: the share the
   student answers above the served threshold (every such answer is a Banking77 intent, so
   wrong by construction) and the confidence histograms.
3. Five scores for the same decision, all free at serving time: the served max softmax (with
   the bundle's temperature), max logit, energy (logsumexp of the logits, Liu et al. 2020),
   cosine similarity to the k-th nearest training embedding (Sun et al. 2022) and the negative
   Mahalanobis distance to the nearest class mean with a shared Ledoit-Wolf covariance (Lee et
   al. 2018). The first three come from the ONNX logits; the two distances use the mean-pooled
   last hidden layer (the vectors selection uses), computed with the PyTorch student, since
   the bundle's graph returns logits only. Each score is compared by AUROC (in- against
   out-of-distribution, paired bootstrap difference with max softmax) and as a second gate on
   top of the served threshold, set on the held-out inputs to give up `coverage_cost` of them,
   then measured on the test split and the out-of-distribution sets.

Everything after the forward passes runs on the committed `scores.npz` (logits and the two
distance scores per input), so `--scores` re-runs the analysis in seconds without a model.

Raw output: `eval/results/shift-<dataset>-<teacher>/` (`report.json`, `meta.json`,
`scores.npz`).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from anvil import __version__, log
from anvil.calibrate import THRESHOLD_TOLERANCE, ece, softmax
from anvil.errors import AnvilError, ConfigError
from anvil.evaluate.bootstrap import bootstrap_ci, resample_indices
from anvil.evaluate.generation import mean_ci
from anvil.evaluate.metrics import cohen_kappa
from anvil.settings import Settings
from eval.classify import TeacherKind, file_safe, git_provenance
from eval.data import DatasetRef, label_name, load_rows, near_duplicates

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
SHIFT_CONFIG = ROOT / "eval" / "shift.yaml"

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
Role = Literal["in_distribution", "out_of_distribution"]

SCORES = ("max_softmax", "max_logit", "energy", "knn", "mahalanobis")
"""Every score is oriented so that higher means "more like the training data"."""
BASELINE = "max_softmax"
HELDOUT = "heldout"
"""The held-out inputs of the eval-classify sample: where the bundle's threshold was chosen and
where each extra gate is set."""
ECE_BINS = 15
"""As in `make eval-calibration`."""

# The hypotheses PLAN.md states for M9, written down before the first run.
H1_COVERAGE_POINTS = 0.03
H1_KAPPA = 0.9
H2_ACCEPTANCE = 0.40
H3_AUROC_GAIN = 0.05

logger = log.get_logger("anvil.eval.shift")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SetCfg(_Strict):
    name: str
    role: Role
    description: str = ""
    repo: str
    revision: str = Field(min_length=40, max_length=40)
    """A full commit SHA: a branch name would let the rows change under a committed result."""
    config: str | None = None
    split: str = "test"
    text_field: str = "text"
    label_field: str = "label"
    keep: str | None = None
    """Keep only rows with this gold label (CLINC150's `oos`)."""
    size: int | None = Field(default=None, ge=1)
    seed: int = 0
    license: str = Field(min_length=1)

    def ref(self) -> DatasetRef:
        return DatasetRef(
            self.repo,
            self.revision,
            self.split,
            text_field=self.text_field,
            label_field=self.label_field,
            license=self.license,
            config=self.config,
        )

    def source(self) -> dict[str, object]:
        return {
            "repo": self.repo,
            "revision": self.revision,
            "config": self.config,
            "split": self.split,
            "keep": self.keep,
            "license": self.license,
            "description": self.description,
        }


class ShiftConfig(_Strict):
    dataset: str
    sets: list[SetCfg] = Field(min_length=2)
    knn_k: int = Field(default=5, ge=1)
    coverage_cost: float = Field(default=0.03, gt=0, lt=1)
    n_bins: int = Field(default=10, ge=2)
    n_resamples: int = Field(default=1000, ge=100)
    seed: int = 0

    @model_validator(mode="after")
    def _one_reference(self) -> ShiftConfig:
        names = [s.name for s in self.sets]
        if len(set(names)) != len(names) or HELDOUT in names:
            raise ValueError(f"set names must be unique and not {HELDOUT!r}: {names}")
        roles = [s.role for s in self.sets]
        if roles.count("in_distribution") != 1 or "out_of_distribution" not in roles:
            raise ValueError("need exactly one in_distribution set and at least one other")
        return self

    @property
    def reference(self) -> SetCfg:
        return next(s for s in self.sets if s.role == "in_distribution")


def load_shift_config(path: Path = SHIFT_CONFIG) -> ShiftConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return ShiftConfig.model_validate(raw)
    except FileNotFoundError as exc:
        raise ConfigError(f"shift config {path} not found") from exc
    except ValidationError as exc:
        raise ConfigError(f"{path} is invalid: {exc}") from exc


def select_rows(
    cfg: SetCfg, texts: Sequence[str], labels: Sequence[str]
) -> tuple[list[str], list[str], list[str]]:
    """Ids, texts and gold label names of the set: rows with the `keep` label (all rows
    without one), distinct non-empty texts, in dataset order or, with `size`, a seeded draw
    of that many. Ids are `<set>-<row index>`."""
    keep = label_name(cfg.keep) if cfg.keep is not None else None
    rows = [i for i, lab in enumerate(labels) if keep is None or lab == keep]
    if cfg.size is not None:
        rows = [rows[i] for i in np.random.default_rng(cfg.seed).permutation(len(rows))]
    ids: list[str] = []
    out_texts: list[str] = []
    out_labels: list[str] = []
    seen: set[str] = set()
    for i in rows:
        text = texts[i].strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ids.append(f"{cfg.name}-{i}")
        out_texts.append(text)
        out_labels.append(labels[i])
        if cfg.size is not None and len(ids) == cfg.size:
            break
    if not ids:
        raise ConfigError(f"set {cfg.name} has no rows (keep={cfg.keep!r}); check eval/shift.yaml")
    return ids, out_texts, out_labels


# ---------------------------------------------------------------------------------- scores


def logit_scores(logits: npt.ArrayLike, temperature: float) -> dict[str, FloatArray]:
    """The three scores one forward pass gives for free: the served confidence (max softmax at
    the bundle's temperature), the max logit, and the energy score logsumexp(z) at T = 1."""
    z = np.asarray(logits, dtype=np.float64)
    m = z.max(axis=1)
    energy = m + np.log(np.exp(z - m[:, None]).sum(axis=1))
    return {
        "max_softmax": softmax(z, temperature).max(axis=1),
        "max_logit": m,
        "energy": energy,
    }


def _unit(x: npt.ArrayLike) -> FloatArray:
    v = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    out: FloatArray = v / np.maximum(norms, 1e-12)
    return out


def knn_similarity(
    train: npt.ArrayLike, x: npt.ArrayLike, k: int, chunk: int = 1024
) -> FloatArray:
    """Cosine similarity of each row of `x` to its k-th nearest row of `train` (both
    L2-normalised): high when the input sits among training inputs."""
    t = _unit(train)
    q = _unit(x)
    if not 1 <= k <= len(t):
        raise ValueError(f"k must be between 1 and the {len(t)} training vectors, got {k}")
    out = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), chunk):
        sims = q[start : start + chunk] @ t.T
        out[start : start + chunk] = -np.partition(-sims, k - 1, axis=1)[:, k - 1]
    return out


@dataclass(frozen=True)
class Gaussians:
    """Class means and the shared precision of the training embeddings."""

    means: FloatArray
    precision: FloatArray


def fit_gaussians(train: npt.ArrayLike, codes: npt.ArrayLike) -> Gaussians:
    """One mean per class present in `codes`, and the inverse of the Ledoit-Wolf shrunk
    covariance of the class-centred vectors (with fewer vectors than dimensions the sample
    covariance is singular, so it cannot be inverted as it is)."""
    from sklearn.covariance import LedoitWolf

    x = np.asarray(train, dtype=np.float64)
    c = np.asarray(codes, dtype=np.int64)
    classes = np.unique(c)
    means = np.stack([x[c == k].mean(axis=0) for k in classes])
    centred = x - means[np.searchsorted(classes, c)]
    precision = np.asarray(LedoitWolf(assume_centered=True).fit(centred).precision_)
    return Gaussians(means, precision.astype(np.float64))


def mahalanobis_score(g: Gaussians, x: npt.ArrayLike) -> FloatArray:
    """Minus the squared Mahalanobis distance to the nearest class mean."""
    v = np.asarray(x, dtype=np.float64)
    vp = v @ g.precision
    mp = g.means @ g.precision
    d2 = (vp * v).sum(axis=1)[:, None] - 2 * vp @ g.means.T + (mp * g.means).sum(axis=1)[None, :]
    out: FloatArray = -d2.min(axis=1)
    return out


def auroc(positive: npt.ArrayLike, negative: npt.ArrayLike) -> float:
    """P(score of a positive > score of a negative), ties counting half: the Mann-Whitney U
    statistic over n_pos * n_neg, from average ranks."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError("AUROC needs at least one positive and one negative score")
    both = np.concatenate([pos, neg])
    _, inverse, counts = np.unique(both, return_inverse=True, return_counts=True)
    first = np.cumsum(counts) - counts  # ranks before each distinct value
    ranks = (first + (counts + 1) / 2)[inverse]
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return float(u / (len(pos) * len(neg)))


def fpr_at_tpr(positive: npt.ArrayLike, negative: npt.ArrayLike, tpr: float = 0.95) -> float:
    """Share of negatives at or above the score that keeps `tpr` of the positives."""
    pos = np.sort(np.asarray(positive, dtype=np.float64))
    neg = np.asarray(negative, dtype=np.float64)
    cut = pos[min(math.floor((1 - tpr) * len(pos) + 1e-9), len(pos) - 1)]
    return float(np.mean(neg >= cut))


def served(confidence: npt.ArrayLike, threshold: float) -> BoolArray:
    """The inputs `anvil serve` answers itself (the same tolerance as the server)."""
    out: BoolArray = np.asarray(confidence) >= threshold - THRESHOLD_TOLERANCE
    return out


def gate_threshold(scores: npt.ArrayLike, accepted: npt.ArrayLike, cost: float) -> float:
    """The cut on `scores` that, applied on top of `accepted`, gives up at most `cost` of all
    the inputs (floor(cost * n) of them, fewer when scores tie at the cut): accept score >=
    the returned value."""
    s = np.asarray(scores, dtype=np.float64)
    a = np.asarray(accepted, dtype=bool)
    drop = math.floor(cost * len(s) + 1e-9)
    kept = np.sort(s[a])
    if drop == 0 or not len(kept):
        return -math.inf
    if drop >= len(kept):
        return math.inf
    return float(kept[drop])


def histogram(confidence: npt.ArrayLike, n_bins: int) -> list[int]:
    """Counts of the served confidence in equal-width bins over [0, 1]."""
    counts, _ = np.histogram(np.asarray(confidence), bins=np.linspace(0.0, 1.0, n_bins + 1))
    return [int(c) for c in counts]


# -------------------------------------------------------------------------------- analysis


@dataclass(frozen=True)
class Scored:
    """One set after the forward passes: what `scores.npz` holds for it."""

    ids: list[str]
    texts: list[str]
    logits: FloatArray
    knn: FloatArray
    mahalanobis: FloatArray
    gold: IntArray | None = None
    """Gold label codes in the bundle's label order (in-distribution sets only)."""

    def scores(self, temperature: float) -> dict[str, FloatArray]:
        return {
            **logit_scores(self.logits, temperature),
            "knn": self.knn,
            "mahalanobis": self.mahalanobis,
        }


def _interval(stats: object) -> dict[str, float]:
    d = asdict(stats)  # type: ignore[call-overload]
    return {k: d[k] for k in ("estimate", "low", "high")}


def _rate(mask: BoolArray, n_resamples: int, seed: int) -> dict[str, float]:
    return _interval(mean_ci([bool(x) for x in mask], n_resamples, seed=seed))


def _auroc_comparison(
    ref: Mapping[str, FloatArray], ood: Mapping[str, FloatArray], n_resamples: int, seed: int
) -> dict[str, dict[str, object]]:
    """AUROC of every score with a percentile bootstrap interval, and its paired difference
    from max softmax: in- and out-of-distribution inputs are resampled separately, and every
    score sees the same resamples."""
    n_ref, n_ood = len(ref[BASELINE]), len(ood[BASELINE])
    idx_ref = resample_indices(n_ref, n_resamples, seed)
    idx_ood = resample_indices(n_ood, n_resamples, seed + 1)
    reps = {
        name: np.array(
            [auroc(ref[name][i], ood[name][j]) for i, j in zip(idx_ref, idx_ood, strict=True)]
        )
        for name in SCORES
    }
    out: dict[str, dict[str, object]] = {}
    for name in SCORES:
        point = auroc(ref[name], ood[name])
        low, high = np.percentile(reps[name], [2.5, 97.5])
        diff = reps[name] - reps[BASELINE]
        d_low, d_high = np.percentile(diff, [2.5, 97.5])
        out[name] = {
            "auroc": {"estimate": point, "low": float(low), "high": float(high)},
            "minus_max_softmax": {
                "estimate": point - auroc(ref[BASELINE], ood[BASELINE]),
                "low": float(d_low),
                "high": float(d_high),
            },
            "fpr_at_95_tpr": fpr_at_tpr(ref[name], ood[name]),
        }
    return out


def _top(labels: Sequence[str], k: int = 5) -> list[dict[str, object]]:
    return [{"label": lab, "n": n} for lab, n in Counter(labels).most_common(k)]


def analyse(
    sets: Mapping[str, Scored],
    roles: Mapping[str, Role],
    heldout: Scored,
    labels: Sequence[str],
    temperature: float,
    threshold: float,
    *,
    cost: float,
    n_bins: int,
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """The whole study from the scored sets: a pure function of its inputs."""
    ref_name = next(n for n, r in roles.items() if r == "in_distribution")
    ref = sets[ref_name]
    if ref.gold is None or heldout.gold is None:
        raise ValueError("the in-distribution set and the held-out inputs need gold codes")
    scores = {name: s.scores(temperature) for name, s in sets.items()}
    held_scores = heldout.scores(temperature)
    answered = {name: served(sc[BASELINE], threshold) for name, sc in scores.items()}
    held_answered = served(held_scores[BASELINE], threshold)

    per_set: dict[str, object] = {}
    for i, (name, s) in enumerate(sets.items()):
        conf = scores[name][BASELINE]
        pred = s.logits.argmax(axis=1)
        mask = answered[name]
        entry: dict[str, object] = {
            "role": roles[name],
            "n": len(s.ids),
            "answered": _rate(mask, n_resamples, seed + i),
            "answered_n": int(mask.sum()),
            "mean_confidence": float(conf.mean()),
            "histogram": histogram(conf, n_bins),
        }
        if roles[name] == "out_of_distribution":
            order = np.argsort(-conf, kind="stable")[:8]
            entry["top_answers"] = _top([labels[k] for k in pred[mask]])
            entry["most_confident"] = [
                {
                    "id": s.ids[k],
                    "text": s.texts[k],
                    "label": labels[int(pred[k])],
                    "confidence": float(conf[k]),
                }
                for k in order
            ]
        per_set[name] = entry

    # The in-distribution reading: what the threshold chosen on held-out does on the test split.
    gold = [labels[k] for k in ref.gold]
    pred_ref = [labels[k] for k in ref.logits.argmax(axis=1)]
    mask = answered[ref_name]
    on = np.flatnonzero(mask)
    held_gold = [labels[k] for k in heldout.gold]
    held_pred = [labels[k] for k in heldout.logits.argmax(axis=1)]
    held_on = np.flatnonzero(held_answered)
    reference = {
        "set": ref_name,
        "heldout_n": len(heldout.ids),
        "heldout_coverage": float(held_answered.mean()),
        "heldout_kappa_answered": cohen_kappa(
            [held_pred[k] for k in held_on], [held_gold[k] for k in held_on]
        ),
        "coverage": _rate(mask, n_resamples, seed),
        "kappa_all": cohen_kappa(pred_ref, gold),
        "kappa_answered": _interval(
            bootstrap_ci(
                [pred_ref[k] for k in on], [gold[k] for k in on], "kappa", n_resamples, seed=seed
            )
        ),
        "accuracy_answered": float(np.mean([pred_ref[k] == gold[k] for k in on])),
        "ece_raw": ece(softmax(ref.logits, 1.0), ref.gold, ECE_BINS),
        "ece_calibrated": ece(softmax(ref.logits, temperature), ref.gold, ECE_BINS),
        "ece_bins": ECE_BINS,
    }

    # Every score as an extra gate on top of the served threshold, set on held-out.
    ood_names = [n for n, r in roles.items() if r == "out_of_distribution"]
    comparison = {
        name: _auroc_comparison(scores[ref_name], scores[name], n_resamples, seed)
        for name in ood_names
    }
    gates: dict[str, object] = {}
    for score in SCORES:
        cut = gate_threshold(held_scores[score], held_answered, cost)
        held_gated = held_answered & (held_scores[score] >= cut)
        gated = {name: answered[name] & (scores[name][score] >= cut) for name in sets}
        ref_on = np.flatnonzero(gated[ref_name])
        gates[score] = {
            "cut": cut if math.isfinite(cut) else None,
            "heldout_coverage": float(held_gated.mean()),
            "coverage": float(gated[ref_name].mean()),
            "coverage_cost": float(mask.mean() - gated[ref_name].mean()),
            "kappa_answered": cohen_kappa(
                [pred_ref[k] for k in ref_on], [gold[k] for k in ref_on]
            ),
            "answered": {name: float(gated[name].mean()) for name in ood_names},
        }

    h1_cov = abs(reference["coverage"]["estimate"] - reference["heldout_coverage"])  # type: ignore[index]
    kappa_answered = reference["kappa_answered"]
    assert isinstance(kappa_answered, dict)
    gains = {
        score: {name: comparison[name][score]["minus_max_softmax"] for name in ood_names}
        for score in SCORES
        if score != BASELINE
    }
    winners = [
        score
        for score, by_set in gains.items()
        if all(d["estimate"] >= H3_AUROC_GAIN for d in by_set.values())  # type: ignore[index]
        and gates[score]["coverage_cost"] < cost  # type: ignore[index]
    ]
    hypotheses = {
        "h1": {
            "coverage_change": h1_cov,
            "kappa_answered": kappa_answered["estimate"],
            "held": bool(h1_cov <= H1_COVERAGE_POINTS and kappa_answered["estimate"] >= H1_KAPPA),
        },
        "h2": {
            name: {
                "answered": float(answered[name].mean()),
                "held": bool(answered[name].mean() >= H2_ACCEPTANCE),
            }
            for name in ood_names
        },
        "h3": {"scores": winners, "held": bool(winners)},
    }
    return {
        "labels": len(labels),
        "temperature": temperature,
        "threshold": threshold,
        "coverage_cost": cost,
        "n_bins": n_bins,
        "n_resamples": n_resamples,
        "seed": seed,
        "sets": per_set,
        "reference": reference,
        "separation": comparison,
        "gates": gates,
        "hypotheses": hypotheses,
        "criteria": {
            "h1_coverage_points": H1_COVERAGE_POINTS,
            "h1_kappa": H1_KAPPA,
            "h2_answered": H2_ACCEPTANCE,
            "h3_auroc_gain": H3_AUROC_GAIN,
        },
    }


# ------------------------------------------------------------------------------ scores.npz


def save_scores(
    path: Path, sets: Mapping[str, Scored], heldout: Scored, context: Mapping[str, object]
) -> None:
    arrays: dict[str, npt.NDArray[np.generic]] = {"context": np.array(json.dumps(context))}
    for name, s in {**sets, HELDOUT: heldout}.items():
        arrays[f"{name}.ids"] = np.array(s.ids)
        arrays[f"{name}.texts"] = np.array(s.texts)
        arrays[f"{name}.logits"] = s.logits.astype(np.float32)
        arrays[f"{name}.knn"] = s.knn
        arrays[f"{name}.mahalanobis"] = s.mahalanobis
        if s.gold is not None:
            arrays[f"{name}.gold"] = s.gold
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]


def load_scores(path: Path) -> tuple[dict[str, Scored], Scored, dict[str, object]]:
    """The sets (in the order the run scored them), the held-out inputs and the run context."""
    with np.load(path) as data:
        context = json.loads(str(data["context"]))
        names = [*context["sets"], HELDOUT]
        loaded = {
            name: Scored(
                ids=[str(x) for x in data[f"{name}.ids"]],
                texts=[str(x) for x in data[f"{name}.texts"]],
                logits=data[f"{name}.logits"].astype(np.float64),
                knn=data[f"{name}.knn"].astype(np.float64),
                mahalanobis=data[f"{name}.mahalanobis"].astype(np.float64),
                gold=(data[f"{name}.gold"].astype(np.int64) if f"{name}.gold" in data else None),
            )
            for name in names
        }
    heldout = loaded.pop(HELDOUT)
    return loaded, heldout, context


def report_from(
    sets: Mapping[str, Scored], heldout: Scored, context: Mapping[str, object]
) -> dict[str, object]:
    """`analyse` with the settings recorded in `context`, plus the context itself."""
    bundle = context["bundle"]
    settings = context["settings"]
    roles = context["roles"]
    assert isinstance(bundle, dict)
    assert isinstance(settings, dict)
    assert isinstance(roles, dict)
    result = analyse(
        sets,
        roles,
        heldout,
        bundle["labels"],
        float(bundle["temperature"]),
        float(bundle["threshold"]),
        cost=float(settings["coverage_cost"]),
        n_bins=int(settings["n_bins"]),
        n_resamples=int(settings["n_resamples"]),
        seed=int(settings["seed"]),
    )
    sources = context.get("sources") or {}
    assert isinstance(sources, dict)
    per_set = result["sets"]
    assert isinstance(per_set, dict)
    for name, entry in per_set.items():
        entry["source"] = sources.get(name)
    keep = ("source", "bundle", "settings", "overlap", "onnx_vs_torch", "seconds")
    extra = {k: context[k] for k in keep if k in context}
    bundle_info = dict(extra.get("bundle") or {})  # type: ignore[call-overload]
    bundle_info.pop("labels", None)
    extra["bundle"] = bundle_info
    return {**extra, **result}


# ----------------------------------------------------------------------------- forward pass


def run(
    config: ShiftConfig, teacher: TeacherKind, settings: Settings
) -> tuple[dict[str, Scored], Scored, dict[str, object]]:
    """Load (or train and export) the eval student, score every set, return what
    `scores.npz` holds."""
    from anvil.runtime import BUNDLE_META, OnnxStudent, read_bundle_meta
    from anvil.store import Split
    from anvil.students.encoder import EncoderClassifier
    from eval.export import bundle_dir
    from eval.export import run as export_run
    from eval.student import trained_student

    started = time.perf_counter()
    ts = trained_student(config.dataset, teacher, settings)
    bundle = bundle_dir(ts)
    meta = read_bundle_meta(bundle) if (bundle / BUNDLE_META).exists() else None
    if meta is None or meta.get("run") != ts.path.name:
        logger.info("exporting the eval student", extra={"bundle": str(bundle)})
        export_run(config.dataset, teacher, settings, probe=False)
        meta = read_bundle_meta(bundle)
    onnx = OnnxStudent.load(bundle, "fp32")
    if onnx.threshold is None:
        raise ConfigError(
            f"the bundle in {bundle} has no threshold (its calibration found none at the"
            " target), so there is no served cascade to study"
        )
    torch_student = EncoderClassifier.load(ts.path)
    labels = onnx.labels
    index = {lab: i for i, lab in enumerate(labels)}
    seconds: dict[str, float] = {"student": round(time.perf_counter() - started, 1)}

    with ts.workspace.open_store(ts.spec) as store:
        pool = store.labelled(ts.teacher_model, ts.spec.prompt_hash, Split.POOL)
    train_texts = [it.text for it, _ in pool]
    train_codes = np.array([index[lab] for _, lab in pool], dtype=np.int64)
    tick = time.perf_counter()
    _, train_vectors = torch_student.predict_with_embeddings(train_texts)
    gaussians = fit_gaussians(train_vectors, train_codes)
    seconds["training_embeddings"] = round(time.perf_counter() - tick, 1)

    diffs: dict[str, float] = {}

    def score(name: str, ids: list[str], texts: list[str], gold: IntArray | None) -> Scored:
        tick = time.perf_counter()
        logits = onnx.logits(texts).astype(np.float64)
        torch_probs, vectors = torch_student.predict_with_embeddings(texts)
        diffs[name] = float(np.abs(softmax(logits) - torch_probs).max())
        seconds[name] = round(time.perf_counter() - tick, 1)
        logger.info("scored", extra={"set": name, "n": len(texts), "seconds": seconds[name]})
        return Scored(
            ids,
            texts,
            logits,
            knn_similarity(train_vectors, vectors, config.knn_k),
            mahalanobis_score(gaussians, vectors),
            gold,
        )

    held = score(
        HELDOUT,
        [it.id for it, _ in ts.heldout],
        [it.text for it, _ in ts.heldout],
        np.array([index[lab] for _, lab in ts.heldout], dtype=np.int64),
    )
    sets: dict[str, Scored] = {}
    sources: dict[str, object] = {}
    overlap: dict[str, object] = {}
    sample_texts = [it.text for it in ts.data.items]
    for cfg in config.sets:
        ids, texts, gold_names = select_rows(cfg, *load_rows(cfg.ref()))
        gold = None
        if cfg.role == "in_distribution":
            unknown = sorted(set(gold_names) - set(index))
            if unknown:
                raise ConfigError(f"{cfg.name} has labels the student lacks: {unknown[:5]}")
            gold = np.array([index[lab] for lab in gold_names], dtype=np.int64)
            in_sample = set(sample_texts)
            overlap[cfg.name] = {
                "identical_to_sample": sum(t in in_sample for t in texts),
                "near_duplicates_of_sample": near_duplicates(texts, sample_texts),
                "sample_size": len(sample_texts),
            }
        sources[cfg.name] = {**cfg.source(), "rows": len(ids)}
        sets[cfg.name] = score(cfg.name, ids, texts, gold)

    calibration = meta.get("calibration")
    context: dict[str, object] = {
        "source": {
            "dataset": config.dataset,
            "sample_sha256": ts.data.sha256,
            "teacher_kind": ts.kind,
            "teacher_model": ts.teacher_model,
            "student_model": ts.spec.student.model,
            "student_run": ts.path.name,
            "trained_here": ts.trained_here,
            "train_inputs": len(train_texts),
        },
        "bundle": {
            "path": str(bundle.relative_to(ts.workspace.root)),
            "variant": onnx.variant,
            "labels": labels,
            "temperature": onnx.temperature,
            "threshold": onnx.threshold,
            "heldout_coverage_at_export": (
                calibration.get("coverage") if isinstance(calibration, dict) else None
            ),
        },
        "settings": {
            "knn_k": config.knn_k,
            "coverage_cost": config.coverage_cost,
            "n_bins": config.n_bins,
            "n_resamples": config.n_resamples,
            "seed": config.seed,
        },
        "sets": list(sets),
        "roles": {cfg.name: cfg.role for cfg in config.sets},
        "sources": sources,
        "overlap": overlap,
        "onnx_vs_torch": {
            "max_abs_prob_diff": diffs,
            "note": "max |softmax(ONNX logits) - PyTorch softmax| per set, at T = 1",
        },
        "seconds": seconds,
    }
    return sets, held, context


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.shift", description=__doc__)
    parser.add_argument(
        "--teacher", choices=["auto", "claude", "ollama", "replay", "gold"], default="auto"
    )
    parser.add_argument("--config", type=Path, default=SHIFT_CONFIG)
    parser.add_argument(
        "--scores",
        type=Path,
        help="re-run the analysis from a committed scores.npz (no model, no download)",
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.shift " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    started = time.perf_counter()
    try:
        if args.scores is not None:
            if not args.scores.is_file():
                raise ConfigError(f"{args.scores} not found; run `make eval-shift` first")
            sets, heldout, context = load_scores(args.scores)
            out = args.scores.parent
        else:
            config = load_shift_config(args.config)
            sets, heldout, context = run(config, args.teacher, settings)
            source = context["source"]
            assert isinstance(source, dict)
            out = RESULTS_DIR / f"shift-{config.dataset}-{file_safe(str(source['teacher_model']))}"
            out.mkdir(parents=True, exist_ok=True)
            save_scores(out / "scores.npz", sets, heldout, context)
        report = report_from(sets, heldout, context)
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    source = report["source"]
    assert isinstance(source, dict)
    meta_path = out / "meta.json"
    if args.scores is not None and meta_path.exists():
        # A re-analysis keeps the forward pass's provenance and records its own command.
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["reanalysed"] = {"command": command, **provenance}
    else:
        meta = {
            "command": command,
            **provenance,
            "anvil": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
            "versions": _versions(),
            "teacher_model": source["teacher_model"],
            "wall_seconds": round(time.perf_counter() - started, 1),
            "spent_usd": 0.0,
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    ref = report["reference"]
    assert isinstance(ref, dict)
    print(  # noqa: T201 - the final summary
        f"{source['dataset']}: test coverage {ref['coverage']['estimate']:.1%} (held-out"
        f" {ref['heldout_coverage']:.1%}), kappa answered"
        f" {ref['kappa_answered']['estimate']:.3f}; raw output {out}"
    )
    return 0


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in ("torch", "transformers", "onnxruntime", "tokenizers", "scikit-learn", "datasets"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


if __name__ == "__main__":
    raise SystemExit(main())
