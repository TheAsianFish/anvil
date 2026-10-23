"""Temperature scaling, reliability, and the confidence threshold of the student/teacher cascade.

The student answers an input when its calibrated confidence (the max softmax probability after
temperature scaling) reaches a threshold, and escalates it to the teacher otherwise. Here
"correct" means "agrees with the teacher": the targets are the teacher's label codes on held-out
inputs, so calibration is calibration of agreement with the teacher, not of truth.

Three steps, all numpy (this module is imported on the torch-free serving image):

1. `fit_temperature` picks the single T that minimises the held-out NLL of softmax(z / T)
   (Guo et al. 2017). It does not change the argmax, so accuracy and kappa on all inputs are
   untouched; it only rescales the confidences.
2. `reliability` and `ece` measure how far confidence is from agreement, before and after.
3. `coverage_curve` and `choose_threshold` trade coverage (share the student answers) against
   kappa on the answered inputs, and pick the threshold that keeps a target kappa at the
   largest coverage, optionally on a one-sided lower bound (bootstrap percentile or BCa, or
   the exact or Wilson binomial bound on the agreement count) so the target holds with stated
   confidence rather than only on this sample.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from anvil.evaluate.binomial import CountMethod, kappa_lower_from_counts
from anvil.evaluate.bootstrap import Method, lower_bounds, resample_indices

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ArrayLike2D = npt.ArrayLike
"""Anything np.asarray turns into a (n, n_labels) float array: an ndarray or nested lists."""
Rule = Literal["point", "lower", "bca", "exact", "wilson"]
"""How `choose_threshold` decides that a threshold keeps the target: "point", the kappa estimate
reaches it; "lower", its one-sided percentile-bootstrap lower bound does; "bca", its one-sided
BCa lower bound does; "exact", the Clopper-Pearson lower bound on the agreement count, mapped to
kappa through the sample's chance term, does; "wilson", the same with the Wilson score bound."""
RULES: tuple[Rule, ...] = ("point", "lower", "bca", "exact", "wilson")
DEFAULT_RULE: Rule = "exact"
"""The default since M11, from `make eval-coverage` (RESULTS.md, "Does the interval cover?"):
the bootstrap bounds cover about 93% where they claim 95%; the exact bound covers the agreement
rate at least 95% at every rate by construction (Wilson dips to 92.5%) and, through the plug-in
chance term, kappa in every simulated cell. It costs the most served coverage where the
kappa-vs-coverage curve is flat near the target; "wilson" is the cheaper rule for a user who
accepts its pointwise dips."""
BOUND_METHOD: dict[Rule, Method] = {"lower": "percentile", "bca": "bca"}
"""The bootstrap rules and the method that reads their replicates."""
COUNT_METHOD: dict[Rule, CountMethod] = {"exact": "exact", "wilson": "wilson"}
"""The rules bounded from the agreement count, with no resampling."""

_BISECTION_STEPS = 60
TEMPERATURE_BOUNDS = (0.05, 20.0)


def _logits(logits: ArrayLike2D) -> FloatArray:
    z = np.asarray(logits, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] < 1 or z.shape[1] < 1:
        raise ValueError(f"logits must have shape (n, n_labels) with n >= 1, got {z.shape}")
    if not np.isfinite(z).all():
        raise ValueError("logits contain NaN or infinite values; check the student's outputs")
    return z


def _codes(codes: npt.ArrayLike, n: int, n_labels: int, name: str) -> IntArray:
    c = np.asarray(codes)
    if c.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {c.shape}")
    if c.size and not np.issubdtype(c.dtype, np.integer):
        raise ValueError(f"{name} must be integer label codes, got dtype {c.dtype}")
    out = c.astype(np.int64)
    if out.size and (out.min() < 0 or out.max() >= n_labels):
        raise ValueError(f"{name} codes must lie in [0, {n_labels})")
    return out


def _temperature(temperature: float) -> float:
    if not (math.isfinite(temperature) and temperature > 0):
        raise ValueError(f"temperature must be a positive finite number, got {temperature}")
    return float(temperature)


def softmax(logits: ArrayLike2D, temperature: float = 1.0) -> FloatArray:
    """Row-wise softmax(z / T) in float64, stable: the row max is subtracted before exp."""
    z = _logits(logits) / _temperature(temperature)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    out: FloatArray = e / e.sum(axis=1, keepdims=True)
    return out


def nll(logits: ArrayLike2D, targets: npt.ArrayLike, temperature: float = 1.0) -> float:
    """Mean negative log-likelihood of `targets` under softmax(logits / T).

    Computed as mean_i [logsumexp(z_i / T) - z_{i, y_i} / T], never through log(softmax), so it
    stays finite for very confident wrong predictions.
    """
    z = _logits(logits) / _temperature(temperature)
    y = _codes(targets, z.shape[0], z.shape[1], "targets")
    m = z.max(axis=1)
    lse = m + np.log(np.exp(z - m[:, None]).sum(axis=1))
    return float(np.mean(lse - z[np.arange(z.shape[0]), y]))


def fit_temperature(
    logits: ArrayLike2D, targets: npt.ArrayLike, bounds: tuple[float, float] = TEMPERATURE_BOUNDS
) -> float:
    """The temperature T in `bounds` that minimises `nll(logits, targets, T)`.

    With beta = 1 / T the NLL is mean_i [logsumexp(beta z_i) - beta z_{i, y_i}], convex in beta
    (logsumexp is convex, the second term linear), with derivative

        g(beta) = mean_i (E_{p_beta}[z_i] - z_{i, y_i}),

    non-decreasing in beta. The minimiser is the root of g, found by bisection on log(beta)
    over [1 / T_max, 1 / T_min] (60 halvings, far below float precision of T). When g does not
    change sign the minimum is at a bound: g < 0 throughout (e.g. perfectly separable data,
    where sharper is always better) gives the smallest T, g > 0 throughout the largest. When g
    is zero everywhere (every row constant, so T cannot matter) this returns 1.0 if it lies in
    `bounds`.
    """
    z = _logits(logits)
    y = _codes(targets, z.shape[0], z.shape[1], "targets")
    t_min, t_max = bounds
    if not (0 < t_min < t_max and math.isfinite(t_max)):
        raise ValueError(f"bounds must satisfy 0 < low < high < inf, got {bounds}")
    z = z - z.max(axis=1, keepdims=True)  # g is invariant to a per-row shift
    z_y = z[np.arange(z.shape[0]), y]

    def grad(log_beta: float) -> float:
        p = softmax(z * math.exp(log_beta))
        return float(np.mean((p * z).sum(axis=1) - z_y))

    lo, hi = math.log(1 / t_max), math.log(1 / t_min)
    g_lo, g_hi = grad(lo), grad(hi)
    if g_lo >= 0 and g_hi <= 0:
        return 1.0 if t_min <= 1.0 <= t_max else t_max
    if g_lo >= 0:
        return t_max
    if g_hi <= 0:
        return t_min
    for _ in range(_BISECTION_STEPS):
        mid = (lo + hi) / 2
        if grad(mid) < 0:
            lo = mid
        else:
            hi = mid
    return float(1 / math.exp((lo + hi) / 2))


def _probs(probs: ArrayLike2D) -> FloatArray:
    p = np.asarray(probs, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 1 or p.shape[1] < 1:
        raise ValueError(f"probs must have shape (n, n_labels) with n >= 1, got {p.shape}")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("probs must be finite probabilities in [0, 1]; pass softmax outputs")
    return p


@dataclass(frozen=True)
class ReliabilityBin:
    """One confidence bin (low, high] of the reliability diagram (the first bin includes 0)."""

    low: float
    high: float
    n: int
    confidence: float
    """Mean max probability of the items in the bin (0.0 when empty)."""
    accuracy: float
    """Share of the items in the bin whose argmax equals the target (0.0 when empty)."""


def reliability(
    probs: ArrayLike2D, targets: npt.ArrayLike, n_bins: int = 15
) -> list[ReliabilityBin]:
    """Equal-width bins over [0, 1] on the max probability: bin i is (i / B, (i + 1) / B].

    Bins are closed on the right, so a confidence of exactly 1.0 lands in the last bin and one
    exactly on an edge in the bin below it; 0.0 goes to the first bin. Edges are
    np.linspace(0, 1, B + 1), and placement compares against those same floats.
    """
    p = _probs(probs)
    y = _codes(targets, p.shape[0], p.shape[1], "targets")
    if n_bins < 1:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    conf = p.max(axis=1)
    hit = (p.argmax(axis=1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    which = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    counts = np.bincount(which, minlength=n_bins)
    conf_sum = np.bincount(which, weights=conf, minlength=n_bins)
    hit_sum = np.bincount(which, weights=hit, minlength=n_bins)
    return [
        ReliabilityBin(
            low=float(edges[i]),
            high=float(edges[i + 1]),
            n=int(counts[i]),
            confidence=float(conf_sum[i] / counts[i]) if counts[i] else 0.0,
            accuracy=float(hit_sum[i] / counts[i]) if counts[i] else 0.0,
        )
        for i in range(n_bins)
    ]


def ece(probs: ArrayLike2D, targets: npt.ArrayLike, n_bins: int = 15) -> float:
    """Expected calibration error: sum_b (n_b / N) * |accuracy_b - confidence_b| (Guo et al. 2017).

    Over the `reliability` bins; accuracy is argmax == target, here agreement with the teacher.
    """
    bins = reliability(probs, targets, n_bins)
    total = sum(b.n for b in bins)
    return float(sum(b.n / total * abs(b.accuracy - b.confidence) for b in bins))


THRESHOLD_TOLERANCE = 1e-6
"""Serving accepts confidence >= threshold - THRESHOLD_TOLERANCE. The threshold is the exact
float64 confidence of one held-out input; the server compares a float32 confidence, whose
rounding (about 3e-8) would otherwise put that input just under its own threshold about half
the time. It does not cover the ONNX-vs-PyTorch logit difference (gate 1e-4, measured 3e-05),
which can move a confidence by a few 1e-5: an input that close to the threshold may escalate
when PyTorch would have answered it, which errs toward the teacher."""


@dataclass(frozen=True)
class CoveragePoint:
    """The cascade at one threshold: the student answers inputs with confidence >= threshold."""

    threshold: float
    accepted: int
    coverage: float
    """accepted / n."""
    kappa: float
    """Cohen's kappa of student vs teacher on the accepted inputs."""
    agreement: float
    """Share of the accepted inputs where the student agrees with the teacher."""
    system_kappa: float
    """Kappa of the cascade's output vs the teacher over all n inputs: the student's label where
    accepted, the teacher's where escalated (so escalated inputs always agree)."""


def _kappa_from_counts(agree: IntArray, chance: IntArray, n: IntArray) -> FloatArray:
    """kappa = (p_o - p_e) / (1 - p_e) from integer counts, with chance = n^2 * p_e.

    Degenerate p_e == 1 is detected on integers and handled as in `metrics.cohen_kappa`.
    """
    nf = n.astype(np.float64)
    p_o = agree / nf
    p_e = chance / (nf * nf)
    degenerate = chance == n * n
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = (p_o - p_e) / (1 - p_e)
    out: FloatArray = np.where(degenerate, np.where(agree == n, 1.0, 0.0), kappa)
    return out


def _cascade_inputs(
    confidence: npt.ArrayLike, teacher: npt.ArrayLike, student: npt.ArrayLike, n_labels: int
) -> tuple[FloatArray, IntArray, IntArray, IntArray]:
    """Validated arrays and the order of the inputs by confidence, highest first (stable)."""
    if n_labels < 1:
        raise ValueError(f"n_labels must be positive, got {n_labels}")
    conf = np.asarray(confidence, dtype=np.float64)
    if conf.ndim != 1 or conf.size < 1:
        raise ValueError(f"confidence must be 1-d with at least one item, got {conf.shape}")
    if not np.isfinite(conf).all():
        raise ValueError("confidence contains NaN or infinite values")
    t = _codes(teacher, conf.size, n_labels, "teacher")
    s = _codes(student, conf.size, n_labels, "student")
    order = np.argsort(-conf, kind="stable").astype(np.int64)
    return conf, t, s, order


@dataclass(frozen=True)
class _CurveCounts:
    """Running counts over the inputs sorted by confidence (highest first), and the cuts a
    threshold can make: the last index of each run of equal confidences."""

    n: int
    confidence: FloatArray
    """Sorted, highest first."""
    rows: IntArray
    """(n, K) running teacher label counts."""
    cols: IntArray
    """(n, K) running student label counts."""
    agree: IntArray
    """Running agreement count."""
    ends: IntArray
    accepted: IntArray
    """ends + 1: inputs accepted at each cut."""

    @property
    def agree_at_cuts(self) -> IntArray:
        return self.agree[self.ends]

    @property
    def chance_at_cuts(self) -> IntArray:
        """n_accepted^2 * p_e at each cut, as an integer."""
        out: IntArray = (self.rows[self.ends] * self.cols[self.ends]).sum(axis=1)
        return out


def _curve_counts(
    confidence: npt.ArrayLike, teacher: npt.ArrayLike, student: npt.ArrayLike, n_labels: int
) -> _CurveCounts:
    conf, t, s, order = _cascade_inputs(confidence, teacher, student, n_labels)
    c_sorted, t_sorted, s_sorted = conf[order], t[order], s[order]
    eye = np.eye(n_labels, dtype=np.int64)
    ends = np.flatnonzero(np.append(c_sorted[1:] != c_sorted[:-1], True)).astype(np.int64)
    return _CurveCounts(
        n=int(conf.size),
        confidence=c_sorted,
        rows=np.cumsum(eye[t_sorted], axis=0),
        cols=np.cumsum(eye[s_sorted], axis=0),
        agree=np.cumsum(t_sorted == s_sorted).astype(np.int64),
        ends=ends,
        accepted=ends + 1,
    )


def count_bounds(
    confidence: npt.ArrayLike,
    teacher: npt.ArrayLike,
    student: npt.ArrayLike,
    n_labels: int,
    method: CountMethod,
    confidence_level: float = 0.95,
) -> FloatArray:
    """The `method`'s one-sided kappa lower bound on the accepted inputs at every point of
    `coverage_curve` (same order), from the agreement count and the plug-in chance term at
    each cut, all in one vectorised call (`binomial.kappa_lower_from_counts`)."""
    c = _curve_counts(confidence, teacher, student, n_labels)
    return kappa_lower_from_counts(
        c.agree_at_cuts, c.chance_at_cuts, c.accepted, method, confidence_level
    )


def coverage_curve(
    confidence: npt.ArrayLike, teacher: npt.ArrayLike, student: npt.ArrayLike, n_labels: int
) -> list[CoveragePoint]:
    """Coverage and agreement at every threshold, from the highest (lowest coverage) down.

    One point per distinct confidence value, with that value as the threshold: inputs with
    equal confidence are accepted or escalated together, since no threshold can separate them.
    The last point accepts everything (coverage 1.0).

    Inputs are sorted by confidence once; teacher and student label counts are cumulated over
    that order (O(n * K)), so the kappa at every cut comes from its running marginals and
    agreement count. For the system kappa the teacher's marginals are the full-sample ones and
    the cascade's are the student's counts on the accepted prefix plus the teacher's on the
    escalated rest.
    """
    c = _curve_counts(confidence, teacher, student, n_labels)
    n, c_sorted, ends, accepted = c.n, c.confidence, c.ends, c.accepted
    agree, rows, cols = c.agree, c.rows, c.cols
    kappa = _kappa_from_counts(c.agree_at_cuts, c.chance_at_cuts, accepted)

    t_total = rows[-1]
    system_cols = cols[ends] + (t_total[None, :] - rows[ends])
    system_agree = agree[ends] + (n - accepted)
    system = _kappa_from_counts(
        system_agree, (t_total[None, :] * system_cols).sum(axis=1), np.full_like(accepted, n)
    )
    return [
        CoveragePoint(
            threshold=float(c_sorted[e]),
            accepted=int(a),
            coverage=float(a / n),
            kappa=float(k),
            agreement=float(agree[e] / a),
            system_kappa=float(sk),
        )
        for e, a, k, sk in zip(ends, accepted, kappa, system, strict=True)
    ]


@dataclass(frozen=True)
class ThresholdChoice:
    """The chosen threshold and the cascade's agreement at it."""

    rule: Rule
    target_kappa: float
    threshold: float
    accepted: int
    coverage: float
    kappa: float
    kappa_low: float | None
    """The rule's one-sided lower bound of kappa on the accepted inputs (None for "point")."""
    system_kappa: float


def choose_threshold(
    confidence: npt.ArrayLike,
    teacher: npt.ArrayLike,
    student: npt.ArrayLike,
    n_labels: int,
    target_kappa: float,
    *,
    rule: Rule = DEFAULT_RULE,
    min_accepted: int = 30,
    confidence_level: float = 0.95,
    n_resamples: int = 1000,
    seed: int = 0,
) -> ThresholdChoice | None:
    """The largest-coverage threshold whose accepted inputs reach `target_kappa`, or None.

    Only points accepting at least `min_accepted` inputs are eligible. Rule "point" requires
    the kappa on the accepted inputs to reach the target; rule "lower" requires its one-sided
    percentile-bootstrap lower bound, the (1 - confidence_level) quantile of `n_resamples`
    kappa replicates over the accepted inputs (the 5th percentile for 0.95), to reach it; rule
    "bca" reads the same replicates at the BCa-adjusted level (`bootstrap.bca_levels`). Rules
    "exact" and "wilson" bound the agreement count instead (`binomial.kappa_lower_from_counts`:
    the one-sided Clopper-Pearson or Wilson bound on p_o, mapped to kappa with the accepted
    inputs' own chance term); they draw no resamples, so `n_resamples` and `seed` do not apply,
    and every candidate's bound comes from one vectorised call.

    Kappa is not monotone in coverage (it can dip and recover as the threshold falls), so the
    scan does not stop at the first failure: it walks from the largest coverage down and
    returns the first point that qualifies, which is the answer by construction and bounds the
    cost in the common case where a high coverage already qualifies. Under rule "lower" a
    candidate whose point kappa misses the target is skipped without a bootstrap. This is a
    cost bound rather than an identity: in rare skewed cases a percentile lower bound can sit
    above the point estimate, but a threshold whose own estimate misses the target is not one
    to serve. Each candidate's resamples come from `resample_indices(accepted, n_resamples,
    seed)`, so the choice is a pure function of the inputs and the seed. Both count bounds sit
    at or below the point estimate, so for them the skip loses nothing.
    """
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; use one of {', '.join(RULES)}")
    if not math.isfinite(target_kappa):
        raise ValueError(f"target_kappa must be finite, got {target_kappa}")
    if min_accepted < 1:
        raise ValueError(f"min_accepted must be positive, got {min_accepted}")
    if rule == "bca" and min_accepted < 2:
        raise ValueError("rule 'bca' needs min_accepted >= 2: its jackknife leaves one item out")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")
    _, t, s, order = _cascade_inputs(confidence, teacher, student, n_labels)
    curve = coverage_curve(confidence, teacher, student, n_labels)
    t_sorted, s_sorted = t[order], s[order]
    counted = (
        count_bounds(confidence, teacher, student, n_labels, COUNT_METHOD[rule], confidence_level)
        if rule in COUNT_METHOD
        else None
    )
    for i in range(len(curve) - 1, -1, -1):
        point = curve[i]
        if point.accepted < min_accepted or point.kappa < target_kappa:
            continue
        low: float | None = None
        if counted is not None:
            low = float(counted[i])
            if low < target_kappa:
                continue
        elif rule != "point":
            k = point.accepted
            method = BOUND_METHOD[rule]
            idx = resample_indices(k, n_resamples, seed)
            low = lower_bounds(
                t_sorted[:k], s_sorted[:k], n_labels, idx, confidence_level, (method,)
            )[method]
            if low < target_kappa:
                continue
        return ThresholdChoice(
            rule=rule,
            target_kappa=target_kappa,
            threshold=point.threshold,
            accepted=point.accepted,
            coverage=point.coverage,
            kappa=point.kappa,
            kappa_low=low,
            system_kappa=point.system_kappa,
        )
    return None


def serving_temperature(logits: ArrayLike2D, targets: npt.ArrayLike) -> tuple[float, bool]:
    """`fit_temperature`, except when the NLL has no minimum inside `TEMPERATURE_BOUNDS`: then
    the fit runs to a bound, and at the lower bound every input, however unlike the training
    data, would score a confidence near 1.0 and never escalate. That happens when the student's
    argmax agrees with every target (there is no disagreement to rescale on) and also with a
    few disagreements whose logits are nearly flat. In both cases T stays 1.0. Returns (T,
    whether T was identified)."""
    z = _logits(logits)
    y = _codes(targets, z.shape[0], z.shape[1], "targets")
    if not (z.argmax(axis=1) != y).any():
        return 1.0, False
    t = fit_temperature(z, y)
    if t in TEMPERATURE_BOUNDS:  # fit_temperature returns a bound exactly when it is clamped
        return 1.0, False
    return t, True


@dataclass(frozen=True)
class Calibration:
    """The fitted temperature, its effect on calibration, and the serving threshold.

    `threshold`, `coverage` and `kappa` are None when no target was asked for or when no
    threshold reaches it; a cascade without a threshold escalates every input to the teacher.
    """

    temperature: float
    n: int
    temperature_identified: bool
    """False when the held-out NLL has no minimum inside the temperature bounds (the student
    agrees with the teacher on every input, or its few errors have nearly flat logits): the
    temperature is then left at 1.0."""
    nll_before: float
    nll_after: float
    ece_before: float
    ece_after: float
    n_bins: int
    target_kappa: float | None
    rule: Rule | None
    threshold: float | None
    coverage: float | None
    kappa: float | None
    kappa_low: float | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate(
    logits: ArrayLike2D,
    teacher: npt.ArrayLike,
    *,
    target_kappa: float | None = 0.9,
    rule: Rule = DEFAULT_RULE,
    n_bins: int = 15,
    min_accepted: int = 30,
    n_resamples: int = 1000,
    seed: int = 0,
) -> Calibration:
    """Fit T on held-out (logits, teacher codes), measure NLL and ECE before (T = 1) and after,
    and choose the cascade threshold on the calibrated confidences.

    Temperature scaling leaves every argmax, and so the student's labels, unchanged, but it
    does change how inputs rank by confidence: the max probability depends on the gaps to all
    the other logits, and dividing them by T reweights those gaps differently per input, so
    two inputs can swap order. The threshold is therefore chosen on max softmax(logits / T),
    and serving must apply the same T before comparing against it. With the same data used
    for T and the threshold the two are fitted jointly on one split; that is the intended use
    (the threshold's bootstrap bound accounts for the sample, not for T's own variance).
    """
    z = _logits(logits)
    y = _codes(teacher, z.shape[0], z.shape[1], "teacher")
    temperature, identified = serving_temperature(z, y)
    before, after = softmax(z), softmax(z, temperature)
    choice = (
        None
        if target_kappa is None
        else choose_threshold(
            after.max(axis=1),
            y,
            z.argmax(axis=1).astype(np.int64),
            z.shape[1],
            target_kappa,
            rule=rule,
            min_accepted=min_accepted,
            n_resamples=n_resamples,
            seed=seed,
        )
    )
    return Calibration(
        temperature=temperature,
        n=int(z.shape[0]),
        temperature_identified=identified,
        nll_before=nll(z, y),
        nll_after=nll(z, y, temperature),
        ece_before=ece(before, y, n_bins),
        ece_after=ece(after, y, n_bins),
        n_bins=n_bins,
        target_kappa=target_kappa,
        rule=None if target_kappa is None else rule,
        threshold=None if choice is None else choice.threshold,
        coverage=None if choice is None else choice.coverage,
        kappa=None if choice is None else choice.kappa,
        kappa_low=None if choice is None else choice.kappa_low,
    )
