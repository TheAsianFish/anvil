"""`make eval-coverage`: do the bounds the cascade's threshold rule can rely on cover?

`choose_threshold` serves the largest coverage whose one-sided 95% bootstrap lower bound of
kappa clears the target, so the rule is only as honest as that bound. Two simulations, both
offline and seeded, answer whether it is, at the kappa, item count and label count where the
rule applies it:

1. Fixed sample (`fixed`): n items from a population whose kappa is known exactly. The teacher
   draws a label uniformly from K; the student copies it with probability kappa and otherwise
   draws its own label independently from the same distribution, so both raters share the
   marginals and the population kappa is exactly the copy probability (p_o = kappa + (1 -
   kappa) / K, p_e = 1 / K). Each replication computes the percentile and BCa bounds from the
   same 1,000 resamples `choose_threshold` would draw, and the exact (Clopper-Pearson) and
   Wilson bounds from the sample's agreement count and plug-in chance term; "covers" means the
   one-sided 95% lower bound is at or below the population kappa (and, for the two-sided 95%
   bootstrap interval, that the interval contains it). Also: the point estimate's bias and
   spread against the bootstrap's. The count bounds draw nothing from the generator, so the
   bootstrap columns are the same samples, resamples and counts as before they were added.
2. Threshold search (`search`): n held-out inputs with a confidence c uniform on [0, 1] and an
   agreement probability 1 - BETA * (1 - c) ** GAMMA (the student copies the teacher with that
   probability, else draws independently as above). The population kappa of the inputs above
   threshold t is then the mean copy probability over c >= t, 1 - BETA * u ** GAMMA /
   (GAMMA + 1) for coverage u = 1 - t, which with BETA = 0.8 and GAMMA = 4 tracks the
   Banking77 gold student's committed coverage curve (0.84 on everything, 0.93 at 80%, 0.99 at
   50%). Each replication runs the threshold search for every rule (`point`, `lower`, `bca`,
   `exact`, `wilson`) on one sample and checks the population kappa at the chosen threshold
   against the target: the selection effect of picking the best of many thresholds, which a
   fixed-threshold coverage study misses.

3. Pointwise (`pointwise`): no sampling at all. Both count bounds bound p_o, the agreement
   rate, before the chance term maps it to kappa; for m answered inputs and every agreement
   rate p on a fine grid over the range a cascade answers at, the probability that the bound
   is at or below p is summed exactly over the binomial distribution of the count. The
   Clopper-Pearson bound is at least 95% at every p by construction; the Wilson bound is not,
   and this says by how much and how often it falls short.

The search shares one bootstrap per candidate threshold between the two bootstrap rules, and
computes both count bounds for every candidate in one vectorised call (`search_all`); a test
checks it picks exactly what `choose_threshold` picks for each rule.
Cells run in parallel worker processes; each has its own seed stream (SeedSequence keyed by
the cell), so the report is the same for any number of workers.

Raw output: `eval/results/coverage-simulation/` (`report.json`, `meta.json`). `--check`
compares the counts of a fresh run with the committed report instead of writing it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt

from anvil import __version__
from anvil.calibrate import (
    BOUND_METHOD,
    COUNT_METHOD,
    RULES,
    CoveragePoint,
    Rule,
    count_bounds,
    coverage_curve,
)
from anvil.evaluate.binomial import (
    COUNT_BOUNDS,
    CountMethod,
    binomial_pmf,
    kappa_lower_from_counts,
)
from anvil.evaluate.bootstrap import (
    bca_levels,
    kappa_jackknife,
    kappa_replicates,
    lower_bounds,
    resample_indices,
)
from eval.render import SMOKE_SUFFIX

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval" / "results" / "coverage-simulation"

KAPPAS = (0.85, 0.9, 0.95)
SEARCH_TARGETS = (0.9, 0.95)
"""At target 0.85 the population's oracle threshold accepts 98% of inputs, so there is almost
nothing to search; the fixed-sample part still covers kappa 0.85."""
SIZES = (100, 250, 500)
LABELS = (4, 77)
CONFIDENCE = 0.95
N_RESAMPLES = 1000
"""What `choose_threshold` draws per candidate, so the simulation measures the served rule."""
MIN_ACCEPTED = 30
BOOTSTRAP_METHODS = ("percentile", "bca")
COUNT_METHODS: tuple[CountMethod, ...] = ("exact", "wilson")
METHODS = BOOTSTRAP_METHODS + COUNT_METHODS
BETA = 0.8
GAMMA = 4.0
POINTWISE_SIZES = (30, 100, 250, 500)
"""30 is `min_accepted`, the fewest inputs a threshold may answer."""
POINTWISE_RATES = (0.8, 0.999, 2000)
"""Agreement rates from 0.8 to 0.999 (2,000 points): kappa 0.85 at 4 labels is p_o 0.89, and a
cascade at kappa 0.95 with 77 labels answers where p_o is above 0.95."""
DEFAULT_FIXED_REPS = 500
DEFAULT_SEARCH_REPS = 30
DEFAULT_SEED = 0

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
Part = Literal["fixed", "search"]


def copy_rater(
    rng: np.random.Generator, teacher: IntArray, k: int, copy: FloatArray | float
) -> IntArray:
    """The student: the teacher's label with probability `copy` (per item or one value), else
    an independent uniform draw from the same K labels (which may coincide with it)."""
    other = rng.integers(0, k, size=teacher.size, dtype=np.int64)
    out: IntArray = np.where(rng.random(teacher.size) < copy, teacher, other)
    return out


def population_kappa(coverage: float) -> float:
    """Kappa of the search population on its most confident `coverage` share of inputs."""
    return float(1 - BETA * coverage**GAMMA / (GAMMA + 1))


def oracle_coverage(target: float) -> float:
    """The largest coverage whose population kappa reaches `target` (1.0 if all of it does)."""
    return float(min(1.0, ((1 - target) * (GAMMA + 1) / BETA) ** (1 / GAMMA)))


def _cell_rng(seed: int, part: Part, index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(part == "search", index)))


@dataclass(frozen=True)
class Cell:
    part: Part
    index: int
    kappa: float
    n: int
    labels: int
    reps: int
    seed: int


def fixed_cell(cell: Cell) -> dict[str, object]:
    """Coverage of the percentile, BCa, exact and Wilson bounds at one (kappa, n, K)."""
    rng = _cell_rng(cell.seed, "fixed", cell.index)
    n, k, kappa = cell.n, cell.labels, cell.kappa
    full = np.arange(n, dtype=np.int64)[None, :]
    alpha = 1 - CONFIDENCE
    levels = np.array([alpha, alpha / 2, 1 - alpha / 2])
    estimates = np.empty(cell.reps)
    boot_sd = np.empty(cell.reps)
    bca_level = np.empty(cell.reps)
    lows = np.empty((cell.reps, len(METHODS)))
    agree = np.empty(cell.reps, dtype=np.int64)
    chance = np.empty(cell.reps, dtype=np.int64)
    two_sided = np.zeros(2, dtype=np.int64)
    for rep in range(cell.reps):
        teacher = rng.integers(0, k, size=n, dtype=np.int64)
        student = copy_rater(rng, teacher, k, kappa)
        boot_seed = int(rng.integers(2**32))
        values = kappa_replicates(teacher, student, k, resample_indices(n, N_RESAMPLES, boot_seed))
        estimate = float(kappa_replicates(teacher, student, k, full)[0])
        adjusted = bca_levels(values, estimate, kappa_jackknife(teacher, student, k), levels)
        # One read of the replicates for both methods: percentile levels, then BCa's.
        ends = np.percentile(values, 100 * np.concatenate([levels, adjusted])).reshape(2, 3)
        estimates[rep] = estimate
        boot_sd[rep] = float(values.std())
        bca_level[rep] = adjusted[0]
        lows[rep, :2] = ends[:, 0]
        two_sided += (ends[:, 1] <= kappa) & (kappa <= ends[:, 2])
        agree[rep] = int(np.count_nonzero(teacher == student))
        rows, cols = np.bincount(teacher, minlength=k), np.bincount(student, minlength=k)
        chance[rep] = int((rows * cols).sum())
    for i, method in enumerate(COUNT_METHODS, start=2):
        lows[:, i] = kappa_lower_from_counts(agree, chance, n, method, CONFIDENCE)
    one_sided = (lows <= kappa).sum(axis=0)
    sd = float(estimates.std(ddof=1))
    methods = METHODS
    return {
        "kappa": kappa,
        "n": n,
        "labels": k,
        "reps": cell.reps,
        "lower_covered": {m: int(one_sided[i]) for i, m in enumerate(methods)},
        "interval_covered": {m: int(two_sided[i]) for i, m in enumerate(BOOTSTRAP_METHODS)},
        "mean_lower": {m: float(lows[:, i].mean()) for i, m in enumerate(methods)},
        "mean_bca_level": float(bca_level.mean()),
        "mean_estimate": float(estimates.mean()),
        "bias": float(estimates.mean() - kappa),
        "bias_se": sd / math.sqrt(cell.reps),
        "sd_estimate": sd,
        "mean_bootstrap_sd": float(boot_sd.mean()),
    }


def search_all(
    confidence: FloatArray,
    teacher: IntArray,
    student: IntArray,
    n_labels: int,
    target: float,
    *,
    seed: int,
    min_accepted: int = MIN_ACCEPTED,
    n_resamples: int = N_RESAMPLES,
    confidence_level: float = CONFIDENCE,
) -> dict[Rule, CoveragePoint | None]:
    """The point `choose_threshold` would choose under every rule, from one scan: the same
    candidates in the same order with the same resamples, but each candidate's replicates are
    computed once and read by both bootstrap rules, and the count rules' bounds come from one
    vectorised call per rule over the whole curve."""
    curve = coverage_curve(confidence, teacher, student, n_labels)
    order = np.argsort(-confidence, kind="stable")
    t_sorted, s_sorted = teacher[order], student[order]
    chosen: dict[Rule, CoveragePoint | None] = {rule: None for rule in RULES}
    eligible = [
        i
        for i in range(len(curve) - 1, -1, -1)
        if curve[i].accepted >= min_accepted and curve[i].kappa >= target
    ]
    if eligible:
        chosen["point"] = curve[eligible[0]]
    for rule, method in COUNT_METHOD.items():
        bound = count_bounds(confidence, teacher, student, n_labels, method, confidence_level)
        chosen[rule] = next((curve[i] for i in eligible if bound[i] >= target), None)
    boot_rules = tuple(BOUND_METHOD)
    methods = tuple(BOUND_METHOD[r] for r in boot_rules)
    for i in eligible:
        m = curve[i].accepted
        idx = resample_indices(m, n_resamples, seed)
        bounds = lower_bounds(t_sorted[:m], s_sorted[:m], n_labels, idx, confidence_level, methods)
        for rule in boot_rules:
            if chosen[rule] is None and bounds[BOUND_METHOD[rule]] >= target:
                chosen[rule] = curve[i]
        if all(chosen[r] is not None for r in boot_rules):
            break
    return chosen


def search_cell(cell: Cell) -> dict[str, object]:
    """How often each rule's chosen threshold truly keeps the target, at one (target, n, K)."""
    rng = _cell_rng(cell.seed, "search", cell.index)
    n, k, target = cell.n, cell.labels, cell.kappa
    oracle = oracle_coverage(target)
    rows: dict[Rule, list[tuple[float, float]]] = {rule: [] for rule in RULES}
    for _ in range(cell.reps):
        conf = rng.random(n)
        teacher = rng.integers(0, k, size=n, dtype=np.int64)
        student = copy_rater(rng, teacher, k, 1 - BETA * (1 - conf) ** GAMMA)
        chosen = search_all(conf, teacher, student, k, target, seed=int(rng.integers(2**32)))
        for rule, point in chosen.items():
            if point is not None:
                true_coverage = 1 - point.threshold
                rows[rule].append((true_coverage, population_kappa(true_coverage)))
    out: dict[str, object] = {
        "target": target,
        "n": n,
        "labels": k,
        "reps": cell.reps,
        "oracle_coverage": oracle,
    }
    for rule, found in rows.items():
        cov = np.array([c for c, _ in found])
        kap = np.array([q for _, q in found])
        met = kap >= target
        out[rule] = {
            "found": len(found),
            "met": int(met.sum()),
            "mean_true_coverage": float(cov.mean()) if found else None,
            "mean_coverage_given_up": float((oracle - cov).mean()) if found else None,
            "mean_true_kappa": float(kap.mean()) if found else None,
            "mean_shortfall_when_missed": float((target - kap[~met]).mean())
            if (~met).any()
            else None,
        }
    return out


def pointwise() -> list[dict[str, object]]:
    """Exact coverage of p_o by each count bound at every agreement rate of the grid."""
    rates = np.linspace(*POINTWISE_RATES[:2], int(POINTWISE_RATES[2]))
    out: list[dict[str, object]] = []
    for m in POINTWISE_SIZES:
        pmf = binomial_pmf(m, rates)
        counts = np.arange(m + 1, dtype=np.int64)
        for method in COUNT_METHODS:
            bound = COUNT_BOUNDS[method](counts, np.full(m + 1, m), CONFIDENCE)
            coverage = (pmf * (bound[None, :] <= rates[:, None])).sum(axis=1)
            worst = int(coverage.argmin())
            out.append(
                {
                    "n": m,
                    "method": method,
                    "min": float(coverage.min()),
                    "rate_at_min": float(rates[worst]),
                    "mean": float(coverage.mean()),
                    "share_below_nominal": float((coverage < CONFIDENCE - 1e-12).mean()),
                }
            )
    return out


def _run_cell(cell: Cell) -> dict[str, object]:
    return fixed_cell(cell) if cell.part == "fixed" else search_cell(cell)


def cells(fixed_reps: int, search_reps: int, seed: int) -> list[Cell]:
    fixed = [(kappa, n, k) for k in LABELS for n in SIZES for kappa in KAPPAS]
    search = [(target, n, k) for k in LABELS for n in SIZES for target in SEARCH_TARGETS]
    out = [Cell("fixed", i, *g, fixed_reps, seed) for i, g in enumerate(fixed)]
    out += [Cell("search", i, *g, search_reps, seed) for i, g in enumerate(search)]
    # Longest first, so the pool is not left waiting on one large cell at the end.
    return sorted(out, key=lambda c: -(c.n * c.reps * (40 if c.part == "search" else 1)))


def simulate(fixed_reps: int, search_reps: int, seed: int, workers: int) -> dict[str, object]:
    todo = cells(fixed_reps, search_reps, seed)
    if workers <= 1:
        results = [_run_cell(c) for c in todo]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_cell, todo))
    done = sorted(zip(todo, results, strict=True), key=lambda cr: (cr[0].part, cr[0].index))
    return {
        "config": {
            "kappas": list(KAPPAS),
            "search_targets": list(SEARCH_TARGETS),
            "sizes": list(SIZES),
            "labels": list(LABELS),
            "confidence": CONFIDENCE,
            "n_resamples": N_RESAMPLES,
            "min_accepted": MIN_ACCEPTED,
            "fixed_reps": fixed_reps,
            "search_reps": search_reps,
            "seed": seed,
            "pointwise_sizes": list(POINTWISE_SIZES),
            "pointwise_rates": {
                "low": POINTWISE_RATES[0],
                "high": POINTWISE_RATES[1],
                "points": int(POINTWISE_RATES[2]),
            },
            "search_population": {
                "beta": BETA,
                "gamma": GAMMA,
                "kappa_at_coverage": {
                    str(u): population_kappa(u) for u in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
                },
            },
        },
        "fixed": [r for c, r in done if c.part == "fixed"],
        "search": [r for c, r in done if c.part == "search"],
        "pointwise": pointwise(),
    }


def counts(report: dict[str, object]) -> list[int]:
    """The integer outcomes of a report in a fixed order (bounds that covered, thresholds
    found and met), for `--check`."""
    fixed = report["fixed"]
    search = report["search"]
    assert isinstance(fixed, list)
    assert isinstance(search, list)
    out: list[int] = []
    for r in fixed:
        out += [r["lower_covered"][m] for m in METHODS]
        out += [r["interval_covered"][m] for m in BOOTSTRAP_METHODS]
    for r in search:
        out += [r[rule][key] for rule in RULES for key in ("found", "met")]
    return out


CHECK_TOLERANCE = 1
"""`--check` accepts a count that differs by one: another CPU may sum in another order, and a
bound that lands within an ulp of the population kappa can then flip one replication. A
different simulation moves many counts by more."""


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.coverage", description=__doc__)
    parser.add_argument("--fixed-reps", type=int, default=DEFAULT_FIXED_REPS)
    parser.add_argument("--search-reps", type=int, default=DEFAULT_SEARCH_REPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--out", type=Path, default=None, help="results directory")
    parser.add_argument(
        "--check",
        action="store_true",
        help="re-run and compare the counts with the committed report; write nothing",
    )
    args = parser.parse_args(argv)
    if args.fixed_reps < 1 or args.search_reps < 1:
        parser.error("--fixed-reps and --search-reps must be positive")
    smoke = (
        args.fixed_reps != DEFAULT_FIXED_REPS
        or args.search_reps != DEFAULT_SEARCH_REPS
        or args.seed != DEFAULT_SEED
    )
    out = args.out or OUT.with_name(OUT.name + (SMOKE_SUFFIX if smoke else ""))
    provenance = {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }
    started = time.perf_counter()
    report = simulate(args.fixed_reps, args.search_reps, args.seed, args.workers)
    wall = time.perf_counter() - started
    if args.check:
        committed = counts(json.loads((out / "report.json").read_text(encoding="utf-8")))
        fresh = counts(report)
        if len(committed) != len(fresh):
            print(f"error: {out / 'report.json'} holds another grid", file=sys.stderr)  # noqa: T201
            return 1
        diffs = [abs(a - b) for a, b in zip(committed, fresh, strict=True)]
        ok = max(diffs, default=0) <= CHECK_TOLERANCE
        print(  # noqa: T201 - the final summary
            f"coverage simulation in {wall:.1f} s: {sum(d > 0 for d in diffs)} of {len(diffs)}"
            f" counts differ from {out / 'report.json'} (largest difference"
            f" {max(diffs, default=0)}, tolerance {CHECK_TOLERANCE})"
        )
        return 0 if ok else 1
    out.mkdir(parents=True, exist_ok=True)
    command = "python -m eval.coverage " + " ".join(argv if argv is not None else sys.argv[1:])
    meta = {
        "command": command.strip(),
        **provenance,
        "anvil": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "workers": args.workers,
        "seed": args.seed,
        "teacher_model": "n/a",
        "wall_seconds": round(wall, 1),
        "spent_usd": 0.0,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    print(f"coverage simulation in {wall:.1f} s; raw output {out}")  # noqa: T201 - the summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
