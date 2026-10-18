"""Active learning: `select -> label -> train -> eval` rounds under a label and a dollar budget.

A loop grows a labelled training set round by round. `budgets` are cumulative pool picks
(say 100, 200, 400, 800, 1600): round 0 picks its inputs at random, since there is no student
to be uncertain yet, and every later round asks the previous round's student which pool inputs
it wants labelled next (`anvil.select`). Each round trains a fresh student on every label so
far, with the same seed and settings, and scores it against the teacher on the held-out
split. Retraining from scratch costs CPU but keeps the curve comparable: a warm-started
student at 400 labels has seen more updates than a fresh one would have, so its kappa would
mix "more labels" with "more training".

The round-0 batch depends only on the seed, so every strategy run with the same seed starts
from the same student and the curves differ only from round 1 on.

Each round appends one row to `<workspace>/loops/<loop id>/frontier.jsonl`: labels, cost,
kappa with its interval, and how long it took. Picks are recorded in the store before they
are labelled, so an interrupted loop resumes where it stopped (same id, same settings) and
repeats exactly the picks it had made; training is seeded, so a resumed loop gives the same
rows as one that never stopped.

Dollars are label cost, not spend: a label replayed from the cache or a recording counts what
it cost to make, so a replay stops at the same round as the live run it replays. The budget
covers everything the loop labels, the held-out split included (labelled first). A row's
`spent_usd` is what the process that wrote it paid during that round; a resumed round reports
only the resumed process's spend, while its label cost columns are complete either way.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from anvil.errors import ConfigError, StoreError
from anvil.log import get_logger
from anvil.pipeline import Workspace, fit_student, score
from anvil.select import STRATEGIES, Pool, implemented, select_batch
from anvil.spec import TaskSpec
from anvil.store import Split, Store, read_inputs
from anvil.students.encoder import EncoderClassifier
from anvil.teacher.base import BatchAbortedError, Item, LabelBatch, Teacher

log = get_logger(__name__)

FRONTIER = "frontier.jsonl"
STOP_LABELS = "label budget"
STOP_DOLLARS = "dollar budget"
STOP_TARGET = "target kappa"
STOP_POOL = "pool exhausted"


@dataclass(frozen=True)
class LoopConfig:
    strategy: str
    budgets: tuple[int, ...]
    """Cumulative pool inputs picked after each round, strictly increasing."""
    seed: int = 0
    """Seeds the round-0 draw, every selection and the student (init, data order)."""
    max_usd: float | None = None
    """Stop before the label cost of the loop (held-out included) would pass this."""
    target_kappa: float | None = None
    """Stop once the lower end of the held-out kappa interval reaches this."""
    heldout_max: int | None = None
    """Score on at most this many teacher-labelled held-out inputs (default: all)."""
    n_resamples: int = 1000

    def __post_init__(self) -> None:
        if self.strategy not in STRATEGIES:
            raise ConfigError(
                f"unknown strategy {self.strategy!r}; choose one of {', '.join(STRATEGIES)}"
            )
        if not self.budgets or any(b < 1 for b in self.budgets):
            raise ConfigError("budgets must be positive label counts, e.g. 100,200,400")
        if any(b >= c for b, c in zip(self.budgets, self.budgets[1:], strict=False)):
            raise ConfigError(f"budgets must increase strictly, got {list(self.budgets)}")
        if self.max_usd is not None and self.max_usd <= 0:
            raise ConfigError("the dollar budget must be positive")
        if self.target_kappa is not None and not 0 < self.target_kappa <= 1:
            raise ConfigError("the target kappa must be in (0, 1]")
        if self.heldout_max is not None and self.heldout_max < 1:
            raise ConfigError("heldout_max must be at least 1")

    def to_json(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            "budgets": list(self.budgets),
            "seed": self.seed,
            "max_usd": self.max_usd,
            "target_kappa": self.target_kappa,
            "heldout_max": self.heldout_max,
            "n_resamples": self.n_resamples,
        }


@dataclass
class LoopReport:
    loop_id: str
    root: Path
    rows: list[dict[str, object]] = field(default_factory=list)
    stopped: str | None = None
    student: Path | None = None
    """The last round's student (the earlier ones are removed as the loop moves on)."""
    resumed_rows: int = 0
    """Rows an earlier run of the same loop id had already written when this one started."""

    @property
    def frontier(self) -> Path:
        return self.root / FRONTIER


def round_seed(seed: int, round_no: int) -> int:
    """The selection seed of one round: distinct per round, the same on every resume."""
    digest = hashlib.sha256(f"anvil-select:{seed}:{round_no}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def default_loop_id(config: LoopConfig) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}-{config.strategy}-s{config.seed}"


def read_frontier(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_frontier(path: Path, rows: Sequence[dict[str, object]]) -> None:
    """Rewrite the whole file through a temporary one, so a crash never leaves half a row."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


class _Labeller:
    """Labels inputs for one loop and keeps what each one cost.

    An input is sent to the teacher at most once per process, and never when the store
    already holds a label or a billed failure for it: a refused pick stays picked (it used
    budget) and is not paid for again in every later round. A failure that cost nothing (a
    dropped connection, a recording with no row for the input) settles nothing, so the next
    run asks again, as `anvil label` does. A failure replayed from a recording is not stored
    (`Store.append_failures`), so its cost is kept here instead; `cost_of` is complete only
    once every input asked about has been through `label` in this process, which is why the
    loop re-labels its earlier picks (a no-op for settled inputs) before it costs them.
    """

    def __init__(self, store: Store, spec: TaskSpec, teacher: Teacher) -> None:
        self._store = store
        self._spec = spec
        self._teacher = teacher
        self._attempted: set[str] = set()
        self._replayed: dict[str, float] = {}

    def label(self, items: Sequence[Item]) -> float:
        """Label the inputs that still need it; returns what this call spent."""
        model, prompt = self._teacher.model_id, self._spec.prompt_hash
        settled = set(self._store.labels(model, prompt)) | {
            f.input_id for f in self._store.failure_records(model, prompt) if f.cost_usd > 0
        }
        todo = [it for it in items if it.id not in settled and it.id not in self._attempted]
        if not todo:
            return 0.0
        self._attempted.update(it.id for it in todo)

        def persist(batch: LabelBatch) -> None:
            self._store.append_labels(batch.labels)
            self._store.append_failures(batch.failures, model, prompt)
            self._replayed.update({f.input_id: f.cost_usd for f in batch.failures if f.cached})

        try:
            batch = self._teacher.label(todo)
        except BatchAbortedError as exc:
            persist(exc.batch)  # keep every answer that was paid for before the error
            raise exc.cause from None
        persist(batch)
        return batch.spent_usd

    def cost_of(self, input_ids: Sequence[str]) -> float:
        model, prompt = self._teacher.model_id, self._spec.prompt_hash
        replayed = sum(self._replayed.get(i, 0.0) for i in set(input_ids))
        return self._store.cost_of(model, prompt, input_ids) + replayed


def _affordable(cost: float, labelled: int, max_usd: float | None) -> int | None:
    """How many more labels the dollar budget allows at the mean cost per label so far
    (None: no limit, or labels have been free so far)."""
    if max_usd is None or labelled == 0 or cost <= 0:
        return None
    per_label = cost / labelled
    return max(0, math.floor((max_usd - cost) / per_label + 1e-9))


def run_loop(
    spec: TaskSpec,
    ws: Workspace,
    teacher: Teacher,
    config: LoopConfig,
    *,
    loop_id: str | None = None,
    inputs: Path | None = None,
) -> LoopReport:
    """Run (or resume) one loop and return its frontier rows. See the module docstring."""
    if not implemented(config.strategy):
        raise ConfigError(
            f"the {config.strategy} strategy is not implemented yet; run the loop with"
            " --strategy random"
        )
    loop_id = loop_id or default_loop_id(config)
    root = ws.root / "loops" / loop_id
    # The student seed follows the loop seed: training randomness is part of a seed's run.
    spec = spec.model_copy(
        update={"student": spec.student.model_copy(update={"seed": config.seed})}
    )
    model, prompt = teacher.model_id, spec.prompt_hash
    settings = {
        **config.to_json(),
        "teacher_model": model,
        "prompt_hash": prompt,
        "student": spec.student.model_dump(),
        "splits": spec.splits.model_dump(),
    }
    report = LoopReport(loop_id, root)

    with ws.open_store(spec) as store:
        if inputs is not None:
            store.add_inputs(read_inputs(inputs))
        recorded = store.loop_config(loop_id)
        if recorded is None:
            store.create_loop(loop_id, settings)
        elif recorded != (wanted_settings := json.loads(json.dumps(settings, sort_keys=True))):
            changed = sorted(k for k in wanted_settings if recorded.get(k) != wanted_settings[k])
            raise ConfigError(
                f"loop {loop_id!r} was started with other settings ({', '.join(changed)});"
                " resume it with the same ones or start a new loop id"
            )
        root.mkdir(parents=True, exist_ok=True)

        spent = 0.0
        heldout_items = store.inputs(Split.HELDOUT)
        if config.heldout_max is not None:
            heldout_items = heldout_items[: config.heldout_max]
        labeller = _Labeller(store, spec, teacher)
        spent += labeller.label(heldout_items)
        wanted = {it.id for it in heldout_items}
        heldout = [p for p in store.labelled(model, prompt, Split.HELDOUT) if p[0].id in wanted]
        if not heldout:
            raise StoreError("no held-out input got a teacher label; add inputs or check failures")
        heldout_ids = [it.id for it in heldout_items]
        pool_items = store.inputs(Split.POOL)
        if not pool_items:
            raise StoreError("the pool split is empty; add inputs first (--inputs)")
        corpus = [it.text for it in pool_items]

        rows = read_frontier(root / FRONTIER)
        report.rows = rows
        report.resumed_rows = len(rows)
        if rows and rows[-1].get("stop"):
            report.stopped = str(rows[-1]["stop"])
            report.student = root / f"round-{len(rows) - 1}"
            return report

        picks = store.picks(loop_id)
        by_id = {it.id: it for it in pool_items}
        student: EncoderClassifier | None = None
        if rows:
            previous = root / f"round-{len(rows) - 1}"
            if previous.exists():
                student = EncoderClassifier.load(previous)
            for stale in root.glob("round-*"):
                # A crash between a row and the removal of the round before it leaves a
                # student the loop no longer selects with.
                if stale.is_dir() and stale != previous:
                    shutil.rmtree(stale)

        for r in range(len(rows), len(config.budgets)):
            budget = config.budgets[r]
            picked = [i for _, i in picks]
            round_picks = [i for rnd, i in picks if rnd == r]
            select_seconds = 0.0
            confidence: float | None = None

            if not round_picks:
                taken = set(picked)
                candidates = [it for it in pool_items if it.id not in taken]
                want = budget - len(picked)
                # Earlier rounds' picks are settled already; on a resume this call only
                # recovers what their replayed failures cost, so the budget check is complete.
                spent += labeller.label([by_id[i] for i in picked])
                cost = labeller.cost_of([*heldout_ids, *picked])
                allowed = _affordable(cost, len(heldout_ids) + len(picked), config.max_usd)
                if allowed is not None:
                    want = min(want, allowed)
                if not candidates or want <= 0:
                    # Nothing more can be labelled: the previous row is the last one.
                    reason = STOP_POOL if not candidates else STOP_DOLLARS
                    if not rows:
                        raise StoreError(
                            f"{reason} before the first round; raise the budget or add inputs"
                        )
                    rows[-1]["stop"] = reason
                    _write_frontier(root / FRONTIER, rows)
                    report.stopped = reason
                    break
                started = time.perf_counter()
                if r == 0:
                    uniform = np.ones((len(candidates), 1), dtype=np.float32)
                    order = select_batch(Pool(uniform), want, "random", round_seed(config.seed, r))
                else:
                    if student is None:
                        raise StoreError(
                            f"loop {loop_id!r} lost the round {r - 1} student it selects with;"
                            " start a new loop id"
                        )
                    probs, vectors = student.predict_with_embeddings(
                        [it.text for it in candidates]
                    )
                    try:
                        order = select_batch(
                            Pool(probs, vectors), want, config.strategy, round_seed(config.seed, r)
                        )
                    except NotImplementedError as exc:
                        raise ConfigError(f"{exc}; run the loop with --strategy random") from None
                    confidence = float(probs[order].max(axis=1).mean()) if order else None
                select_seconds = time.perf_counter() - started
                round_picks = [candidates[i].id for i in order]
                store.add_picks(loop_id, r, round_picks)
                picks = store.picks(loop_id)
            elif r > 0:
                # Resumed after this round's picks were recorded: score them again with the
                # same forward pass over the same candidates, so the row matches a loop that
                # never stopped.
                if student is None:
                    raise StoreError(
                        f"loop {loop_id!r} lost the round {r - 1} student that scored its"
                        f" round {r} picks; start a new loop id"
                    )
                earlier = {i for rnd, i in picks if rnd < r}
                candidates = [it for it in pool_items if it.id not in earlier]
                probs, _ = student.predict_with_embeddings([it.text for it in candidates])
                position = {it.id: k for k, it in enumerate(candidates)}
                order = [position[i] for i in round_picks]
                confidence = float(probs[order].max(axis=1).mean())

            picked = [i for _, i in picks]
            spent += labeller.label([by_id[i] for i in picked])
            chosen = set(picked)
            pairs = [p for p in store.labelled(model, prompt, Split.POOL) if p[0].id in chosen]
            if not pairs:
                raise StoreError(f"round {r}: the teacher labelled none of the picked inputs")

            out = root / f"round-{r}"
            student, result = fit_student(spec, model, pairs, corpus, root / f"round-{r}.jsonl")
            agreement = score(
                spec, student, heldout, n_resamples=config.n_resamples, seed=config.seed
            )
            if out.exists():
                shutil.rmtree(out)
            student.save(out)

            pool_cost = labeller.cost_of(picked)
            heldout_cost = labeller.cost_of(heldout_ids)
            row: dict[str, object] = {
                "loop_id": loop_id,
                "round": r,
                "strategy": config.strategy,
                "seed": config.seed,
                "budget": budget,
                "picked": len(picked),
                "labels": len(pairs),
                "failures": len(picked) - len(pairs),
                "classes_covered": len(Counter(lab for _, lab in pairs)),
                "heldout": agreement.n,
                "kappa": agreement.kappa,
                "kappa_low": agreement.kappa_ci.low,
                "kappa_high": agreement.kappa_ci.high,
                "accuracy": agreement.accuracy,
                "macro_f1": agreement.macro_f1,
                "selected": len([1 for rnd, _ in picks if rnd == r]),
                "selected_confidence": confidence,
                "label_cost_usd": pool_cost,
                "heldout_cost_usd": heldout_cost,
                "spent_usd": spent,
                "select_seconds": round(select_seconds, 3),
                "train_seconds": round(result.seconds, 3),
                "epochs": result.epochs,
                "best_epoch": result.best_epoch,
                "teacher_model": model,
                "student_model": spec.student.model,
                "stop": None,
            }
            spent = 0.0  # each row reports what its own round paid
            low = agreement.kappa_ci.low
            next_allowed = _affordable(
                pool_cost + heldout_cost, len(heldout_ids) + len(picked), config.max_usd
            )
            # A round cut short (fewer picks than its budget) stopped for the pool or the
            # dollars, whether or not it was the last budget; a full last round is the label
            # budget even when the dollars run out at the same time.
            if config.target_kappa is not None and low >= config.target_kappa:
                row["stop"] = STOP_TARGET
            elif r == len(config.budgets) - 1 and len(picked) >= budget:
                row["stop"] = STOP_LABELS
            elif len(picked) >= len(pool_items):
                row["stop"] = STOP_POOL
            elif next_allowed is not None and next_allowed <= 0:
                row["stop"] = STOP_DOLLARS
            elif r == len(config.budgets) - 1:
                row["stop"] = STOP_LABELS
            rows.append(row)
            _write_frontier(root / FRONTIER, rows)
            # Only once the row is written: until then a resume selects with this student.
            if r > 0 and (root / f"round-{r - 1}").exists():
                shutil.rmtree(root / f"round-{r - 1}")
            log.info(
                "round finished",
                extra={k: row[k] for k in ("loop_id", "round", "labels", "kappa", "stop")},
            )
            if row["stop"]:
                report.stopped = str(row["stop"])
                break

    report.rows = rows
    last = root / f"round-{len(rows) - 1}"
    report.student = last if last.exists() else None
    return report
