"""An LLM judge that checks a parity metric: does the student's answer mean the teacher's?

Execution match (run both SQL queries, compare the rows) is the generation parity metric, and
it can be wrong both ways: on a small table two different queries can return the same rows by
coincidence, and two equivalent queries can differ only in how the output is formatted. The
judge reads the question, the table's columns and both queries and says whether they answer
the question the same way; `judge_vs_metric` then puts its verdicts next to the metric's in a
2x2 table, which is what tells us how far to trust execution match.

The judge is Claude through structured outputs, one request per item, cached and costed like
the teacher (`anvil.teacher.claude`). Judgments a live run pays for are written to a verdict
file and committed, so every later run replays them offline and for free; a changed rubric
changes `rubric_hash()` and a stale file is refused instead of replayed.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from anvil.errors import ConfigError, TeacherError
from anvil.evaluate.metrics import cohen_kappa
from anvil.log import get_logger
from anvil.teacher.cache import DiskCache, request_key
from anvil.teacher.pricing import Usage, cost_usd, price_for
from anvil.teacher.transport import JSONDict, Transport

log = get_logger(__name__)

DEFAULT_MODEL = "claude-haiku-5-5"

JUDGE_INSTRUCTIONS = """You compare two answers to the same question about a database table: a \
reference answer and a candidate answer, each a SQL query.

Decide whether the candidate answers the question the same way the reference does:
- equivalent: for any reasonable contents of the described table, the two queries return the \
same answer, not just by coincidence on some particular rows. Ignore differences in \
formatting, whitespace, column aliases, identifier quoting and keyword case.
- different: there are reasonable table contents on which they return different answers \
(another column, another filter, another aggregate, a missing or extra condition), or the \
candidate is not valid SQL.

Judge the candidate against the reference, not against your own reading of the question. Give \
a one-sentence reason, then the verdict."""

OUTPUT_SCHEMA: JSONDict = {
    "type": "object",
    # `reason` comes first: structured output is generated in property order, so the model
    # writes its reason before it commits to a verdict instead of justifying one after it.
    "properties": {
        "reason": {"type": "string"},
        "verdict": {"type": "string", "enum": ["equivalent", "different"]},
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}

NOT_RECORDED = "no recorded verdict for this item"
"""`Verdict.reason` for an item the verdict file has no row for (never judged), as opposed to
a recorded failure, which keeps the judge's own reason."""


@dataclass(frozen=True)
class JudgeItem:
    id: str
    question: str  # the full student input text (table columns + question)
    reference: str  # the teacher's answer
    candidate: str  # the student's answer


@dataclass(frozen=True)
class Verdict:
    id: str
    equivalent: bool | None  # None: the judge failed (refusal, truncation, invalid output)
    reason: str  # the judge's one-sentence reason, or why it failed
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False

    @property
    def failed(self) -> bool:
        return self.equivalent is None


class Judge(Protocol):
    @property
    def model_id(self) -> str: ...

    def judge(self, items: Sequence[JudgeItem]) -> list[Verdict]: ...


def render(item: JudgeItem) -> str:
    """The user message for one item.

    The reference always comes first. Position bias is deliberately not randomised: the two
    roles are asymmetric (the reference is the teacher, the candidate is judged against it),
    the rubric names them, and a fixed layout keeps the request, and so the cache key, a pure
    function of the item.
    """
    return (
        f"{item.question.strip()}\n\n"
        f"Reference answer:\n{item.reference.strip()}\n\n"
        f"Candidate answer:\n{item.candidate.strip()}"
    )


def item_hash(item: JudgeItem) -> str:
    """What one verdict is about: the rendered question, reference and candidate. The id names
    the question, but the candidate is the student's answer, which changes with every
    retrain, so a recorded verdict replays only for the exact pair it judged."""
    return hashlib.sha256(render(item).encode("utf-8")).hexdigest()[:16]


def rubric_hash() -> str:
    """What makes two verdicts comparable: the rubric, the schema and the message layout.

    The model, effort and token limit are not in it; the model is recorded per row and checked
    separately. 16 hex characters, like the other short hashes in recordings.
    """
    layout = render(JudgeItem("", "{question}", "{reference}", "{candidate}"))
    blob = json.dumps(
        {"instructions": JUDGE_INSTRUCTIONS, "schema": OUTPUT_SCHEMA, "layout": layout},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def spent_usd(verdicts: Iterable[Verdict]) -> float:
    """Dollars actually spent by a run: cached and replayed verdicts cost nothing again."""
    return sum(v.cost_usd for v in verdicts if not v.cached)


class ClaudeJudge:
    """Judges items with Claude through structured outputs.

    As in `ClaudeTeacher`, each item is its own request so the cache is per item; refusals,
    truncation and schema violations become failed verdicts (billed) rather than guesses, and
    a per-request API failure fails that item only. Configuration errors (bad credentials, a
    rejected request) propagate and stop the run; verdicts already received are in the cache,
    so the rerun does not pay for them again.
    """

    def __init__(
        self,
        transport: Transport,
        model: str = DEFAULT_MODEL,
        effort: Literal["low", "medium", "high"] = "low",
        max_tokens: int = 2048,
        cache: DiskCache | None = None,
        workers: int = 4,
    ) -> None:
        price_for(model)  # fail fast on a model we cannot account for
        self._transport = transport
        self._model = model
        self._effort = effort
        self._max_tokens = max_tokens
        self._cache = cache
        self._workers = workers

    @property
    def model_id(self) -> str:
        return self._model

    def request_params(self, item: JudgeItem) -> JSONDict:
        return {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "system": JUDGE_INSTRUCTIONS,
            "messages": [{"role": "user", "content": render(item)}],
            "output_config": {
                "effort": self._effort,
                "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
            },
        }

    def judge(self, items: Sequence[JudgeItem]) -> list[Verdict]:
        """One verdict per item, in input order. Items with identical requests are sent once
        and share the answer (the copies are `cached=True`; a shared failure is billed on the
        first item only)."""
        if not items:
            return []
        params = [self.request_params(item) for item in items]
        keys = [request_key(p) for p in params]
        first: dict[str, int] = {}
        for i, key in enumerate(keys):
            first.setdefault(key, i)
        with ThreadPoolExecutor(max_workers=self._workers) as pool:
            futures = {key: pool.submit(self._fetch, params[i], key) for key, i in first.items()}
            try:
                verdicts = [
                    self._outcome(item, futures[keys[i]].result(), keys[i], first[keys[i]] != i)
                    for i, item in enumerate(items)
                ]
            except BaseException:
                for future in futures.values():
                    future.cancel()
                raise
        log.info(
            "judge batch",
            extra={
                "model": self.model_id,
                "judged": sum(not v.failed for v in verdicts),
                "failed": sum(v.failed for v in verdicts),
                "cache_hits": sum(v.cached for v in verdicts),
                "spent_usd": round(spent_usd(verdicts), 6),
            },
        )
        return verdicts

    def _fetch(self, params: JSONDict, key: str) -> tuple[JSONDict, bool] | TeacherError:
        cached = self._cache.get(key) if self._cache else None
        if cached is not None:
            return cached, True
        try:
            return self._transport(params), False
        except TeacherError as exc:
            return exc

    def _outcome(
        self,
        item: JudgeItem,
        fetched: tuple[JSONDict, bool] | TeacherError,
        key: str,
        shared: bool,
    ) -> Verdict:
        if isinstance(fetched, TeacherError):
            return Verdict(item.id, None, str(fetched))
        response, cached = fetched
        verdict = self._parse(item, response, cached=cached or shared)
        if shared and verdict.failed:
            verdict = Verdict(item.id, None, verdict.reason)  # billed on the first copy
        # Only valid verdicts are cached, so a refusal or truncation is retried next run.
        if not cached and not shared and self._cache is not None and not verdict.failed:
            self._cache.put(key, response)
        return verdict

    def _parse(self, item: JudgeItem, response: JSONDict, cached: bool) -> Verdict:
        # Mirrors ClaudeTeacher._parse; kept separate because that method is bound to a spec.
        usage = Usage.from_response(response.get("usage") or {})
        cost = cost_usd(self.model_id, usage)

        def verdict(equivalent: bool | None, reason: str) -> Verdict:
            return Verdict(
                item.id,
                equivalent,
                reason,
                usage.input_tokens,
                usage.output_tokens,
                cost,
                cached,
            )

        stop = response.get("stop_reason")
        if stop == "refusal":
            details = response.get("stop_details") or {}
            return verdict(None, f"refusal ({details.get('category') or 'no category'})")
        if stop == "max_tokens":
            return verdict(None, "hit max_tokens before answering (thinking counts); raise it")
        text = "".join(
            block.get("text", "")
            for block in response.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        try:
            answer = json.loads(text)
        except json.JSONDecodeError:
            return verdict(None, f"output is not JSON: {text[:80]!r}")
        if not isinstance(answer, dict):
            return verdict(None, f"output is not an object: {text[:80]!r}")
        choice, reason = answer.get("verdict"), answer.get("reason")
        if choice not in ("equivalent", "different"):
            return verdict(None, f"verdict {choice!r} is not equivalent or different")
        if not isinstance(reason, str):
            return verdict(None, "output has no `reason` string")
        return verdict(choice == "equivalent", reason)


# --- Recorded verdicts ---------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedVerdict:
    """One row of a verdict file; `equivalent` is None for a recorded judge failure. `item` is
    the `item_hash` of the pair judged."""

    id: str
    item: str
    equivalent: bool | None
    reason: str
    model: str
    rubric_hash: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        for name in ("id", "item", "reason", "model", "rubric_hash"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")
        if self.equivalent is not None and not isinstance(self.equivalent, bool):
            raise ValueError("equivalent must be true, false or null")


def write_verdict_file(
    path: Path,
    items: Sequence[JudgeItem],
    verdicts: Sequence[Verdict],
    model: str,
    rubric_hash: str,
) -> int:
    """Merge this run's verdicts (one per item, in order) into the file as JSONL sorted by id
    and item, atomically. Returns the row count.

    Rows are keyed by (id, `item_hash`), so verdicts on another student's answers to the same
    questions are kept beside these, not overwritten; rows made under another model or rubric
    are dropped, since they can no longer be replayed. A recorded failure (refusal,
    truncation, invalid output) is kept so a replay sees what the live run saw. A verdict that
    never reached the judge (`NOT_RECORDED`, or a transport failure, which has no tokens) is
    not a judgment and is left out, so the next live run asks again. A valid verdict for a
    pair supersedes a failure for it.
    """
    if len(items) != len(verdicts):
        raise ValueError(f"{len(items)} items but {len(verdicts)} verdicts")
    rows: dict[tuple[str, str], RecordedVerdict] = {}
    if path.exists():
        for old in read_verdict_file(path):
            if old.model == model and old.rubric_hash == rubric_hash:
                rows[(old.id, old.item)] = old
    for it, v in zip(items, verdicts, strict=True):
        if v.id != it.id:
            raise ValueError(f"verdict {v.id!r} is not for item {it.id!r}")
        if v.reason == NOT_RECORDED or (v.failed and v.input_tokens == 0):
            continue
        key = (v.id, item_hash(it))
        if v.failed and key in rows and rows[key].equivalent is not None:
            continue
        rows[key] = RecordedVerdict(
            v.id,
            key[1],
            v.equivalent,
            v.reason,
            model,
            rubric_hash,
            v.input_tokens,
            v.output_tokens,
            v.cost_usd,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for key in sorted(rows):
                fh.write(json.dumps(asdict(rows[key]), ensure_ascii=False, sort_keys=True))
                fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return len(rows)


def read_verdict_file(path: Path) -> list[RecordedVerdict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise ConfigError(f"verdict file {path} not found") from exc
    rows: list[RecordedVerdict] = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("expected a JSON object")
            rows.append(RecordedVerdict(**obj))
        except (ValueError, TypeError) as exc:  # JSONDecodeError is a ValueError
            raise ConfigError(
                f"{path}:{n} is not a recorded verdict: {exc}; fix or delete the row"
                " (a deleted row is judged again by the next live run)"
            ) from exc
    return rows


class VerdictFileJudge:
    """Answers from recorded verdicts made by `model_id` under the current rubric.

    Verdicts come back `cached=True` with their original tokens and cost, so a replay spends
    nothing while the report still knows what judging cost. An item with no row for its exact
    pair (a new question, or a retrained student's new answer) is a failed verdict with reason
    `NOT_RECORDED`, never a guess.
    """

    def __init__(self, rows: Sequence[RecordedVerdict], model_id: str = DEFAULT_MODEL) -> None:
        self._model = model_id
        expected = rubric_hash()
        self._rows: dict[tuple[str, str], RecordedVerdict] = {}
        for row in rows:
            if row.model != model_id or row.rubric_hash != expected:
                raise ConfigError(
                    f"recorded verdict for {row.id!r} is from ({row.model}, {row.rubric_hash}),"
                    f" expected ({model_id}, {expected}); the judge rubric or model changed"
                    " since recording, so re-judge with the live judge"
                )
            self._rows[(row.id, row.item)] = row

    @classmethod
    def from_file(cls, path: Path, model_id: str = DEFAULT_MODEL) -> VerdictFileJudge:
        return cls(read_verdict_file(path), model_id)

    @property
    def model_id(self) -> str:
        return self._model

    def __len__(self) -> int:
        return len(self._rows)

    def judge(self, items: Sequence[JudgeItem]) -> list[Verdict]:
        out: list[Verdict] = []
        for item in items:
            row = self._rows.get((item.id, item_hash(item)))
            if row is None:
                out.append(Verdict(item.id, None, NOT_RECORDED))
                continue
            out.append(
                Verdict(
                    item.id,
                    row.equivalent,
                    row.reason,
                    row.input_tokens,
                    row.output_tokens,
                    row.cost_usd,
                    cached=True,
                )
            )
        return out


class RecordedThenLiveJudge:
    """Answers from recorded verdicts first and sends only the pairs without a row to a live
    judge of the same model. A recorded failure is returned as recorded, not asked again;
    delete its row to pay for a retry (mirrors `RecordedThenLiveTeacher`)."""

    def __init__(self, recorded: VerdictFileJudge, live: Judge) -> None:
        if recorded.model_id != live.model_id:
            raise ConfigError(
                f"recorded verdicts are from {recorded.model_id}, the live judge is"
                f" {live.model_id}; use the same judge model or re-judge"
            )
        self._recorded = recorded
        self._live = live

    @property
    def model_id(self) -> str:
        return self._live.model_id

    def judge(self, items: Sequence[JudgeItem]) -> list[Verdict]:
        out = self._recorded.judge(items)
        todo = [i for i, v in enumerate(out) if v.reason == NOT_RECORDED and v.failed]
        if not todo:
            return out
        fresh = self._live.judge([items[i] for i in todo])
        for i, v in zip(todo, fresh, strict=True):
            out[i] = v
        return out


# --- Agreement with a metric ---------------------------------------------------------------


@dataclass(frozen=True)
class JudgeAgreement:
    """The judge's "equivalent" against a metric's "match" over the judged items.

    `both_match`: equivalent and matched; `judge_only`: equivalent, metric says mismatch (the
    metric is too strict, e.g. output formatting); `exec_only`: matched, judge says different
    (a coincidental match on a small table); `neither`: different and mismatched.
    `equivalent_rate` is NaN and `kappa` None when nothing was judged.
    """

    n: int
    judged: int
    failed: int
    equivalent_rate: float
    both_match: int
    judge_only: int
    exec_only: int
    neither: int
    kappa: float | None


def judge_vs_metric(verdicts: Sequence[Verdict], metric: Mapping[str, bool]) -> JudgeAgreement:
    """Cross-tabulate verdicts with a per-id binary metric; failed verdicts are excluded and
    counted. Every judged id must have a metric value."""
    judged = [v for v in verdicts if not v.failed]
    missing = [v.id for v in judged if v.id not in metric]
    if missing:
        raise ValueError(f"no metric value for {len(missing)} judged ids, e.g. {missing[:3]}")
    pairs = [(bool(v.equivalent), metric[v.id]) for v in judged]
    kappa = (
        cohen_kappa([str(j) for j, _ in pairs], [str(m) for _, m in pairs])
        if len(pairs) >= 2
        else None
    )
    return JudgeAgreement(
        n=len(verdicts),
        judged=len(judged),
        failed=len(verdicts) - len(judged),
        equivalent_rate=sum(j for j, _ in pairs) / len(pairs) if pairs else math.nan,
        both_match=sum(j and m for j, m in pairs),
        judge_only=sum(j and not m for j, m in pairs),
        exec_only=sum(m and not j for j, m in pairs),
        neither=sum(not j and not m for j, m in pairs),
        kappa=kappa,
    )


def judge_sample(ids: Sequence[str], n: int, seed: int) -> list[str]:
    """A seeded draw of `n` ids without replacement, in input order; all ids if n >= len."""
    if n < 0:
        raise ValueError(f"sample size must be non-negative, got {n}")
    if n >= len(ids):
        return list(ids)
    picked = np.random.default_rng(seed).permutation(len(ids))[:n]
    return [ids[i] for i in sorted(int(i) for i in picked)]
