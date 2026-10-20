"""Execution match for text-to-SQL: run the teacher's and the student's SQL on the example's
own table and compare the results.

Two queries that differ as strings often mean the same thing (`COUNT("Player")` against
`COUNT(*)`, a reordered `AND`, different quoting), so exact string match understates
agreement; execution match is WikiSQL's headline metric for that reason. It overstates it
in the other direction (two different queries can happen to return the same rows on one
small table, most often both returning nothing), which is why `ExecMatch.reference_empty`
is reported next to it and `normalise_sql` gives a strict string comparison too.

Each comparison runs in a fresh in-memory SQLite database holding one table, `t`, built
from the WikiSQL layout (header, column types, string cells). The SQL comes from a model,
so the connection is locked down before any of it runs:

- `PRAGMA query_only = ON` refuses writes at the engine level;
- an authorizer allows only the actions a read-only query needs (SQLITE_SELECT,
  SQLITE_READ, SQLITE_FUNCTION and SQLITE_RECURSIVE) and denies everything else: ATTACH,
  PRAGMA, every write and DDL, temp tables and triggers. `load_extension` is denied by name
  as well, on top of extension loading being off by default on a new connection;
- a progress handler aborts a query after about `max_steps` VM instructions, so a
  `WITH RECURSIVE` loop or a cross join of the table with itself many times over cannot
  hang an eval, and a result longer than `max_rows` rows is refused;
- the longest string or blob a query may build is capped at a few kilobytes (`zeroblob(1e9)`
  would otherwise allocate a gigabyte, and the built-ins that are quadratic in their
  arguments, `ltrim(X, Y)` or `instr`, run inside one VM instruction where the step limit
  cannot interrupt them: at 10 KB the worst case is milliseconds), and the bytes a result
  may hold in total are capped too, since the row cap alone would still let ten thousand
  rows of large blobs exhaust memory.

Bad SQL never raises: `execute` returns an `Execution` that carries the error instead, and
a candidate (or a reference) that errors never matches.
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from dataclasses import dataclass

TABLE_NAME = "t"
"""The name of the one table in every database; the prompt tells the model to query `t`."""

_TYPES = ("text", "real")

# The read-only query surface. SQLITE_RECURSIVE is reported for every `WITH RECURSIVE` CTE
# (as its own action, not as a SELECT); it is allowed because the step limit, not the
# authorizer, is what keeps a runaway recursion from hanging an eval.
_ALLOWED_ACTIONS = frozenset(
    {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
)
# `fts3_tokenizer` takes and returns raw pointers (its two-argument form registers one), so it
# is refused even though the virtual tables that would use it are already denied.
_DENIED_FUNCTIONS = frozenset({"load_extension", "fts3_tokenizer"})

# How often (in VM instructions) the progress handler runs. Small enough that the step
# limit overshoots by at most this much, large enough that the callback costs nothing.
_PROGRESS_EVERY = 1000
# Longest string or blob a query may build, in bytes. WikiSQL cells are short (the longest is
# well under a kilobyte); this stops a query from allocating memory without bound and keeps
# the quadratic built-ins fast.
_MAX_LENGTH = 10_000
# Most bytes a result may hold across all its rows, in str and bytes values.
_MAX_RESULT_BYTES = 1_000_000

_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")
# A fence's language tag ends at a newline, or, on a one-line fence, `sql` and a space.
_FENCE = re.compile(r"```(?:[A-Za-z0-9_+-]*[ \t]*\n|(?i:sql)[ \t]+)?(.*?)(?:```|\Z)", re.DOTALL)


@dataclass(frozen=True)
class Table:
    """One WikiSQL table: column names, column types ("text" or "real") and string cells."""

    header: list[str]
    types: list[str]
    rows: list[list[str]]

    def __post_init__(self) -> None:
        if not self.header:
            raise ValueError("a table needs at least one column")
        if len(self.header) != len(self.types):
            raise ValueError(
                f"header has {len(self.header)} columns but types has {len(self.types)}"
            )
        bad = sorted({t for t in self.types if t not in _TYPES})
        if bad:
            raise ValueError(f"column types must be 'text' or 'real', got {bad}")
        # SQLite column names are case-insensitive, so "Year" and "year" would collide in
        # CREATE TABLE (and a query could not tell them apart anyway).
        seen: dict[str, str] = {}
        for name in self.header:
            key = name.casefold()
            if key in seen:
                raise ValueError(
                    f"duplicate column name {name!r} (same as {seen[key]!r} ignoring case)"
                )
            seen[key] = name
        for i, row in enumerate(self.rows):
            if len(row) != len(self.header):
                raise ValueError(
                    f"row {i} has {len(row)} cells but the header has {len(self.header)} columns"
                )


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _authorizer(
    action: int, arg1: str | None, arg2: str | None, db: str | None, src: str | None
) -> int:
    if action not in _ALLOWED_ACTIONS:
        return sqlite3.SQLITE_DENY
    # For SQLITE_FUNCTION the function name is the second argument.
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in _DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def connect(table: Table) -> sqlite3.Connection:
    """An in-memory database holding `table` as `t`, locked down to read-only queries.

    Text columns are declared `TEXT COLLATE NOCASE`: WikiSQL's question values are
    lower-cased while the table cells are not ('butler cc (ks)' against 'Butler CC (KS)'),
    and WikiSQL's own evaluator compares case-insensitively, so `=` and `IN` on text do too.
    Real columns are declared `REAL`, so SQLite's column affinity stores numeric-looking
    cells ("21", "3.5") as numbers and `MAX`, `<` and `SUM` work on them, while dirty cells
    ("32, 44") stay text.
    """
    conn = sqlite3.connect(":memory:")
    columns = ", ".join(
        f"{_quote_identifier(name)} {'TEXT COLLATE NOCASE' if kind == 'text' else 'REAL'}"
        for name, kind in zip(table.header, table.types, strict=True)
    )
    conn.execute(f"CREATE TABLE {TABLE_NAME} ({columns})")
    placeholders = ", ".join("?" for _ in table.header)
    conn.executemany(f"INSERT INTO {TABLE_NAME} VALUES ({placeholders})", table.rows)
    conn.commit()
    conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, _MAX_LENGTH)
    conn.execute("PRAGMA query_only = ON")
    # Installed last: from here on even PRAGMA is refused, so query_only cannot be undone.
    conn.set_authorizer(_authorizer)
    return conn


@dataclass(frozen=True)
class Execution:
    """The rows a statement returned, or why it failed: exactly one of the two is set."""

    rows: tuple[tuple[object, ...], ...] | None
    error: str | None

    def __post_init__(self) -> None:
        if (self.rows is None) == (self.error is None):
            raise ValueError("an Execution holds exactly one of rows and error")


def execute(
    conn: sqlite3.Connection, sql: str, max_steps: int = 1_000_000, max_rows: int = 10_000
) -> Execution:
    """Run one statement on `conn` and return its rows, or an error as "<category>: <message>".

    The category is "invalid_sql" (not a string, or empty), "step_limit" (aborted after
    about `max_steps` VM instructions), "too_many_rows" (more than `max_rows` rows),
    "too_many_bytes" (the rows hold more than `_MAX_RESULT_BYTES` of text and blobs),
    "no_result" (the statement returns no result set, e.g. only a comment), "memory" (the
    interpreter ran out of memory building the rows), or the sqlite3 exception's class name
    ("OperationalError" for syntax errors, unknown columns and denied actions;
    "ProgrammingError" for more than one statement). Never raises for bad SQL.
    """
    if not isinstance(sql, str) or not sql.strip():
        return Execution(None, "invalid_sql: the SQL is empty or not a string")
    steps = 0
    aborted = False

    def progress() -> int:
        nonlocal steps, aborted
        steps += _PROGRESS_EVERY
        if steps > max_steps:
            aborted = True
            return 1  # non-zero interrupts the running statement
        return 0

    conn.set_progress_handler(progress, _PROGRESS_EVERY)
    rows: list[tuple[object, ...]] = []
    size = 0
    try:
        cursor = conn.execute(sql)
        if cursor.description is None:
            return Execution(None, "no_result: the statement returns no result set")
        # The VM keeps stepping while rows are fetched, so the handler must stay installed;
        # rows are fetched one at a time so the byte budget stops a query before it has
        # built a result too large to hold.
        for row in cursor:
            rows.append(tuple(row))
            if len(rows) > max_rows:
                return Execution(None, f"too_many_rows: more than {max_rows} rows")
            size += sum(len(v) for v in row if isinstance(v, str | bytes))
            if size > _MAX_RESULT_BYTES:
                return Execution(
                    None, f"too_many_bytes: the rows hold more than {_MAX_RESULT_BYTES} bytes"
                )
    except (sqlite3.Error, sqlite3.Warning) as exc:
        if aborted:
            return Execution(None, f"step_limit: aborted after about {max_steps} VM steps")
        return Execution(None, f"{type(exc).__name__}: {exc}")
    except MemoryError:
        return Execution(None, "memory: the result could not be held in memory")
    finally:
        conn.set_progress_handler(None, 0)
    return Execution(tuple(rows), None)


def _normalise_value(value: object) -> object:
    # bool is an int subclass; SQLite never returns one, but a caller's rows might.
    if isinstance(value, int | float) and not isinstance(value, bool):
        return round(float(value), 6)
    return value


def normalise(rows: Sequence[Sequence[object]]) -> tuple[tuple[object, ...], ...]:
    """Rows made comparable across equivalent queries: every number becomes a float rounded
    to 6 decimal places (so `COUNT` 5 and `SUM` 5.0 match, and float noise in `AVG` does not
    break a match); strings, None and blobs are kept as they are."""
    return tuple(tuple(_normalise_value(v) for v in row) for row in rows)


@dataclass(frozen=True)
class ExecMatch:
    """Whether the candidate's result equals the reference's, with both executions."""

    match: bool
    reference: Execution
    candidate: Execution
    reference_on_empty: Execution | None = None
    """The reference run on the same table with no rows (None when it was not run)."""

    @property
    def reference_empty(self) -> bool:
        """The reference ran and found nothing: its result does not depend on the table's
        rows, which shows as the same result on a copy of the table with no rows at all. That
        covers no rows, an aggregate over no matching rows (one row holding NULL, or 0 from
        `COUNT`) and a constant. A match here is weak evidence: any query that finds nothing,
        `MIN` for `MAX` over the same empty set or any `COUNT` with a wrong filter, matches
        it, and doubling the table's rows cannot tell them apart either."""
        rows = self.reference.rows
        if rows is None:
            return False
        if all(all(v is None for v in row) for row in rows):
            return True
        empty = self.reference_on_empty
        return (
            empty is not None
            and empty.rows is not None
            and _same_result(rows, empty.rows, ordered=False)
        )


def _same_result(
    reference: tuple[tuple[object, ...], ...],
    candidate: tuple[tuple[object, ...], ...],
    ordered: bool,
) -> bool:
    ref, cand = normalise(reference), normalise(candidate)
    if ordered:
        return ref == cand
    return Counter(ref) == Counter(cand)


def execution_match(table: Table, reference_sql: str, candidate_sql: str) -> ExecMatch:
    """Run both queries on `table` and compare their results.

    Results are compared as multisets of rows (row order from SQLite is unspecified without
    an ORDER BY), or as ordered lists when the reference has an ORDER BY outside its string
    literals. Each query gets its own fresh connection, so nothing one does can affect the
    other. A query that errors never matches, on either side. The reference also runs on a
    copy of the table with no rows, which is what `ExecMatch.reference_empty` compares with.
    """
    try:
        with closing(connect(table)) as conn:
            reference = execute(conn, reference_sql)
        with closing(connect(table)) as conn:
            candidate = execute(conn, candidate_sql)
    except (sqlite3.Error, sqlite3.Warning) as exc:
        # The table itself cannot be held (a NUL byte in a column name, thousands of
        # columns): neither side ran, and neither matches.
        failed = Execution(None, f"table: {type(exc).__name__}: {exc}")
        return ExecMatch(match=False, reference=failed, candidate=failed)
    on_empty: Execution | None = None
    if reference.rows is not None:  # a property of the reference alone, whatever the candidate did
        with closing(connect(Table(table.header, table.types, []))) as conn:
            on_empty = execute(conn, reference_sql)
    if reference.rows is None or candidate.rows is None:
        return ExecMatch(
            match=False, reference=reference, candidate=candidate, reference_on_empty=on_empty
        )
    ordered = bool(_ORDER_BY.search(_without_literals(reference_sql)))
    return ExecMatch(
        match=_same_result(reference.rows, candidate.rows, ordered),
        reference=reference,
        candidate=candidate,
        reference_on_empty=on_empty,
    )


# --- text helpers -------------------------------------------------------------------------


def extract_sql(text: str) -> str:
    """The SQL in a model completion: the first fenced code block's contents if there is one
    (```sql ... ``` or ``` ... ```; an unclosed fence runs to the end, as when generation
    stopped at the token limit), else the whole text; stripped, minus one trailing `;`."""
    sql = text.strip()
    fence = _FENCE.search(sql)
    if fence is not None:
        inner = fence.group(1).strip()
        # A fence with nothing inside is a stray closing fence after the query ("SELECT 1
        # ```"); the query is the text around it.
        sql = inner if inner else sql.replace("```", "").strip()
    if sql.endswith(";"):
        sql = sql[:-1].rstrip()
    return sql


def _segments(sql: str) -> list[tuple[str, bool]]:
    """`sql` split into (text, is_literal) pieces; a literal piece keeps its quotes, and a
    doubled '' inside a literal is part of it. An unclosed literal runs to the end."""
    out: list[tuple[str, bool]] = []
    i, start, n = 0, 0, len(sql)
    while i < n:
        if sql[i] != "'":
            i += 1
            continue
        if start < i:
            out.append((sql[start:i], False))
        j = i + 1
        while j < n:
            if sql[j] == "'":
                if j + 1 < n and sql[j + 1] == "'":
                    j += 2
                    continue
                break
            j += 1
        end = min(j + 1, n)
        out.append((sql[i:end], True))
        i = start = end
    if start < n:
        out.append((sql[start:], False))
    return out


def _without_literals(sql: str) -> str:
    return " ".join(text for text, literal in _segments(sql) if not literal)


def normalise_sql(sql: str, casefold_literals: bool = False) -> str:
    """`sql` in a canonical spelling for strict exact match: outside single-quoted string
    literals, runs of whitespace become one space and letters are lower-cased; literal
    contents are kept exactly unless `casefold_literals` lower-cases them too (the spelling
    that matches how the sandbox compares text, case-insensitively). Leading and trailing
    whitespace and trailing semicolons are removed.

    All trailing semicolons go, not just one, so that the function is idempotent ("x;;"
    would otherwise become "x;" and then "x").
    """
    pieces = [
        (text.lower() if casefold_literals else text)
        if literal
        else _WHITESPACE.sub(" ", text).lower()
        for text, literal in _segments(sql)
    ]
    out = "".join(pieces).strip()
    while out.endswith(";"):
        out = out[:-1].rstrip()
    return out
