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
- the longest string or blob a query may build is capped (`zeroblob(1e9)` would otherwise
  allocate a gigabyte).

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
_DENIED_FUNCTIONS = frozenset({"load_extension"})

# How often (in VM instructions) the progress handler runs. Small enough that the step
# limit overshoots by at most this much, large enough that the callback costs nothing.
_PROGRESS_EVERY = 1000
# Longest string or blob a query may build, in bytes. WikiSQL cells are short; this only
# stops a query from allocating memory without bound.
_MAX_LENGTH = 1_000_000

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
    "no_result" (the statement returns no result set, e.g. only a comment), or the sqlite3
    exception's class name ("OperationalError" for syntax errors, unknown columns and denied
    actions; "ProgrammingError" for more than one statement). Never raises for bad SQL.
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
    try:
        cursor = conn.execute(sql)
        if cursor.description is None:
            return Execution(None, "no_result: the statement returns no result set")
        # The VM keeps stepping while rows are fetched, so the handler must stay installed.
        rows = cursor.fetchmany(max_rows + 1)
    except (sqlite3.Error, sqlite3.Warning) as exc:
        if aborted:
            return Execution(None, f"step_limit: aborted after about {max_steps} VM steps")
        return Execution(None, f"{type(exc).__name__}: {exc}")
    finally:
        conn.set_progress_handler(None, 0)
    if len(rows) > max_rows:
        return Execution(None, f"too_many_rows: more than {max_rows} rows")
    return Execution(tuple(tuple(row) for row in rows), None)


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

    @property
    def reference_empty(self) -> bool:
        """The reference ran and found nothing: no rows, or rows of NULLs only (an aggregate
        over no matching rows returns one row holding NULL). A match here is weak evidence:
        any query that finds nothing, `MIN` for `MAX` over the same empty set, matches it."""
        rows = self.reference.rows
        return rows is not None and all(all(v is None for v in row) for row in rows)


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
    other. A query that errors never matches, on either side.
    """
    with closing(connect(table)) as conn:
        reference = execute(conn, reference_sql)
    with closing(connect(table)) as conn:
        candidate = execute(conn, candidate_sql)
    if reference.rows is None or candidate.rows is None:
        return ExecMatch(match=False, reference=reference, candidate=candidate)
    ordered = bool(_ORDER_BY.search(_without_literals(reference_sql)))
    return ExecMatch(
        match=_same_result(reference.rows, candidate.rows, ordered),
        reference=reference,
        candidate=candidate,
    )


# --- text helpers -------------------------------------------------------------------------


def extract_sql(text: str) -> str:
    """The SQL in a model completion: the first fenced code block's contents if there is one
    (```sql ... ``` or ``` ... ```; an unclosed fence runs to the end, as when generation
    stopped at the token limit), else the whole text; stripped, minus one trailing `;`."""
    sql = text.strip()
    fence = _FENCE.search(sql)
    if fence is not None:
        sql = fence.group(1).strip()
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


def normalise_sql(sql: str) -> str:
    """`sql` in a canonical spelling for strict exact match: outside single-quoted string
    literals, runs of whitespace become one space and letters are lower-cased; literal
    contents are kept exactly. Leading and trailing whitespace and trailing semicolons are
    removed.

    All trailing semicolons go, not just one, so that the function is idempotent ("x;;"
    would otherwise become "x;" and then "x").
    """
    pieces = [
        text if literal else _WHITESPACE.sub(" ", text).lower() for text, literal in _segments(sql)
    ]
    out = "".join(pieces).strip()
    while out.endswith(";"):
        out = out[:-1].rstrip()
    return out
