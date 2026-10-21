"""The WikiSQL loader for the generation eval: pinned rows, gold queries as SQLite SQL.

WikiSQL stores each query as indices (`sel`, `agg`, `conds`) into a table's header, not as
text. The student and the teacher write SQL, so the gold answer is rendered here into the
same dialect a prediction is executed in: one table named `t`, double-quoted column names,
single-quoted text values. The original Hub repo is a loader script `datasets` 5 no longer
runs, so the rows come from the Hub's parquet conversion at a pinned commit.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from anvil.errors import ConfigError
from anvil.evaluate.sql import TABLE_NAME, Table
from anvil.teacher.base import Item
from eval.data import Sample

# WikiSQL's lib/query.py. The "OP" operator is declared there but never used in the data.
AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]
TABLE = TABLE_NAME  # the sandbox names its one table `t`
COLUMN_TYPES = ("text", "real")

# A plain decimal literal. `float()` alone would also accept `nan`, `inf` and `1_000`, none
# of which SQLite reads as a number.
_NUMBER = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


@dataclass(frozen=True)
class WikiSqlRef:
    repo: str
    revision: str
    filename: str
    license: str = ""


@dataclass(frozen=True)
class SqlExample:
    id: str
    question: str
    header: list[str]
    types: list[str]
    """`text` or `real`, one per column."""
    rows: list[list[str]]
    sql: str
    """The gold query as SQLite SQL over table `t` (`gold_sql`)."""
    agg: str
    """`""` or the aggregate name (`COUNT`, ...), for error slices."""
    n_conds: int

    def table(self) -> Table:
        """The example's table in the sandbox the eval executes queries in."""
        return Table(self.header, self.types, self.rows)


@dataclass
class LoadReport:
    rows: int = 0
    kept: int = 0
    skipped: Counter[str] = field(default_factory=Counter)
    """Skip reason -> rows dropped for it."""


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def quote_text(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _literal(value: str, column_type: str) -> str:
    """A condition value as an SQL literal: a bare number for a `real` column whose value
    reads as one, a quoted string otherwise.

    SQLite would match either form here: a column declared REAL has numeric affinity, and a
    comparison with a text literal converts the literal to a number first when it looks like
    one (`x = '21'` and `x = 21` select the same rows, checked in `sqlite3`). The bare number
    is still the canonical form because it is what a person writes, so a student trained on
    it learns ordinary SQL, and it does not depend on the column's declared affinity. Values
    in `real` columns that are not plain numbers (`117,682`) stay quoted: SQLite keeps such
    cells as text, and the quoted value matches them as text.
    """
    stripped = value.strip()
    if column_type == "real" and _NUMBER.fullmatch(stripped):
        number = float(stripped)
        if math.isfinite(number):
            if number.is_integer() and abs(number) < 1e15:
                return str(int(number))
            return repr(number)
    return quote_text(value)


def gold_sql(
    header: Sequence[str],
    types: Sequence[str],
    sel: int,
    agg: int,
    conds: Sequence[tuple[int, int, str]],
) -> str:
    """The query `(sel, agg, conds)` as SQLite SQL over table `t`. `conds` are
    `(column index, operator index, value)` triples, joined by `AND` as WikiSQL means them.
    An operator or aggregate index WikiSQL does not define raises `ConfigError`."""
    if not 0 <= agg < len(AGG_OPS):
        raise ConfigError(f"aggregate index {agg} is not one of WikiSQL's {len(AGG_OPS)}")
    column = quote_ident(header[sel])
    target = f"{AGG_OPS[agg]}({column})" if AGG_OPS[agg] else column
    sql = f"SELECT {target} FROM {TABLE}"
    parts: list[str] = []
    for col, op, value in conds:
        if not 0 <= op < len(COND_OPS) - 1:
            raise ConfigError(
                f"condition operator index {op} is not one of = > <; WikiSQL never uses its"
                " `OP` operator, so the dataset revision is not the one this loader expects"
            )
        parts.append(f"{quote_ident(header[col])} {COND_OPS[op]} {_literal(value, types[col])}")
    if parts:
        sql += " WHERE " + " AND ".join(parts)
    return sql


def render_input(header: Sequence[str], types: Sequence[str], question: str) -> str:
    """The input the teacher and the student see: the schema on one line, the question on
    the next. Cell values are left out: training and labelling cost scale with tokens, and
    WikiSQL's questions carry the values a condition needs."""
    columns = ", ".join(f"{quote_ident(h)} ({ty})" for h, ty in zip(header, types, strict=True))
    return f"Table {TABLE} columns: {columns}\nQuestion: {question.strip()}"


def _skip_reason(
    header: list[str], types: list[str], rows: list[list[str]], sel: int, cond_cols: list[int]
) -> str | None:
    if not header:
        return "empty_header"
    # SQLite folds identifier case (ASCII only), so `"Notes"` and `"notes"` are one column
    # and `CREATE TABLE` would fail; casefold is a superset of that and costs a few rows.
    if len({h.casefold() for h in header}) != len(header):
        return "duplicate_columns"
    if len(types) != len(header) or any(ty not in COLUMN_TYPES for ty in types):
        return "bad_types"
    if any(len(r) != len(header) for r in rows):
        return "ragged_rows"
    if not all(0 <= c < len(header) for c in [sel, *cond_cols]):
        return "column_out_of_range"
    return None


def _field[T](obj: Mapping[str, object], key: str, kind: type[T]) -> T:
    """`obj[key]`, checked to be a `kind`: a layout change in the parquet conversion should
    fail loudly here, not as a confusing error further down."""
    value = obj.get(key)
    if not isinstance(value, kind):
        raise ConfigError(
            f"WikiSQL row field {key!r} is {type(value).__name__}, expected {kind.__name__};"
            " the parquet layout changed, pin the revision this loader was written for"
        )
    return value


def examples_from_rows(
    rows: Iterable[Mapping[str, object]],
) -> tuple[list[SqlExample], LoadReport]:
    """Every usable row as an `SqlExample` (id `wikisql-<row index>`), plus how many rows were
    skipped and why: tables SQLite cannot hold as written (empty or duplicate column names,
    rows of the wrong width, unknown types) and queries naming a column the table lacks."""
    report = LoadReport()
    examples: list[SqlExample] = []
    for i, row in enumerate(rows):
        report.rows += 1
        table = _field(row, "table", dict)
        query = _field(row, "sql", dict)
        conds = _field(query, "conds", dict)
        header = [str(h) for h in _field(table, "header", list)]
        types = [str(ty) for ty in _field(table, "types", list)]
        cells = [[str(c) for c in r] for r in _field(table, "rows", list)]
        sel = _field(query, "sel", int)
        agg = _field(query, "agg", int)
        triples = [
            (int(c), int(o), str(v))
            for c, o, v in zip(
                _field(conds, "column_index", list),
                _field(conds, "operator_index", list),
                _field(conds, "condition", list),
                strict=True,
            )
        ]
        reason = _skip_reason(header, types, cells, sel, [c for c, _, _ in triples])
        if reason is not None:
            report.skipped[reason] += 1
            continue
        examples.append(
            SqlExample(
                id=f"wikisql-{i}",
                question=str(row["question"]),
                header=header,
                types=types,
                rows=cells,
                sql=gold_sql(header, types, sel, agg, triples),
                agg=AGG_OPS[agg],
                n_conds=len(triples),
            )
        )
    report.kept = len(examples)
    return examples, report


def read_parquet_rows(path: Path) -> Iterator[dict[str, object]]:
    """The rows of a parquet file as dicts, a batch at a time (the train file is 25 MB
    compressed, mostly repeated tables)."""
    # Imported here: pyarrow is only installed with the `eval` dependency group.
    import pyarrow.parquet as pq

    for batch in pq.ParquetFile(path).iter_batches(batch_size=4096):
        yield from batch.to_pylist()


def load_examples(ref: WikiSqlRef) -> tuple[list[SqlExample], LoadReport]:
    """Download the pinned parquet file and read it with `examples_from_rows`."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(ref.repo, ref.filename, repo_type="dataset", revision=ref.revision)
    return examples_from_rows(read_parquet_rows(Path(path)))


def encode_sql_answer(sql: str) -> str:
    """A query as the stored answer string the generation teacher and the gold teacher use."""
    return json.dumps({"sql": sql}, ensure_ascii=False, separators=(",", ":"))


def sample_examples(
    examples: Sequence[SqlExample], n: int, seed: int, max_columns: int | None = None
) -> tuple[Sample, dict[str, SqlExample]]:
    """`n` examples with distinct, non-empty inputs in a seeded order, like `eval.data.sample`,
    optionally only from tables of at most `max_columns` columns (a wide table is a long
    prompt). Items keep the example ids, so the sample is traceable to the source row; the
    examples come back by id because scoring executes queries against their tables."""
    order = np.random.default_rng(seed).permutation(len(examples))
    items: list[Item] = []
    gold: dict[str, str] = {}
    chosen: dict[str, SqlExample] = {}
    seen: set[str] = set()
    for i in order.tolist():
        ex = examples[i]
        if max_columns is not None and len(ex.header) > max_columns:
            continue
        if not ex.question.strip():
            continue
        text = render_input(ex.header, ex.types, ex.question)
        if text in seen:
            continue
        seen.add(text)
        items.append(Item(id=ex.id, text=text))
        gold[ex.id] = encode_sql_answer(ex.sql)
        chosen[ex.id] = ex
        if len(items) == n:
            break
    return Sample(items, gold, len(examples)), chosen
