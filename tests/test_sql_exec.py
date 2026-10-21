from __future__ import annotations

import time

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from anvil.evaluate.sql import (
    Execution,
    Table,
    connect,
    execute,
    execution_match,
    extract_sql,
    normalise,
    normalise_sql,
)

SCHOOLS = Table(
    header=["Player", "No.", "School/Club Team"],
    types=["text", "real", "text"],
    rows=[
        ["Antonio Lang", "21", "Duke"],
        ["Voshon Lenard", "2", "Minnesota"],
        ["Martin Lewis", "32, 44", "Butler CC (KS)"],
        ["Brad Lohaus", "33", "Iowa"],
        ["Art Long", "3.5", "Cincinnati"],
    ],
)


def run(sql: str, table: Table = SCHOOLS) -> Execution:
    conn = connect(table)
    try:
        return execute(conn, sql)
    finally:
        conn.close()


# --- table validation ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("header", "types", "rows", "message"),
    [
        (["a", "b"], ["text"], [], "types has 1"),
        (["a"], ["integer"], [], "'text' or 'real'"),
        (["a", "b"], ["text", "text"], [["x"]], "row 0 has 1 cells"),
        (["Year", "year"], ["text", "text"], [], "duplicate column name"),
        ([], [], [], "at least one column"),
    ],
)
def test_table_rejects_bad_layouts(
    header: list[str], types: list[str], rows: list[list[str]], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Table(header=header, types=types, rows=rows)


# --- typing and collation -----------------------------------------------------------------


def test_text_columns_compare_case_insensitively() -> None:
    result = run('SELECT "Player" FROM t WHERE "School/Club Team" = \'butler cc (ks)\'')
    assert result.rows == (("Martin Lewis",),)


def test_real_columns_hold_numbers_and_dirty_cells_stay_text() -> None:
    assert run('SELECT typeof("No.") FROM t WHERE "Player" = \'Antonio Lang\'').rows == (
        ("real",),
    )
    assert run('SELECT "No." FROM t WHERE "Player" = \'Martin Lewis\'').rows == (("32, 44",),)
    # Numeric comparison, not string comparison: "21" < "3.5" as text but not as numbers.
    assert run('SELECT "Player" FROM t WHERE "No." < 4 ORDER BY "No."').rows == (
        ("Voshon Lenard",),
        ("Art Long",),
    )
    assert run('SELECT MAX("No.") FROM t WHERE typeof("No.") = \'real\'').rows == ((33.0,),)


def test_column_names_with_quotes_spaces_and_slashes() -> None:
    table = Table(
        header=['Say "hi"', "School/Club Team", "Two words"],
        types=["text", "text", "real"],
        rows=[["a", "b", "1"], ["c", "d", "2"]],
    )
    assert run('SELECT "Say ""hi""" FROM t WHERE "Two words" = 2', table).rows == (("c",),)
    assert run('SELECT "School/Club Team" FROM t WHERE "Say ""hi""" = \'A\'', table).rows == (
        ("b",),
    )


# --- comparison ---------------------------------------------------------------------------


def test_rows_compare_as_a_multiset_without_order_by() -> None:
    result = execution_match(
        SCHOOLS,
        "SELECT \"Player\" FROM t WHERE \"School/Club Team\" IN ('duke', 'iowa')",
        "SELECT \"Player\" FROM t WHERE \"School/Club Team\" IN ('iowa', 'duke') "
        'ORDER BY "Player" DESC',
    )
    assert result.match


def test_multiset_counts_duplicates() -> None:
    result = execution_match(SCHOOLS, "SELECT 1 FROM t", "SELECT DISTINCT 1 FROM t")
    assert not result.match


def test_rows_compare_in_order_when_the_reference_orders() -> None:
    ref = 'SELECT "Player" FROM t ORDER BY "Player"'
    assert execution_match(SCHOOLS, ref, 'SELECT "Player" FROM t ORDER BY "Player"').match
    assert not execution_match(SCHOOLS, ref, 'SELECT "Player" FROM t ORDER BY "Player" DESC').match


def test_order_by_inside_a_literal_does_not_make_the_comparison_ordered() -> None:
    ref = 'SELECT "Player" FROM t WHERE "Player" != \'order by\''
    assert execution_match(SCHOOLS, ref, 'SELECT "Player" FROM t ORDER BY "Player" DESC').match


def test_count_matches_an_equal_sum() -> None:
    table = Table(header=["x"], types=["real"], rows=[["1"]] * 5)
    assert execution_match(table, 'SELECT COUNT("x") FROM t', 'SELECT SUM("x") FROM t').match


def test_normalise_rounds_numbers_and_keeps_strings() -> None:
    assert normalise([(5, 5.0000000001, "5", None)]) == ((5.0, 5.0, "5", None),)


def test_reference_error_never_matches() -> None:
    result = execution_match(SCHOOLS, "SELECT nope FROM t", "SELECT nope FROM t")
    assert not result.match
    assert result.reference.error is not None
    assert result.reference.error.startswith("OperationalError: no such column")


def test_candidate_error_never_matches() -> None:
    result = execution_match(SCHOOLS, 'SELECT "Player" FROM t', "SELEC Player FROM t")
    assert not result.match
    assert result.reference.rows is not None
    assert result.candidate.error is not None
    assert "syntax error" in result.candidate.error


def test_empty_results_match_and_are_flagged() -> None:
    result = execution_match(
        SCHOOLS,
        'SELECT "Player" FROM t WHERE "School/Club Team" = \'yale\'',
        'SELECT "Player" FROM t WHERE "School/Club Team" = \'harvard\'',
    )
    assert result.match
    assert result.reference_empty
    found = execution_match(SCHOOLS, 'SELECT "Player" FROM t', 'SELECT "Player" FROM t')
    assert found.match
    assert not found.reference_empty


def test_a_count_over_no_rows_counts_as_an_empty_reference() -> None:
    # COUNT over nothing is 0, not NULL, and 0 doubled is still 0: any COUNT whose filter
    # matches nothing agrees with it, so it is flagged like a NULL aggregate.
    result = execution_match(
        SCHOOLS,
        'SELECT COUNT("Player") FROM t WHERE "School/Club Team" = \'yale\'',
        'SELECT COUNT("No.") FROM t WHERE "Player" = \'nobody\'',
    )
    assert result.match
    assert result.reference_empty
    # A count that found rows depends on them.
    counted = execution_match(
        SCHOOLS, 'SELECT COUNT("Player") FROM t', 'SELECT COUNT("Player") FROM t'
    )
    assert counted.match
    assert not counted.reference_empty
    assert counted.reference_on_empty is not None
    assert counted.reference_on_empty.rows == ((0,),)


def test_reference_empty_does_not_depend_on_the_candidate() -> None:
    reference = 'SELECT COUNT("Player") FROM t WHERE "School/Club Team" = \'yale\''
    with_error = execution_match(SCHOOLS, reference, "SELEC 1")
    assert not with_error.match
    assert with_error.reference_empty
    assert execution_match(SCHOOLS, reference, "SELECT 0").reference_empty


def test_a_constant_reference_is_flagged_as_data_independent() -> None:
    assert execution_match(SCHOOLS, "SELECT 1", "SELECT 1").reference_empty


def test_a_reference_that_errors_is_not_empty() -> None:
    result = execution_match(SCHOOLS, "SELEC 1", "SELECT 1")
    assert not result.match
    assert not result.reference_empty
    assert result.reference_on_empty is None


def test_a_table_sqlite_cannot_hold_is_an_error_on_both_sides() -> None:
    table = Table(["a\x00b"], ["text"], [["x"]])
    result = execution_match(table, "SELECT 1", "SELECT 1")
    assert not result.match
    assert result.reference.error is not None
    assert result.reference.error.startswith("table:")
    assert result.candidate.error == result.reference.error


# --- sandbox ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql",
    [
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA query_only = OFF",
        "PRAGMA table_info(t)",
        "INSERT INTO t VALUES ('a', 1, 'b')",
        "UPDATE t SET \"Player\" = 'x'",
        "DELETE FROM t",
        "CREATE TEMP TABLE u (x)",
        "CREATE TABLE u (x)",
        "DROP TABLE t",
        "SELECT load_extension('/tmp/evil')",
    ],
)
def test_forbidden_statements_are_refused_not_raised(sql: str) -> None:
    conn = connect(SCHOOLS)
    result = execute(conn, sql)
    assert result.rows is None
    assert result.error is not None
    assert result.error.startswith(("DatabaseError:", "OperationalError:"))
    # The table is untouched and the connection still answers.
    assert execute(conn, "SELECT COUNT(*) FROM t").rows == ((5,),)
    conn.close()


def test_multiple_statements_are_refused() -> None:
    result = run("SELECT 1; SELECT 2")
    assert result.error is not None
    assert result.error.startswith("ProgrammingError:")


@pytest.mark.parametrize("sql", ["", "   ", "-- only a comment"])
def test_empty_or_comment_only_sql_is_an_error(sql: str) -> None:
    assert run(sql).error is not None


def test_non_string_sql_is_an_error() -> None:
    conn = connect(SCHOOLS)
    assert execute(conn, None).error == "invalid_sql: the SQL is empty or not a string"  # type: ignore[arg-type]
    conn.close()


def test_infinite_recursion_is_stopped_by_the_step_limit() -> None:
    conn = connect(SCHOOLS)
    start = time.perf_counter()
    result = execute(
        conn,
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT COUNT(*) FROM r",
    )
    assert time.perf_counter() - start < 30.0  # the limit is about a million steps, not time
    assert result.error is not None
    assert result.error.startswith("step_limit:")
    # The handler is reset afterwards: a long but finite query now runs to completion.
    assert execute(conn, "SELECT 1", max_steps=10**9).rows == ((1,),)
    conn.close()


def test_a_finite_recursive_cte_runs() -> None:
    sql = (
        "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r WHERE x < 3) "
        "SELECT x FROM r"
    )
    assert run(sql).rows == ((1,), (2,), (3,))


def test_too_many_rows_are_refused() -> None:
    conn = connect(SCHOOLS)
    sql = "SELECT * FROM t AS a, t AS b"  # 25 rows
    assert execute(conn, sql, max_rows=25).rows is not None
    result = execute(conn, sql, max_rows=24)
    assert result.error == "too_many_rows: more than 24 rows"
    conn.close()


def test_huge_blobs_are_refused() -> None:
    assert run("SELECT length(zeroblob(100000000))").error is not None


def test_many_rows_of_large_values_are_refused_before_they_fill_memory() -> None:
    # Each row is under the length cap and there are fewer rows than the row cap, so only
    # the byte budget stops this (10,000 rows of 9,999 bytes would be 100 MB per column).
    sql = (
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c LIMIT 9000) "
        "SELECT zeroblob(9999), zeroblob(9999), zeroblob(9999) FROM c"
    )
    result = run(sql)
    assert result.error is not None
    assert result.error.startswith("too_many_bytes:")
    # The same shape under the budget runs.
    small = run(sql.replace("LIMIT 9000", "LIMIT 3"))
    assert small.rows is not None
    assert len(small.rows) == 3


def test_quadratic_builtins_are_bounded_by_the_length_cap() -> None:
    # ltrim(X, Y) is O(|X| * |Y|) inside one VM instruction, where the step limit cannot
    # interrupt it; the length cap keeps the worst case to milliseconds.
    sql = (
        "SELECT length(ltrim(replace(hex(zeroblob(2000)), '0', 'a'),"
        " replace(hex(zeroblob(2000)), '0', 'b') || 'a'))"
    )
    start = time.perf_counter()
    result = run(sql)  # refused outright (the character set's table exceeds the cap) or fast
    assert result.rows == ((0,),) or result.error == "DataError: string or blob too big"
    assert time.perf_counter() - start < 5.0
    assert run("SELECT length(replace(hex(zeroblob(6000)), '0', 'a'))").error is not None


def test_fts3_tokenizer_is_denied() -> None:
    result = run("SELECT hex(fts3_tokenizer('simple'))")
    assert result.error is not None
    assert "not authorized" in result.error or "no such function" in result.error


# --- extract_sql and normalise_sql --------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("SELECT 1", "SELECT 1"),
        ("  SELECT 1;  \n", "SELECT 1"),
        ("SELECT 1;;", "SELECT 1;"),
        ('```sql\nSELECT "a" FROM t;\n```', 'SELECT "a" FROM t'),
        ("Here you go:\n```\nSELECT 1\n```\nand ```sql\nSELECT 2\n```", "SELECT 1"),
        ("```SELECT 1```", "SELECT 1"),
        ("```sql SELECT 4```", "SELECT 4"),
        ("```SQL  SELECT 4;```", "SELECT 4"),
        ("```sqlite\nSELECT 5\n```", "SELECT 5"),
        ("```sql\nSELECT 1 FROM t WHERE", "SELECT 1 FROM t WHERE"),
        ("SELECT 1 ```", "SELECT 1"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_extract_sql(text: str, expected: str) -> None:
    assert extract_sql(text) == expected


def test_normalise_sql_can_casefold_literals() -> None:
    sql = 'SELECT  "A" FROM t WHERE "B" = \'North Island\''
    assert normalise_sql(sql) == 'select "a" from t where "b" = \'North Island\''
    assert normalise_sql(sql, casefold_literals=True) == (
        'select "a" from t where "b" = \'north island\''
    )


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        ("SELECT  COUNT(*)\n FROM t ;", "select count(*) from t"),
        (
            'SELECT "A" FROM t WHERE "B" = \'Butler  CC\'',
            'select "a" from t where "b" = \'Butler  CC\'',
        ),
        ("SELECT 'It''s  A' , X FROM T", "select 'It''s  A' , x from t"),
        ("  select 1;;  ", "select 1"),
        ("SELECT 'unclosed  LIT", "select 'unclosed  LIT"),
        ("", ""),
    ],
)
def test_normalise_sql(sql: str, expected: str) -> None:
    assert normalise_sql(sql) == expected


# --- properties ---------------------------------------------------------------------------

_cell = st.text(
    alphabet=st.characters(codec="utf-8", exclude_characters="\x00", exclude_categories=["Cs"]),
    max_size=8,
)
_name = st.text(
    alphabet=st.characters(codec="utf-8", exclude_characters="\x00", exclude_categories=["Cs"]),
    min_size=1,
    max_size=8,
)


@st.composite
def tables_and_queries(draw: st.DrawFn) -> tuple[Table, str]:
    header = draw(st.lists(_name, min_size=1, max_size=4, unique_by=lambda s: s.casefold()))
    types = draw(
        st.lists(st.sampled_from(["text", "real"]), min_size=len(header), max_size=len(header))
    )
    rows = draw(st.lists(st.lists(_cell, min_size=len(header), max_size=len(header)), max_size=5))
    table = Table(header=header, types=types, rows=rows)
    col = '"' + draw(st.sampled_from(header)).replace('"', '""') + '"'
    value = draw(st.one_of(_cell, st.sampled_from([r[0] for r in rows] or [""])))
    literal = "'" + value.replace("'", "''") + "'"
    return table, f"SELECT {col} FROM t WHERE {col} = {literal}"


@settings(max_examples=60, deadline=None)
@given(tables_and_queries())
def test_a_query_matches_itself(case: tuple[Table, str]) -> None:
    table, sql = case
    result = execution_match(table, sql, sql)
    assert result.reference.error is None, result.reference.error
    assert result.match


@settings(max_examples=200, deadline=None)
@given(st.text(max_size=40))
def test_normalise_sql_is_idempotent(sql: str) -> None:
    once = normalise_sql(sql)
    assert normalise_sql(once) == once


def test_an_aggregate_over_no_rows_counts_as_an_empty_reference() -> None:
    # MAX over nothing is one row holding NULL, so MIN over nothing matches it.
    result = execution_match(
        SCHOOLS,
        'SELECT MAX("No.") FROM t WHERE "Player" = \'nobody\'',
        'SELECT MIN("No.") FROM t WHERE "Player" = \'nobody\'',
    )
    assert result.match
    assert result.reference_empty
