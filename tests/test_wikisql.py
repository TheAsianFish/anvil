"""The WikiSQL loader: gold SQL rendering, the input text, skipping rules and sampling."""

from __future__ import annotations

import json
from typing import Any

import pytest

from anvil.errors import ConfigError
from anvil.evaluate.sql import connect, execute
from eval.wikisql import (
    SqlExample,
    examples_from_rows,
    gold_sql,
    render_input,
    sample_examples,
)

HEADER = ["Player", "No.", "Team"]
TYPES = ["text", "real", "text"]


def row(
    question: str = "Who wears 21?",
    header: list[str] | None = None,
    types: list[str] | None = None,
    rows: list[list[str]] | None = None,
    sel: int = 0,
    agg: int = 0,
    conds: list[tuple[int, int, str]] | None = None,
) -> dict[str, Any]:  # Any: the nested parquet layout, as pyarrow returns it
    header = HEADER if header is None else header
    return {
        "phase": 1,
        "question": question,
        "table": {
            "header": header,
            "types": TYPES if types is None else types,
            "rows": [["Ann", "21", "Reds"], ["Bo", "7", "Blues"]] if rows is None else rows,
            "id": "1-1-1",
        },
        "sql": {
            "human_readable": "",
            "sel": sel,
            "agg": agg,
            "conds": {
                "column_index": [c for c, _, _ in conds or []],
                "operator_index": [o for _, o, _ in conds or []],
                "condition": [v for _, _, v in conds or []],
            },
        },
    }


def test_gold_sql_without_aggregate_or_conditions() -> None:
    assert gold_sql(HEADER, TYPES, 0, 0, []) == 'SELECT "Player" FROM t'


def test_gold_sql_with_aggregate_and_one_condition() -> None:
    sql = gold_sql(HEADER, TYPES, 1, 3, [(2, 0, "Reds")])
    assert sql == 'SELECT COUNT("No.") FROM t WHERE "Team" = \'Reds\''


@pytest.mark.parametrize(
    ("agg", "expected"),
    [
        (0, 'SELECT "No." FROM t'),
        (1, 'SELECT MAX("No.") FROM t'),
        (2, 'SELECT MIN("No.") FROM t'),
        (3, 'SELECT COUNT("No.") FROM t'),
        (4, 'SELECT SUM("No.") FROM t'),
        (5, 'SELECT AVG("No.") FROM t'),
    ],
)
def test_gold_sql_renders_every_wikisql_aggregate(agg: int, expected: str) -> None:
    # WikiSQL's lib/query.py: agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG'].
    assert gold_sql(HEADER, TYPES, 1, agg, []) == expected


@pytest.mark.parametrize(("op", "symbol"), [(0, "="), (1, ">"), (2, "<")])
def test_gold_sql_renders_every_wikisql_condition_operator(op: int, symbol: str) -> None:
    # WikiSQL's lib/query.py: cond_ops = ['=', '>', '<', 'OP']; 'OP' never occurs in the data.
    assert (
        gold_sql(HEADER, TYPES, 0, 0, [(1, op, "3")])
        == f'SELECT "Player" FROM t WHERE "No." {symbol} 3'
    )


def test_gold_sql_joins_three_conditions_with_and() -> None:
    sql = gold_sql(HEADER, TYPES, 0, 1, [(1, 1, "3"), (1, 2, "30"), (2, 0, "Reds")])
    assert sql == (
        'SELECT MAX("Player") FROM t WHERE "No." > 3 AND "No." < 30 AND "Team" = \'Reds\''
    )


def test_gold_sql_escapes_quotes_in_columns_and_values() -> None:
    header = ['Say "hi"', "Name"]
    sql = gold_sql(header, ["text", "text"], 0, 0, [(1, 0, "O'Bryant")])
    assert sql == 'SELECT "Say ""hi""" FROM t WHERE "Name" = \'O\'\'Bryant\''


@pytest.mark.parametrize(
    ("value", "literal"),
    [
        ("21", "21"),
        (" 21 ", "21"),
        ("21.0", "21"),
        ("21.5", "21.5"),
        ("-0.25", "-0.25"),
        ("21a", "'21a'"),
        ("117,682", "'117,682'"),
        ("nan", "'nan'"),
        ("1_000", "'1_000'"),
    ],
)
def test_real_values_are_bare_numbers_only_when_numeric(value: str, literal: str) -> None:
    assert gold_sql(HEADER, TYPES, 0, 0, [(1, 0, value)]) == (
        f'SELECT "Player" FROM t WHERE "No." = {literal}'
    )


def test_text_columns_keep_numbers_quoted() -> None:
    sql = gold_sql(HEADER, TYPES, 1, 0, [(0, 0, "21")])
    assert sql == 'SELECT "No." FROM t WHERE "Player" = \'21\''


def test_gold_sql_rejects_the_unused_op_operator_and_unknown_aggregates() -> None:
    with pytest.raises(ConfigError, match="OP"):
        gold_sql(HEADER, TYPES, 0, 0, [(1, 3, "1")])
    with pytest.raises(ConfigError, match="aggregate"):
        gold_sql(HEADER, TYPES, 0, 6, [])


def test_render_input_is_schema_line_then_question() -> None:
    text = render_input(HEADER, TYPES, "  Who wears 21? ")
    assert text == (
        'Table t columns: "Player" (text), "No." (real), "Team" (text)\nQuestion: Who wears 21?'
    )


def test_examples_from_rows_skips_unusable_rows_and_counts_them() -> None:
    rows = [
        row(sel=1, agg=3, conds=[(2, 0, "Reds")]),
        row(header=["Name", "name", "Team"]),
        row(header=[], types=[], rows=[]),
        row(rows=[["Ann", "21"]]),
        row(sel=3),
        row(conds=[(5, 0, "x")]),
        row(types=["text", "integer", "text"]),
        row(question="Who plays for the Blues?", conds=[(2, 0, "Blues"), (1, 2, "10")]),
    ]
    examples, report = examples_from_rows(rows)
    assert [e.id for e in examples] == ["wikisql-0", "wikisql-7"]
    assert (report.rows, report.kept) == (8, 2)
    assert report.skipped == {
        "duplicate_columns": 1,
        "empty_header": 1,
        "ragged_rows": 1,
        "column_out_of_range": 2,
        "bad_types": 1,
    }
    first, last = examples
    assert (first.agg, first.n_conds) == ("COUNT", 1)
    assert (last.agg, last.n_conds) == ("", 2)
    assert last.sql == 'SELECT "Player" FROM t WHERE "Team" = \'Blues\' AND "No." < 10'


def test_examples_from_rows_rejects_a_changed_layout() -> None:
    broken = row()
    broken["table"]["header"] = "Player"
    with pytest.raises(ConfigError, match="header"):
        examples_from_rows([broken])


def _execute(example: SqlExample) -> list[tuple[object, ...]]:
    result = execute(connect(example.table()), example.sql)
    assert result.error is None, result.error
    assert result.rows is not None
    return list(result.rows)


def test_every_gold_query_runs_in_sqlite() -> None:
    header = ['Say "hi"', "No.", "Team's"]
    cells = [["O'Bryant", "21", "Reds"], ["Bo", "7.5", "Blues"], ["Cy", "n/a", "reds"]]
    rows = [
        row(header=header, rows=cells, sel=sel, agg=agg, conds=conds)
        for sel in range(3)
        for agg in range(6)
        for conds in [[], [(0, 0, "O'Bryant")], [(1, 1, "3"), (1, 2, "21.5"), (2, 0, "REDS")]]
    ]
    examples, _ = examples_from_rows(rows)
    assert len(examples) == len(rows)
    for example in examples:
        _execute(example)
    # Text compares without case and a REAL column compares as a number; the text cell
    # `n/a` stays text, which SQLite orders after every number, so `< 30` excludes it.
    count = gold_sql(header, TYPES, 0, 3, [(1, 2, "30"), (2, 0, "REDS")])
    assert _execute(SqlExample("x", "q", header, TYPES, cells, count, "COUNT", 2)) == [(1,)]


def _examples(n: int, columns: int = 3) -> list[SqlExample]:
    header = [f"c{j}" for j in range(columns)]
    types = ["text"] * columns
    return [
        SqlExample(f"wikisql-{i}", f"question {i % (n - 2)}", header, types, [], "", "", 0)
        for i in range(n)
    ]


def test_sample_examples_is_seeded_and_distinct() -> None:
    pool = _examples(30)
    sample, by_id = sample_examples(pool, 10, seed=3)
    again, _ = sample_examples(pool, 10, seed=3)
    other, _ = sample_examples(pool, 10, seed=4)
    assert [it.id for it in sample.items] == [it.id for it in again.items]
    assert [it.id for it in sample.items] != [it.id for it in other.items]
    assert len({it.text for it in sample.items}) == 10
    assert sample.source_rows == 30
    assert set(by_id) == set(sample.gold) == {it.id for it in sample.items}
    for it in sample.items:
        ex = by_id[it.id]
        assert it.text == render_input(ex.header, ex.types, ex.question)
        assert sample.gold[it.id] == json.dumps({"sql": ex.sql}, separators=(",", ":"))
    # Only 28 distinct questions exist, so asking for all 30 yields 28.
    assert len(sample_examples(pool, 30, seed=0)[0].items) == 28


def test_sample_examples_caps_the_column_count() -> None:
    pool = _examples(10, columns=3) + [
        SqlExample(f"wide-{i}", f"wide {i}", ["a", "b", "c", "d"], ["text"] * 4, [], "", "", 0)
        for i in range(10)
    ]
    sample, by_id = sample_examples(pool, 20, seed=0, max_columns=3)
    assert sample.items
    assert all(len(ex.header) <= 3 for ex in by_id.values())


# Six rows copied from `default/train/0000.parquet` at revision 48cfb60 (row indices 0, 4, 17,
# 66, 70 and 1261) to pin the real layout and the rendering of real queries. The questions,
# headers, types and queries are verbatim; the tables are cut to two rows, some cells edited so
# the condition matches one of them, and the empty page and section fields are dropped.
REAL_ROWS: list[dict[str, Any]] = [  # Any: the nested parquet layout
    {
        "phase": 1,
        "question": "Tell me what the notes are for South Australia ",
        "table": {
            "header": [
                "State/territory",
                "Text/background colour",
                "Format",
                "Current slogan",
                "Current series",
                "Notes",
            ],
            "types": ["text", "text", "text", "text", "text", "text"],
            "rows": [
                [
                    "Australian Capital Territory",
                    "blue/white",
                    "Yaa·nna",
                    "ACT · CELEBRATION OF A CENTURY 2013",
                    "YIL·00A",
                    "Slogan screenprinted on plate",
                ],
                [
                    "South Australia",
                    "black/white",
                    "Snnn·aaa",
                    "SOUTH AUSTRALIA",
                    "S000·AZD",
                    "No slogan on current series",
                ],
            ],
            "id": "1-1000181-1",
        },
        "sql": {
            "human_readable": "SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA",
            "sel": 5,
            "agg": 0,
            "conds": {
                "column_index": [3],
                "operator_index": [0],
                "condition": ["SOUTH AUSTRALIA"],
            },
        },
    },
    {
        "phase": 1,
        "question": "how many times is the fuel propulsion is cng?",
        "table": {
            "header": [
                "Order Year",
                "Manufacturer",
                "Model",
                "Fleet Series (Quantity)",
                "Powertrain (Engine/Transmission)",
                "Fuel Propulsion",
            ],
            "types": ["text", "text", "text", "text", "text", "text"],
            "rows": [
                [
                    "1992-93",
                    "Gillig",
                    "Phantom (High Floor)",
                    "444-464 (21)",
                    "DD S50EGR Allison WB-400R",
                    "Diesel",
                ],
                ["1996", "Gillig", "Phantom (High Floor)", "465-467 (3)", "DD S50", "CNG"],
            ],
            "id": "1-10007452-3",
        },
        "sql": {
            "human_readable": (
                "SELECT COUNT Fleet Series (Quantity) FROM table WHERE Fuel Propulsion = CNG"
            ),
            "sel": 3,
            "agg": 3,
            "conds": {"column_index": [5], "operator_index": [0], "condition": ["CNG"]},
        },
    },
    {
        "phase": 1,
        "question": "Which number was Shaquille O'Neal?",
        "table": {
            "header": [
                "Player",
                "No.",
                "Nationality",
                "Position",
                "Years in Toronto",
                "School/Club Team",
            ],
            "types": ["text", "real", "text", "text", "text", "text"],
            "rows": [
                ["Shaquille O'Neal", "13", "United States", "Center", "2009-10", "Bradley"],
                [
                    "Jermaine O'Neal",
                    "6",
                    "United States",
                    "Forward-Center",
                    "2008-09",
                    "Eau Claire High School",
                ],
            ],
            "id": "1-10015132-14",
        },
        "sql": {
            "human_readable": "SELECT No. FROM table WHERE Player = Shaquille O'Neal",
            "sel": 1,
            "agg": 0,
            "conds": {
                "column_index": [0],
                "operator_index": [0],
                "condition": ["Shaquille O'Neal"],
            },
        },
    },
    {
        "phase": 1,
        "question": "Name the member that has 12 chapters",
        "table": {
            "header": ["Member", "Headquarters", "Classification", "Chapters", "Founded", "UCCFS"],
            "types": ["text", "text", "text", "real", "text", "real"],
            "rows": [
                [
                    "Alpha Nu Omega",
                    "Baltimore, Maryland",
                    "Fraternity & Sorority",
                    "26",
                    "1988 at Morgan State University",
                    "2006",
                ],
                [
                    "Men of God",
                    "San Antonio, Texas",
                    "Fraternity",
                    "12",
                    "1999 at Texas Tech University",
                    "2006",
                ],
            ],
            "id": "1-10054296-1",
        },
        "sql": {
            "human_readable": "SELECT Member FROM table WHERE Chapters = 12",
            "sel": 0,
            "agg": 0,
            "conds": {"column_index": [3], "operator_index": [0], "condition": ["12"]},
        },
    },
    {
        "phase": 1,
        "question": "What are all the typhus number when smallpox is 4",
        "table": {
            "header": [
                "Year",
                "Typhus",
                "Typhoid fever",
                "Relapsing fever",
                "Smallpox",
                "Malaria",
            ],
            "types": ["text", "real", "real", "real", "real", "text"],
            "rows": [
                ["1913", "120", "424", "30", "67", "3600"],
                ["1918\u201322", "1300", "293", "639", "106", "2940 (avg)"],
            ],
            "id": "1-1007688-1",
        },
        "sql": {
            "human_readable": "SELECT Typhus FROM table WHERE Smallpox = 4",
            "sel": 1,
            "agg": 0,
            "conds": {"column_index": [4], "operator_index": [0], "condition": ["4"]},
        },
    },
    {
        "phase": 1,
        "question": (
            "what is the player who's college is listed as direct to nba, school is st. "
            "vincent \u2013 st. mary high school and the year is 2001-2002?"
        ),
        "table": {
            "header": ["Year", "Player", "School", "Hometown", "College", "NBA Draft"],
            "types": ["text", "text", "text", "text", "text", "text"],
            "rows": [
                [
                    "1982-1983",
                    "Reggie Williams",
                    "Dunbar High School",
                    "Baltimore, MD",
                    "Georgetown",
                    "1st Round - 4th Pick of 1987 ( Clippers )",
                ],
                [
                    "2001-2002",
                    "LeBron James",
                    "St. Vincent \u2013 St. Mary High School",
                    "Akron, OH",
                    "Direct to NBA",
                    "1st Round - 1st Pick of 2003 ( Cavaliers )",
                ],
            ],
            "id": "1-11677760-1",
        },
        "sql": {
            "human_readable": (
                "SELECT Player FROM table WHERE College = Direct to NBA AND School = St. Vincent"
                " \u2013 St. Mary High School AND Year = 2001-2002"
            ),
            "sel": 1,
            "agg": 0,
            "conds": {
                "column_index": [4, 2, 0],
                "operator_index": [0, 0, 0],
                "condition": [
                    "Direct to NBA",
                    "St. Vincent \u2013 St. Mary High School",
                    "2001-2002",
                ],
            },
        },
    },
]


def test_real_rows_load_render_and_execute() -> None:
    examples, report = examples_from_rows(REAL_ROWS)
    assert (report.kept, sum(report.skipped.values())) == (6, 0)
    assert [e.sql for e in examples] == [
        'SELECT "Notes" FROM t WHERE "Current slogan" = \'SOUTH AUSTRALIA\'',
        'SELECT COUNT("Fleet Series (Quantity)") FROM t WHERE "Fuel Propulsion" = \'CNG\'',
        "SELECT \"No.\" FROM t WHERE \"Player\" = 'Shaquille O''Neal'",
        'SELECT "Member" FROM t WHERE "Chapters" = 12',
        'SELECT "Typhus" FROM t WHERE "Smallpox" = 4',
        'SELECT "Player" FROM t WHERE "College" = \'Direct to NBA\' AND "School" ='
        " 'St. Vincent \u2013 St. Mary High School' AND \"Year\" = '2001-2002'",
    ]
    assert [e.agg for e in examples] == ["", "COUNT", "", "", "", ""]
    assert [e.n_conds for e in examples] == [1, 1, 1, 1, 1, 3]
    # Row 70's condition matches no row of the trimmed table; the others each match one.
    results = [_execute(e) for e in examples]
    assert results == [
        [("No slogan on current series",)],
        [(1,)],
        [(13.0,)],
        [("Men of God",)],
        [],
        [("LeBron James",)],
    ]
