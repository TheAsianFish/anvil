"""The `make eval-generate` harness end to end, offline: hand-made tables stand in for WikiSQL."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from anvil.errors import ConfigError
from anvil.evaluate.judge import Verdict, judge_sample, rubric_hash, write_verdict_file
from anvil.evaluate.sql import Table
from anvil.settings import Settings
from anvil.teacher.base import Item
from eval import generate, render
from eval.generate import (
    EvalConfig,
    ExportCfg,
    JudgeCfg,
    _judge,
    _override,
    load_config,
    rescore,
    run,
    score_pair,
    slices,
)
from eval.wikisql import LoadReport, SqlExample, gold_sql, render_input, sample_examples

PLAYERS = ["Ann", "Bo", "Cy", "Di", "Ed", "Flo", "Gus", "Hal", "Ivy", "Jo"]
TEAMS = ["Reds", "Blues", "Greens"]
CITIES = ["Oslo", "Lima", "Kyiv", "Nice", "Bern", "Graz", "Pisa", "Faro", "Lund", "Riga"]
COUNTRIES = ["Norway", "Peru", "France"]
P_HEADER, P_TYPES = ["Player", "No.", "Team"], ["text", "real", "text"]
C_HEADER, C_TYPES = ["City", "Population", "Country"], ["text", "real", "text"]
P_ROWS = [[p, str(3 * i + 1), TEAMS[i % 3]] for i, p in enumerate(PLAYERS)]
C_ROWS = [[c, str(1000 * (i + 1)), COUNTRIES[i % 3]] for i, c in enumerate(CITIES)]
SIZE = 64


def _example(
    n: int,
    question: str,
    table: str,
    sel: int,
    agg: int = 0,
    conds: list[tuple[int, int, str]] | None = None,
) -> SqlExample:
    header, types, rows = (
        (P_HEADER, P_TYPES, P_ROWS) if table == "players" else (C_HEADER, C_TYPES, C_ROWS)
    )
    conds = conds or []
    return SqlExample(
        id=f"wikisql-{n}",
        question=question,
        header=header,
        types=types,
        rows=rows,
        sql=gold_sql(header, types, sel, agg, conds),
        agg=["", "MAX", "MIN", "COUNT", "SUM", "AVG"][agg],
        n_conds=len(conds),
    )


def fake_wikisql() -> list[SqlExample]:
    """About 70 templated questions over two small tables, with WikiSQL's query shapes."""
    specs: list[tuple[str, str, int, int, list[tuple[int, int, str]]]] = []
    for i, p in enumerate(PLAYERS):
        team = TEAMS[i % 3]
        specs.append((f"What team does {p} play for?", "players", 2, 0, [(0, 0, p)]))
        specs.append((f"What number does {p} wear?", "players", 1, 0, [(0, 0, p)]))
        specs.append(
            (f"Which player on the {team} is {p}?", "players", 0, 0, [(0, 0, p), (2, 0, team)])
        )
    for c in CITIES:
        specs.append((f"What is the population of {c}?", "cities", 1, 0, [(0, 0, c)]))
        specs.append((f"Which country is {c} in?", "cities", 2, 0, [(0, 0, c)]))
    for team in TEAMS:
        specs.append((f"How many players are on the {team}?", "players", 0, 3, [(2, 0, team)]))
        specs.append((f"Highest number on the {team}?", "players", 1, 1, [(2, 0, team)]))
    for country in COUNTRIES:
        specs.append((f"Largest city population in {country}?", "cities", 1, 1, [(2, 0, country)]))
        specs.append((f"How many cities are in {country}?", "cities", 0, 3, [(2, 0, country)]))
    for n in (5, 10, 15, 20):
        specs.append((f"Who wears a number above {n}?", "players", 0, 0, [(1, 1, str(n))]))
    specs.append(("List every player.", "players", 0, 0, []))
    specs.append(("List every city.", "cities", 0, 0, []))
    return [_example(k, q, t, sel, agg, conds) for k, (q, t, sel, agg, conds) in enumerate(specs)]


def _config(judge_n: int = 4) -> EvalConfig:
    config = _override(
        load_config(generate.CONFIG_DIR / "wikisql.yaml"),
        "tiny",
        1,
        SIZE,
        baseline=True,
        max_length=1200,
    )
    return config.model_copy(update={"judge": JudgeCfg(n=judge_n), "export": ExportCfg(probes=4)})


# --- the whole run -------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gold_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], Path]:
    root = tmp_path_factory.mktemp("eval-generate")
    settings = Settings(
        home=root / "home", teacher_workers=1, record_dir=None, log_level="WARNING"
    )
    loaded = fake_wikisql()
    assert len(loaded) >= SIZE
    data, examples = sample_examples(loaded, SIZE, 0)
    out = root / "results"
    report = run(
        _config(),
        data,
        examples,
        "gold",
        out,
        settings,
        n_resamples=50,
        command="test",
        provenance={"git_sha": "x", "git_dirty": False},
    )
    return report, out


def test_run_writes_its_raw_output(gold_run: tuple[dict[str, object], Path]) -> None:
    report, out = gold_run
    assert json.loads((out / "report.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(report)
    )
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["command"] == "test"
    assert meta["git_sha"] == "x"
    assert meta["teacher_model"] == "gold"
    assert meta["student_model"] == "tiny"
    assert meta["n_resamples"] == 50
    assert meta["spent_usd"] == 0.0
    assert (out / "train-metrics.jsonl").read_text(encoding="utf-8").strip()


def test_run_report_has_every_section(gold_run: tuple[dict[str, object], Path]) -> None:
    report, _ = gold_run
    teacher = report["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["kind"] == "gold"
    assert teacher["model"] == "gold"
    n = teacher["heldout_labels"]
    assert isinstance(n, int)
    assert n > 0

    ag = report["agreement"]
    assert isinstance(ag, dict)
    assert ag["n"] == n
    em = ag["execution_match"]
    assert 0.0 <= em["low"] <= em["estimate"] <= em["high"] <= 1.0
    assert em["n_resamples"] == 50
    for key in ("exact_match", "nonempty_reference", "candidate_errors", "reference_empty"):
        assert key in ag
    assert ag["reference_errors"] == 0  # every gold query runs on its table

    sl = report["slices"]
    assert isinstance(sl, dict)
    assert set(sl) == {"aggregate", "conditions", "length"}
    for rows in sl.values():
        assert sum(r["n"] for r in rows) == n

    zs = report["baselines"]["zero_shot"]  # type: ignore[index]
    assert "student_minus_zero_shot" in zs
    assert zs["n"] == n
    assert report["ceiling"] == {
        "skipped": "gold queries are fixed, so the teacher agrees with itself"
    }
    gold = report["gold"]
    assert isinstance(gold, dict)
    assert gold["student_vs_gold"]["n"] == n
    assert "teacher_vs_gold" not in gold

    judged = report["judge"]
    assert isinstance(judged, dict)
    assert "no ANTHROPIC_API_KEY" in judged["skipped"]
    assert 0 < len(judged["sample"]) <= 4

    export = report["export"]
    assert isinstance(export, dict)
    assert export["parity"]["passed"] is True
    assert export["parity"]["probes"] == 4
    assert export["adapter_parameters"] > 0

    preds = report["predictions"]
    assert isinstance(preds, dict)
    for key in ("input_id", "teacher", "student", "zero_shot", "execution_match"):
        assert len(preds[key]) == n, key
    assert report["spent_usd"] == 0.0


def test_summary_line_names_the_headline(gold_run: tuple[dict[str, object], Path]) -> None:
    report, out = gold_run
    line = generate.summary_line(report, out)
    assert line.startswith("wikisql: gold -> student: execution match ")
    assert str(out) in line


# --- scoring -------------------------------------------------------------------------------

TABLE = Table(P_HEADER, P_TYPES, [["Ann", "21", "Reds"], ["Bo", "7", "Blues"]])


def test_score_pair_counts_each_kind_of_outcome() -> None:
    ann = 'SELECT "Team" FROM t WHERE "Player" = \'Ann\''
    refs = [ann, ann, ann, 'SELECT "Team" FROM t WHERE "Player" = \'Zed\'']
    cands = [
        'select  "Team" from t where "Player" = \'Ann\';',  # the same query, respelled
        'SELECT "Team" FROM t WHERE "No." = 21',  # the same rows by coincidence
        "SELEC nonsense",  # not SQL
        'SELECT "Team" FROM t WHERE "Player" = \'Nobody\'',  # empty, like the reference
    ]
    stats, matches = score_pair([TABLE] * 4, refs, cands, n_resamples=50, seed=0)
    assert [m.match for m in matches] == [True, True, False, True]
    assert stats["n"] == 4
    assert stats["execution_match"]["estimate"] == 0.75  # type: ignore[index]
    assert stats["exact_match"] == 0.25
    assert stats["executed_match_not_exact"] == 2  # the coincidence and the empty pair
    assert stats["exact_but_not_executed_match"] == 0
    assert stats["candidate_errors"] == {"OperationalError": 1}
    assert stats["reference_empty"] == 1
    assert stats["reference_errors"] == 0
    assert stats["nonempty_reference"] == 3
    assert stats["execution_match_nonempty"] == pytest.approx(2 / 3)


def test_score_pair_coincidence_is_a_match_but_not_exact() -> None:
    ref = 'SELECT "Team" FROM t WHERE "Player" = \'Ann\''
    stats, _ = score_pair([TABLE], [ref], ['SELECT "Team" FROM t WHERE "No." > 10'], 50, 0)
    assert stats["execution_match"]["estimate"] == 1.0  # type: ignore[index]
    assert stats["exact_match"] == 0.0
    assert stats["executed_match_not_exact"] == 1


def test_score_pair_with_only_empty_references_has_no_nonempty_rate() -> None:
    ref = 'SELECT "Team" FROM t WHERE "Player" = \'Zed\''
    stats, _ = score_pair([TABLE], [ref], [ref], 50, 0)
    assert stats["nonempty_reference"] == 0
    assert stats["execution_match_nonempty"] is None


def test_slices_on_a_hand_example() -> None:
    exs = [
        _example(0, "a", "players", 0),
        _example(1, "b", "players", 0, 3, [(2, 0, "Reds")]),
        _example(2, "c", "players", 0, 0, [(2, 0, "Reds"), (1, 1, "3")]),
        _example(3, "d", "players", 1, 1, [(2, 0, "Reds"), (1, 1, "3"), (1, 2, "9"), (0, 0, "x")]),
    ]
    texts = ["w " * 2, "w " * 4, "w " * 6, "w " * 8]  # 2, 4, 6 and 8 words
    got = slices(exs, texts, [True, False, True, True])
    assert got["aggregate"] == [
        {"aggregate": "none", "n": 2, "execution_match": 1.0},
        {"aggregate": "aggregate", "n": 2, "execution_match": 0.5},
    ]
    assert got["conditions"] == [
        {"conditions": "0", "n": 1, "execution_match": 1.0},
        {"conditions": "1", "n": 1, "execution_match": 0.0},
        {"conditions": "2", "n": 1, "execution_match": 1.0},
        {"conditions": "3+", "n": 1, "execution_match": 1.0},
    ]
    # Quartiles 3.5, 5 and 6.5 truncate to the edges 3, 5 and 6.
    assert [(r["words"], r["n"]) for r in got["length"]] == [
        ("<= 3", 1),
        ("4-5", 1),
        ("6-6", 1),
        ("> 6", 1),
    ]


# --- the judge -----------------------------------------------------------------------------


def _judge_inputs() -> tuple[list[tuple[Item, str]], list[str], list[str]]:
    ids = ["wikisql-1", "wikisql-2", "wikisql-3"]
    ann = 'SELECT "Team" FROM t WHERE "Player" = \'Ann\''
    heldout = [(Item(i, render_input(P_HEADER, P_TYPES, f"q {i}")), "{}") for i in ids]
    refs = [ann, ann, ann]
    cands = [ann, 'SELECT "Team" FROM t WHERE "No." = 21', 'SELECT "Team" FROM t']
    return heldout, refs, cands


def test_judge_replays_recorded_verdicts_for_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(generate, "LABELS_DIR", tmp_path / "labels")
    config = _config(judge_n=3)
    path = generate.judge_file(config.dataset.name, config.judge.model)
    assert path.is_relative_to(tmp_path)
    verdicts = [
        Verdict("wikisql-1", True, "same", 100, 10, 0.001),
        Verdict("wikisql-2", False, "different filter", 100, 10, 0.001),
        Verdict("wikisql-3", False, "no filter", 100, 10, 0.001),
    ]
    write_verdict_file(path, verdicts, config.judge.model, rubric_hash())
    heldout, refs, cands = _judge_inputs()
    _, matches = score_pair([TABLE] * 3, refs, cands, 50, 0)
    assert [m.match for m in matches] == [True, True, False]

    stats, spent = _judge(config, settings, heldout, refs, cands, matches)
    assert spent == 0.0
    assert stats["spent_usd"] == 0.0
    assert stats["model"] == config.judge.model
    assert stats["rubric_hash"] == rubric_hash()
    vs = stats["vs_execution_match"]
    assert isinstance(vs, dict)
    assert (vs["judged"], vs["both_match"], vs["exec_only"], vs["judge_only"], vs["neither"]) == (
        3,
        1,
        1,
        0,
        1,
    )
    assert [v["cached"] for v in stats["verdicts"]] == [True] * 3  # type: ignore[attr-defined]


def test_judge_without_key_or_recording_is_skipped(settings: Settings) -> None:
    config = _config(judge_n=2)
    assert not generate.judge_file(config.dataset.name, config.judge.model).exists()
    heldout, refs, cands = _judge_inputs()
    _, matches = score_pair([TABLE] * 3, refs, cands, 50, 0)
    stats, spent = _judge(config, settings, heldout, refs, cands, matches)
    assert spent == 0.0
    assert "no ANTHROPIC_API_KEY" in str(stats["skipped"])
    ids = [it.id for it, _ in heldout]
    assert stats["sample"] == judge_sample(ids, 2, config.judge.seed)
    assert "verdicts" not in stats


def test_judge_with_n_zero_is_skipped(settings: Settings) -> None:
    heldout, refs, cands = _judge_inputs()
    _, matches = score_pair([TABLE] * 3, refs, cands, 50, 0)
    stats, spent = _judge(_config(judge_n=0), settings, heldout, refs, cands, matches)
    assert (stats, spent) == ({"skipped": "judge.n is 0"}, 0.0)


def test_judge_skip_reason_with_labels_outside_the_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(generate, "LABELS_DIR", tmp_path / "labels")
    reason = generate._make_judge(_config(), settings)
    assert isinstance(reason, str)
    assert "no ANTHROPIC_API_KEY" in reason


# --- main ----------------------------------------------------------------------------------


def test_main_refuses_out_without_the_smoke_suffix(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as info:
        generate.main(["wikisql", "--student", "tiny", "--out", str(tmp_path / "plain")])
    assert info.value.code == 2


def test_main_auto_without_key_or_labels_is_a_config_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(generate, "load_examples", lambda ref: (fake_wikisql(), LoadReport()))
    monkeypatch.setattr(generate, "git_provenance", lambda: {"git_sha": "x", "git_dirty": False})
    out = tmp_path / "run-smoke"
    code = generate.main(["wikisql", "--teacher", "auto", "--size", str(SIZE), "--out", str(out)])
    assert code == 2
    err = capsys.readouterr().err
    assert "no recorded labels" in err
    assert "--teacher gold" in err
    assert not out.exists()


def test_override_rejects_an_invalid_value() -> None:
    with pytest.raises(ConfigError, match="invalid override"):
        _override(load_config(generate.CONFIG_DIR / "wikisql.yaml"), None, None, 5, True)


def test_doubled_rows_expose_a_missing_aggregate_but_not_an_equivalent_query() -> None:
    table = Table(["Prod #", "Episode #"], ["text", "real"], [["a1", "1"], ["b2", "2"]])
    ref = 'SELECT MAX("Prod #") FROM t WHERE "Episode #" = 1'
    stats, _ = generate.score_pair(
        [table, table],
        [ref, ref],
        [
            'SELECT "Prod #" FROM t WHERE "Episode #" = 1',
            'select max("Prod #") from t where "Episode #" = 1.0',
        ],
        50,
        0,
    )
    assert stats["execution_match"]["estimate"] == 1.0  # type: ignore[index]
    assert stats["execution_match_doubled"]["estimate"] == 0.5  # type: ignore[index]
    assert stats["coincidental_matches"] == 1
    assert generate.doubled(table).rows == [*table.rows, *table.rows]


# --- rescoring a finished run ----------------------------------------------------------------


def _copy_run(out: Path, tmp_path: Path) -> Path:
    copy = tmp_path / "results"
    shutil.copytree(out, copy)
    return copy


def test_rescore_with_the_same_scorer_changes_nothing_but_meta(
    gold_run: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    report, out = gold_run
    copy = _copy_run(out, tmp_path)
    data, examples = sample_examples(fake_wikisql(), SIZE, 0)
    again = rescore(copy, data, examples, "again", {"git_sha": "y", "git_dirty": True})
    assert again == json.loads(json.dumps(report))
    meta = json.loads((copy / "meta.json").read_text(encoding="utf-8"))
    assert meta["command"] == "test"
    assert meta["rescored"]["command"] == "again"
    assert meta["rescored"]["git_sha"] == "y"
    text = render.render_generate([("generate-toy-gold", again, meta)])
    assert "- Rescored: `again` at `y` (dirty tree)" in text
    assert "Rescored" not in render.render_generate([("generate-toy-gold", report, {})])


def test_rescore_recomputes_every_statistic_from_the_stored_queries(
    gold_run: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    report, out = gold_run
    copy = _copy_run(out, tmp_path)
    stale = json.loads((copy / "report.json").read_text(encoding="utf-8"))
    stale["agreement"] = {"n": -1}
    stale["slices"] = {}
    stale["gold"]["student_vs_gold"] = {}
    stale["baselines"]["zero_shot"]["execution_match"] = {}
    stale["predictions"]["execution_match"] = [
        not f for f in stale["predictions"]["execution_match"]
    ]
    (copy / "report.json").write_text(json.dumps(stale), encoding="utf-8")
    data, examples = sample_examples(fake_wikisql(), SIZE, 0)
    again = rescore(copy, data, examples, "again", {"git_sha": "y", "git_dirty": False})
    assert again == json.loads(json.dumps(report))


def test_rescore_without_a_finished_run_is_a_config_error(tmp_path: Path) -> None:
    data, examples = sample_examples(fake_wikisql(), SIZE, 0)
    with pytest.raises(ConfigError, match="nothing to rescore"):
        rescore(
            tmp_path / "absent", data, examples, provenance={"git_sha": "x", "git_dirty": False}
        )


def test_rescore_refuses_questions_outside_the_sample(
    gold_run: tuple[dict[str, object], Path], tmp_path: Path
) -> None:
    _, out = gold_run
    copy = _copy_run(out, tmp_path)
    data, examples = sample_examples(fake_wikisql(), SIZE, 0)
    first = next(iter(examples))
    fewer = {k: v for k, v in examples.items() if k != first}
    stored = json.loads((copy / "report.json").read_text(encoding="utf-8"))
    if first not in stored["predictions"]["input_id"]:
        fewer = {k: v for k, v in examples.items() if k != stored["predictions"]["input_id"][0]}
    with pytest.raises(ConfigError, match="not in the sample"):
        rescore(copy, data, fewer, provenance={"git_sha": "x", "git_dirty": False})
