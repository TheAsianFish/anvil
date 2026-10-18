"""The `make eval-extract` harness end to end, offline: toy queries stand in for a dataset."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from anvil.align import align_mentions
from anvil.errors import ConfigError
from anvil.evaluate.extraction import entity_scores
from anvil.settings import Settings
from anvil.spans import Mention, Span, decode_answer, encode_answer
from anvil.store import Split, read_inputs
from anvil.teacher.base import Label
from anvil.teacher.fake import FakeTeacher
from anvil.teacher.labelfile import write_label_file
from eval import classify, extract, render
from eval.data import Sample, TaggedRef, bio_to_spans, load_tagged_rows, sample_tagged
from eval.extract import EvalConfig, alignment_stress, gazetteer, load_config
from tests.conftest import ROOT

TOY_DIR = ROOT / "examples" / "toy-extract"


def test_bio_to_spans_follows_conll_chunking() -> None:
    text, spans = bio_to_spans(
        ["cheap", "pizza", "hut", "near", "new", "york"],
        ["B-Price", "B-Restaurant_Name", "I-Restaurant_Name", "O", "I-Location", "I-Location"],
    )
    assert text == "cheap pizza hut near new york"
    assert [(s.text_in(text), s.type) for s in spans] == [
        ("cheap", "Price"),
        ("pizza hut", "Restaurant_Name"),
        ("new york", "Location"),  # I after O opens a chunk
    ]
    with pytest.raises(ConfigError, match="2 words but 1 tags"):
        bio_to_spans(["a", "b"], ["O"])


def test_bio_to_spans_opens_a_chunk_for_i_after_another_type() -> None:
    text, spans = bio_to_spans(["thai", "pizza", "hut"], ["B-Cuisine", "I-Name", "I-Name"])
    assert spans == [Span(0, 4, "Cuisine"), Span(5, 14, "Name")]
    assert text[5:14] == "pizza hut"


@pytest.mark.parametrize(
    ("words", "tags", "match"),
    [
        (["john", "smith"], ["PER", "PER"], "tag 'PER' at word 0"),
        (["new", "york"], ["B-LOC", "E-LOC"], "tag 'E-LOC' at word 1"),
        (["paris"], ["S-LOC"], "tag 'S-LOC' at word 0"),
        (["paris"], ["B-"], "tag 'B-' at word 0"),
        (["a", "", "b"], ["O", "O", "O"], "an empty word at position 1"),
    ],
)
def test_bio_to_spans_rejects_tags_outside_bio(
    words: list[str], tags: list[str], match: str
) -> None:
    with pytest.raises(ConfigError, match=f"row 7 has {match}"):
        bio_to_spans(words, tags, row=7)


def test_sample_tagged_rejects_gold_types_the_spec_does_not_list() -> None:
    rows = [
        (["thai", "food"], ["B-Cuisine", "O"]),
        (["at", "nine"], ["O", "B-Hours"]),
        (["by", "the", "sea"], ["O", "O", "B-View"]),
    ]
    with pytest.raises(ConfigError, match="types the spec does not list: Hours, View"):
        sample_tagged("ds", rows, n=3, seed=0, types=["Cuisine"])
    # A tag scheme the loader cannot read names the source row.
    with pytest.raises(ConfigError, match="row 0 has tag 'PER'"):
        sample_tagged("ds", [(["bob"], ["PER"])], n=1, seed=0, types=["PER"])


def test_load_tagged_rows_reads_the_pinned_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import huggingface_hub

    (tmp_path / "label.json").write_text(
        json.dumps({"O": 0, "B-Dish": 1, "I-Dish": 2}), encoding="utf-8"
    )
    lines = [
        json.dumps({"tokens": ["pad", "thai"], "tags": [1, 2]}),
        "",
        "   ",
        json.dumps({"tokens": ["now"], "tags": [0]}),
    ]
    (tmp_path / "train.json").write_text("\n".join(lines) + "\n", encoding="utf-8")
    calls: list[tuple[str, str, str | None, str | None]] = []

    def download(
        repo_id: str, filename: str, repo_type: str | None = None, revision: str | None = None
    ) -> str:
        calls.append((repo_id, filename, repo_type, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    ref = TaggedRef("org/ds", "a" * 40, "train.json", "label.json")
    assert load_tagged_rows(ref) == [(["pad", "thai"], ["B-Dish", "I-Dish"]), (["now"], ["O"])]
    assert sorted(calls) == [
        ("org/ds", "label.json", "dataset", "a" * 40),
        ("org/ds", "train.json", "dataset", "a" * 40),
    ]


def test_sample_tagged_is_seeded_distinct_and_answers_are_exact() -> None:
    rows = [
        (["thai", "food", "nearby"], ["B-Cuisine", "O", "B-Location"]),
        (["thai", "food", "nearby"], ["B-Cuisine", "O", "B-Location"]),  # a duplicate
        (["is", "nobu", "open"], ["O", "B-Restaurant_Name", "O"]),
        ([], []),
    ]
    types = ["Cuisine", "Location", "Restaurant_Name"]
    data = sample_tagged("ds", rows, n=5, seed=1, types=types)
    assert data == sample_tagged("ds", rows, n=5, seed=1, types=types)
    assert len(data.items) == 2
    for it in data.items:
        mentions = decode_answer(data.gold[it.id])
        aligned = align_mentions(it.text, mentions, types)
        assert aligned.report.matched["exact"] == len(mentions)


def test_shipped_config_is_valid() -> None:
    config = load_config(extract.CONFIG_DIR / "mit_restaurant.yaml")
    assert config.spec.kind == "extract"
    assert len(config.spec.labels) == 8
    assert config.spec.teacher.model == "claude-haiku-5-5"
    assert config.paraphrase_spec().prompt_hash != config.spec.prompt_hash
    assert config.sample.pool_labels + config.sample.heldout_labels == config.sample.size


def test_gazetteer_matches_memorised_surface_forms() -> None:
    train_texts = ["thai food in boston", "Thai near me", "pad thai please"]
    train_spans = [
        [Span(0, 4, "Cuisine"), Span(13, 19, "Location")],
        [Span(0, 4, "Cuisine"), Span(5, 12, "Location")],
        [Span(0, 8, "Dish")],
    ]
    types = ["Cuisine", "Dish", "Location"]
    (predicted,) = gazetteer(train_texts, train_spans, ["THAI or pad thai in boston"], types)
    assert predicted == [Span(0, 4, "Cuisine"), Span(8, 16, "Dish"), Span(20, 26, "Location")]
    assert gazetteer([], [], ["anything"], types) == [[]]
    # Whole words only: "thailand" is not "thai".
    assert gazetteer(train_texts, train_spans, ["thailand"], types) == [[]]


def test_alignment_stress_shows_what_repair_recovers() -> None:
    texts = ["cheap pizza near me", "is nobu open late tonight"]
    gold = [
        [Span(0, 5, "Price"), Span(6, 11, "Dish")],
        [Span(3, 7, "Name"), Span(13, 25, "Hours")],
    ]
    types = ["Price", "Dish", "Name", "Hours"]
    study = alignment_stress(texts, gold, types, seed=0)
    rows = study["corruptions"]
    assert isinstance(rows, dict)
    assert study["spans"] == 4
    assert rows["none"]["changed"] == 0
    assert rows["off_by_one"]["changed"] == 4
    assert rows["off_by_one"]["verify_only"]["recovered"] == 0
    # "cheap" starts at 0, so offset zero leaves it alone and it is not scored.
    assert rows["offset_zero"]["changed"] == 3
    for name, row in rows.items():
        repair = row["repair"]
        assert repair["recovered"] == row["changed"], name
        assert repair["recovered"] + repair["misplaced"] + repair["lost"] == row["changed"]
    assert alignment_stress(texts, gold, types, seed=0) == study  # deterministic


def _toy_config() -> tuple[EvalConfig, Sample]:
    raw_spec = yaml.safe_load((TOY_DIR / "anvil.yaml").read_text(encoding="utf-8"))
    raw_spec["teacher"]["model"] = "claude-haiku-5-5"
    raw_spec["student"].update({"epochs": 4, "max_length": 24})
    raw = {
        "dataset": {
            "name": "toy",
            "repo": "local/toy",
            "revision": "0" * 40,
            "filename": "train.json",
            "label_map": "label.json",
        },
        "sample": {"size": 240, "pool_labels": 160, "heldout_labels": 60, "seed": 0},
        "ceiling": {"n": 30, "instructions": "Mark the search filters in this query."},
        "spec": raw_spec,
    }
    config = EvalConfig.model_validate(raw)
    assert config.spec.fake is not None
    exact = FakeTeacher(
        config.spec.model_copy(update={"fake": config.spec.fake.model_copy(update={"noise": 0})})
    )
    items = read_inputs(TOY_DIR / "inputs.jsonl")[:240]
    gold = {it.id: encode_answer(exact.mentions(it)) for it in items}
    return config, Sample(items, gold, len(items))


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        home=tmp_path / "home", teacher_workers=1, record_dir=None, log_level="WARNING"
    )


def test_gold_run_writes_a_complete_report_and_renders(tmp_path: Path) -> None:
    config, data = _toy_config()
    out = tmp_path / "results" / "extract-toy-gold"
    report = extract.run(config, data, "gold", out, _settings(tmp_path), n_resamples=100)

    on_disk = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(report))
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert {"git_sha", "seed", "wall_seconds", "spent_usd", "versions"} <= set(meta)
    assert (out / "train-metrics.jsonl").exists()

    ag, teacher = on_disk["agreement"], on_disk["teacher"]
    assert teacher["model"] == "gold"
    assert teacher["pool_labels"] == 160
    assert teacher["heldout_labels"] == ag["n"] <= 60
    ci = ag["entity_f1_ci"]
    assert ci["low"] <= ag["entity"]["f1"] <= ci["high"]
    assert ci["n_resamples"] == 100
    assert "skipped" in on_disk["ceiling"]
    assert "teacher_vs_gold" not in on_disk["gold"]
    # Gold is the teacher, so student-vs-gold is the headline again.
    assert on_disk["gold"]["student_vs_gold"]["entity"]["f1"] == pytest.approx(ag["entity"]["f1"])
    assert on_disk["alignment"]["pool"]["unaligned_rate"] == 0.0

    # The per-item predictions reproduce the headline and the gazetteer's score.
    pred = on_disk["predictions"]

    def spans(rows: list[list[list[object]]]) -> list[list[Span]]:
        return [[Span(int(str(a)), int(str(b)), str(c)) for a, b, c in row] for row in rows]

    assert entity_scores(spans(pred["teacher"]), spans(pred["student"])).f1 == pytest.approx(
        ag["entity"]["f1"]
    )
    gaz = on_disk["baselines"]["gazetteer"]
    assert entity_scores(spans(pred["teacher"]), spans(pred["gazetteer"])).f1 == pytest.approx(
        gaz["entity"]["f1"]
    )
    diff = gaz["student_minus_gazetteer_f1"]
    assert diff["estimate"] == pytest.approx(ag["entity"]["f1"] - gaz["entity"]["f1"])
    stress = on_disk["alignment_stress"]["corruptions"]
    assert stress["off_by_one"]["repair"]["recovered_rate"] == 1.0

    table = render.render_extract(render.load_runs(tmp_path / "results", prefix="extract-"))
    assert "| toy | gold | tiny | 160 |" in table
    assert "| off_by_one |" in table
    assert "| none |" not in table  # nothing changed, so no row
    assert "not a parity-with-Claude number" in table


def _recorded(config: EvalConfig, data: Sample, paraphrase: bool) -> list[Label]:
    """A "recorded" Claude: the toy gold answers, offsets off by one on every third input,
    and on the paraphrase every fifth input's first mention dropped."""
    spec = config.paraphrase_spec() if paraphrase else config.spec
    rows = []
    for i, it in enumerate(data.items):
        mentions = decode_answer(data.gold[it.id])
        if not paraphrase and i % 3 == 0:
            mentions = [Mention(m.text, m.type, m.start + 1) for m in mentions]
        if paraphrase and i % 5 == 0:
            mentions = mentions[1:]
        rows.append(
            Label(it.id, encode_answer(mentions), spec.teacher.model, spec.prompt_hash, 400, 30)
        )
    return rows


def test_replay_run_measures_the_ceiling_and_teacher_vs_gold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config()
    monkeypatch.setattr(classify, "LABELS_DIR", tmp_path / "labels")
    spec, para = config.spec, config.paraphrase_spec()
    model = spec.teacher.model
    write_label_file(
        classify.label_file("toy", model, spec.prompt_hash), _recorded(config, data, False)
    )
    write_label_file(
        classify.label_file("toy", model, para.prompt_hash), _recorded(config, data, True)
    )
    settings = _settings(tmp_path)
    report = extract.run(config, data, "auto", tmp_path / "out", settings, n_resamples=100)
    teacher = report["teacher"]
    assert isinstance(teacher, dict)
    assert teacher["kind"] == "replay"
    assert report["spent_usd"] == 0

    # Shifted offsets are repaired, so the teacher agrees with gold exactly.
    gold = report["gold"]
    assert isinstance(gold, dict)
    assert gold["teacher_vs_gold"]["entity"]["f1"] == 1.0
    alignment = report["alignment"]
    assert isinstance(alignment, dict)
    assert alignment["heldout"]["matched"]["shifted"] > 0

    ceiling = report["ceiling"]
    assert isinstance(ceiling, dict)
    ws = extract.workspace(settings, spec, data)
    with ws.open_store(spec) as store:
        heldout = store.labelled(model, spec.prompt_hash, Split.HELDOUT)
    assert ceiling["n"] == min(config.ceiling.n, len(heldout))
    # The first labelling repairs to gold; the paraphrase drops the first mention of every
    # fifth input, so precision is 1 and recall is what is left of the subset's spans.
    position = {it.id: i for i, it in enumerate(data.items)}
    subset = [it for it, _ in heldout[: config.ceiling.n]]
    total = sum(len(decode_answer(data.gold[it.id])) for it in subset)
    dropped = sum(
        1 for it in subset if position[it.id] % 5 == 0 and decode_answer(data.gold[it.id])
    )
    assert dropped > 0
    recall = (total - dropped) / total
    assert ceiling["entity_f1"] == pytest.approx(2 * recall / (1 + recall))

    table = render.render_extract(render.load_runs(tmp_path, prefix="out"))
    assert "claude-haiku-5-5" in table


def test_main_sends_override_runs_to_a_smoke_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, data = _toy_config()
    seen: dict[str, object] = {}

    def fake_run(
        cfg: EvalConfig, d: Sample, kind: str, out: Path, *a: object
    ) -> dict[str, object]:
        seen["out"], seen["epochs"] = out, cfg.spec.student.epochs
        return {
            "agreement": {
                "n": 1,
                "entity": {"f1": 0.5},
                "entity_f1_ci": {"low": 0.1, "high": 0.9},
                "token_kappa": 0.4,
                "token_kappa_ci": {"low": 0.1, "high": 0.8},
            },
            "teacher": {"model": "gold", "pool_labels": 1},
            "dataset": {"name": "mit_restaurant"},
        }

    monkeypatch.setattr(extract, "load_tagged_rows", lambda ref: [])
    monkeypatch.setattr(extract, "sample_tagged", lambda *a, **k: data)
    monkeypatch.setattr(extract, "run", fake_run)
    assert extract.main(["mit_restaurant", "--teacher", "gold", "--epochs", "1"]) == 0
    assert str(seen["out"]).endswith("extract-mit_restaurant-gold-smoke")
    assert seen["epochs"] == 1
    assert "entity F1 0.500" in capsys.readouterr().out


def test_gazetteer_survives_case_folds_that_lower_does_not_undo() -> None:
    (predicted,) = gazetteer(["sushi bar"], [[Span(0, 5, "Dish")]], ["\u017fushi now"], ["Dish"])
    assert predicted == [Span(0, 5, "Dish")]


def test_stress_counts_an_exact_match_of_a_corrupted_text_as_misplaced() -> None:
    texts = ["is it open late."]
    gold = [[Span(11, 15, "Hours")]]
    row = alignment_stress(texts, gold, ["Hours"], seed=0)["corruptions"]
    assert isinstance(row, dict)
    stop = row["trailing_full_stop"]
    assert stop["changed"] == 1
    assert stop["repair"]["misplaced"] == 1  # "late." is in the text, so it matches as is


def test_stress_attributes_each_aligned_span_to_one_gold_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "pancakes and more pancakes"
    gold = [[Span(0, 8, "Dish"), Span(18, 26, "Dish")]]

    def vanish_second(text: str, m: Mention, rng: object) -> Mention:
        # The first mention is untouched, the second names words the sentence lacks.
        return m if m.start == 0 else Mention("waffles", m.type, m.start)

    def shift_first_vanish_second(text: str, m: Mention, rng: object) -> Mention:
        return Mention(m.text, m.type, 1) if m.start == 0 else Mention("waffles", m.type, 0)

    monkeypatch.setattr(
        extract,
        "CORRUPTIONS",
        {"vanish": vanish_second, "shift_and_vanish": shift_first_vanish_second},
    )
    rows = alignment_stress([text], gold, ["Dish"], seed=0)["corruptions"]
    assert isinstance(rows, dict)
    # The other "pancakes" is aligned, but it is the first gold span's, not the lost one's.
    vanish = rows["vanish"]["repair"]
    assert rows["vanish"]["changed"] == 1
    assert (vanish["recovered"], vanish["misplaced"], vanish["lost"]) == (0, 0, 1)
    both = rows["shift_and_vanish"]["repair"]
    assert rows["shift_and_vanish"]["changed"] == 2
    assert (both["recovered"], both["misplaced"], both["lost"]) == (1, 0, 1)


def test_stress_counts_a_mention_on_another_occurrence_as_misplaced() -> None:
    # Only the second "pancakes" is an entity; offset zero lands it on the first.
    rows = alignment_stress(["pancakes or pancakes"], [[Span(12, 20, "Dish")]], ["Dish"], seed=0)[
        "corruptions"
    ]
    assert isinstance(rows, dict)
    zero = rows["offset_zero"]
    assert zero["changed"] == 1
    assert (zero["repair"]["recovered"], zero["repair"]["misplaced"]) == (0, 1)
    assert (zero["verify_only"]["recovered"], zero["verify_only"]["misplaced"]) == (0, 1)


def test_run_refuses_a_sample_with_no_heldout_sentences(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, data = _toy_config()
    # A labelling that put nothing in the held-out split (as a tiny sample can).
    monkeypatch.setattr(
        extract, "label", lambda *a, **k: SimpleNamespace(spent_usd=0.0, failures=[])
    )

    def no_training(*a: object, **k: object) -> None:
        raise AssertionError("trained on a run with nothing to score")

    monkeypatch.setattr(extract, "train", no_training)
    with pytest.raises(ConfigError, match=r"none in the held-out split.*raise --size"):
        extract.run(config, data, "gold", tmp_path / "out", _settings(tmp_path))
    assert not (tmp_path / "out").exists()


def test_override_validates_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config = load_config(extract.CONFIG_DIR / "mit_restaurant.yaml")
    with pytest.raises(ConfigError, match="epochs"):
        extract._override(config, None, -3, None)
    with pytest.raises(ConfigError, match="size"):
        extract._override(config, None, None, -5)
    assert extract._override(config, None, 2, None).spec.student.epochs == 2

    def unreachable(*a: object, **k: object) -> None:
        raise AssertionError("the dataset was loaded for an invalid override")

    monkeypatch.setattr(extract, "load_tagged_rows", unreachable)
    assert extract.main(["mit_restaurant", "--teacher", "gold", "--epochs", "-3"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: invalid override")
    assert "Traceback" not in err
