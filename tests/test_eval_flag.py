"""`eval.flag`: out-of-fold flagging of a teacher's labels, on scripted students, offline."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from eval import flag, render
from eval.noise import Pool
from eval.shift import auroc
from tests.test_eval_fallible import _local_toy, _settings
from tests.test_teacher_ollama import MODEL

CLASSES = ["a", "b", "c", "d"]


class Believer:
    """A scripted student that puts `p` on the label it believes for each text and spreads
    the rest evenly; it never looks at the labels it was trained on."""

    def __init__(self, belief: dict[str, str], classes: Sequence[str], p: float = 0.85) -> None:
        self.belief, self.classes, self.p = belief, list(classes), p

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        k = len(self.classes)
        out = np.full((len(texts), k), (1 - self.p) / (k - 1), dtype=np.float32)
        for row, text in enumerate(texts):
            out[row, self.classes.index(self.belief[text])] = self.p
        return out


# --- folds and the out-of-fold loop -------------------------------------------------------------


def test_fold_ids_are_balanced_seeded_and_cover_every_item() -> None:
    folds = flag.fold_ids(100, 3, seed=0)
    assert sorted(np.bincount(folds).tolist()) == [33, 33, 34]
    assert np.array_equal(folds, flag.fold_ids(100, 3, seed=0))
    assert not np.array_equal(folds, flag.fold_ids(100, 3, seed=1))
    with pytest.raises(ValueError, match="folds"):
        flag.fold_ids(2, 3, seed=0)
    with pytest.raises(ValueError, match="folds"):
        flag.fold_ids(10, 1, seed=0)


def test_out_of_fold_never_scores_an_item_its_student_trained_on() -> None:
    texts = [f"t{i}" for i in range(30)]
    labels = [CLASSES[i % 4] for i in range(30)]
    folds = flag.fold_ids(len(texts), 3, seed=0)
    seen: list[set[str]] = []

    class Recorder:
        def __init__(self, trained: set[str]) -> None:
            self.trained = trained

        def predict_proba(self, scored: Sequence[str]) -> npt.NDArray[np.float32]:
            assert not self.trained & set(scored), "scored an item it trained on"
            return np.full((len(scored), 4), 0.25, dtype=np.float32)

    def fit(train_texts: Sequence[str], train_labels: Sequence[str]) -> flag.Predictor:
        assert len(train_texts) == len(train_labels)
        seen.append(set(train_texts))
        return Recorder(set(train_texts))

    proba = flag.out_of_fold_proba(texts, labels, folds, 4, fit)
    assert len(seen) == 3
    assert all(len(s) == 20 for s in seen)  # each student trains on the other two folds
    assert set().union(*seen) == set(texts)
    assert not np.isnan(proba).any()
    with pytest.raises(ValueError, match="same length"):
        flag.out_of_fold_proba(texts, labels[:-1], folds, 4, fit)


def test_out_of_fold_rejects_a_student_with_the_wrong_number_of_classes() -> None:
    texts = [f"t{i}" for i in range(9)]

    def fit(train_texts: Sequence[str], train_labels: Sequence[str]) -> flag.Predictor:
        return Believer({t: "a" for t in texts}, CLASSES[:3])

    with pytest.raises(ValueError, match="shape"):
        flag.out_of_fold_proba(texts, ["a"] * 9, flag.fold_ids(9, 3, 0), 4, fit)


# --- the score and the flag ---------------------------------------------------------------------


def test_self_confidence_reads_the_given_labels_column() -> None:
    proba = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    assert flag.self_confidence(proba, np.array([0, 0])).tolist() == [0.7, 0.1]


def test_flag_lowest_flags_exactly_the_share_and_breaks_ties_by_position() -> None:
    score = np.array([0.5, 0.1, 0.1, 0.9, 0.1])
    assert flag.flag_lowest(score, 0.4).tolist() == [False, True, True, False, False]
    assert flag.flag_lowest(score, 0.0).sum() == 0
    assert flag.flag_lowest(score, 1.0).all()
    with pytest.raises(ValueError, match="fraction"):
        flag.flag_lowest(score, 1.5)


def _brute_recall_at_precision(score: list[float], wrong: list[bool], precision: float) -> float:
    best = 0.0
    for cut in sorted(set(score)):
        flagged = [s <= cut for s in score]
        hits = sum(f and w for f, w in zip(flagged, wrong, strict=True))
        if hits >= precision * sum(flagged) - 1e-12 and sum(wrong):
            best = max(best, hits / sum(wrong))
    return best


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.tuples(st.integers(0, 6), st.booleans()), min_size=1, max_size=40),
    st.sampled_from([0.5, 0.7, 0.9]),
)
def test_recall_at_precision_matches_every_cut(
    rows: list[tuple[int, bool]], precision: float
) -> None:
    score = [s / 6 for s, _ in rows]
    wrong = [w for _, w in rows]
    got = flag.recall_at_precision(np.array(score), np.array(wrong), precision)
    assert got["recall"] == pytest.approx(_brute_recall_at_precision(score, wrong, precision))
    if got["recall"] > 0:
        assert got["precision"] >= precision - 1e-12
        # The cut never splits tied scores.
        assert got["flagged_share"] * len(score) == pytest.approx(
            sum(s <= got["cut"] for s in score)
        )


def test_auroc_interval_brackets_the_estimate() -> None:
    rng = np.random.default_rng(0)
    wrong = rng.random(300) < 0.3
    suspicion = rng.normal(size=300) + wrong
    out = flag.auroc_ci(suspicion, wrong, 200, seed=0)
    assert out["auroc"] == pytest.approx(auroc(suspicion[wrong], suspicion[~wrong]))
    assert out["low"] < out["auroc"] < out["high"]


# --- what the analysis says about two scripted students -------------------------------------


def _scripted_case() -> tuple[list[str], list[str], list[str]]:
    """60 texts with gold labels and a teacher that is wrong on every fifth input."""
    texts = [f"text {i}" for i in range(60)]
    gold = [CLASSES[i % 4] for i in range(60)]
    teacher = [CLASSES[(i + 1) % 4] if i % 5 == 0 else g for i, g in enumerate(gold)]
    return texts, gold, teacher


def _analyse_believer(belief: dict[str, str]) -> dict[str, object]:
    texts, gold, teacher = _scripted_case()
    index = {c: i for i, c in enumerate(CLASSES)}

    def fit(train_texts: Sequence[str], train_labels: Sequence[str]) -> flag.Predictor:
        return Believer(belief, CLASSES)

    proba = flag.out_of_fold_proba(texts, teacher, flag.fold_ids(60, 3, 0), 4, fit)
    return flag.analyse_flags(
        proba,
        np.array([index[t] for t in teacher]),
        np.array([index[g] for g in gold]),
        n_resamples=100,
    )


def test_a_student_that_sees_through_the_noise_flags_every_wrong_label() -> None:
    texts, gold, _ = _scripted_case()
    out = _analyse_believer(dict(zip(texts, gold, strict=True)))
    assert out["wrong"] == 12
    assert out["error_rate"] == pytest.approx(0.2)
    assert out["auroc"] == {"auroc": 1.0, "low": 1.0, "high": 1.0}
    assert out["at_fraction"] == {"flagged": 12, "hits": 12, "precision": 1.0, "recall": 1.0}
    assert out["best_possible_recall"] == 1.0
    at = out["at_precision"]
    assert isinstance(at, dict)
    assert at["recall"] == 1.0
    assert at["flagged_share"] == pytest.approx(0.2)
    assert out["caught_argmax_is_gold"] == 1.0
    assert out["out_of_fold_accuracy_vs_gold"] == 1.0


def test_a_student_that_learned_the_teachers_errors_flags_nothing_useful() -> None:
    # Errors tied to the inputs: the other folds taught the student the teacher's answer, so
    # it backs every given label and the score cannot tell wrong from right.
    texts, _, teacher = _scripted_case()
    out = _analyse_believer(dict(zip(texts, teacher, strict=True)))
    auc = out["auroc"]
    assert isinstance(auc, dict)
    assert auc["auroc"] == 0.5
    at_fraction = out["at_fraction"]
    assert isinstance(at_fraction, dict)
    # Every score ties, so the flag is the first 12 by position, 3 of them wrong.
    assert at_fraction["precision"] == pytest.approx(3 / 12)
    at = out["at_precision"]
    assert isinstance(at, dict)
    assert at["recall"] == 0.0
    assert out["caught_argmax_is_gold"] == 0.0
    assert out["out_of_fold_accuracy_vs_given"] == 1.0


def test_no_wrong_labels_leaves_the_auroc_undefined() -> None:
    proba = np.full((6, 4), 0.25)
    codes = np.array([0, 1, 2, 3, 0, 1])
    out = flag.analyse_flags(proba, codes, codes, n_resamples=10)
    assert out["auroc"] is None
    assert out["caught_argmax_is_gold"] is None


# --- the run end to end on a scripted trainer ------------------------------------------------


def _toy_pool(tmp_path: Path) -> Pool:
    config, data = _local_toy(tmp_path)
    names = config.spec.label_names
    labelled = []
    for i, it in enumerate(data.items):
        g = data.gold[it.id]
        # The teacher is wrong on every sixth input.
        lab = names[(names.index(g) + 1) % len(names)] if i % 6 == 0 else g
        labelled.append((it, lab))
    return Pool(config, config.spec, data, MODEL, labelled[:160], labelled[160:220])


def test_run_dataset_trains_each_student_once_and_writes_a_reanalysable_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool = _toy_pool(tmp_path)
    gold = pool.data.gold
    calls: list[int] = []

    def scripted_train(
        spec: object, teacher: str, pairs: list[tuple[object, str]], corpus: list[str]
    ) -> tuple[flag.Predictor, None]:
        calls.append(len(pairs))
        by_text = {it.text: gold[it.id] for it in pool.data.items}
        return Believer(by_text, pool.spec.label_names, p=0.6), None

    monkeypatch.setattr(flag, "load_pool", lambda name, settings, size=None: pool)
    monkeypatch.setattr(flag, "fit_student", scripted_train)
    monkeypatch.setattr(
        flag,
        "load_frontier_config",
        lambda: SimpleNamespace(student=pool.spec.student),
    )
    settings_ = _settings(tmp_path)
    report, arrays = flag.run_dataset("toy", settings_, flag.CONDITIONS, 50)

    # 3 folds x 3 conditions, a filtered retrain per condition, the random and the unfiltered
    # teacher retrains.
    assert len(calls) == 9 + 3 + 2
    assert sorted(set(calls)) == [106, 107, round(0.8 * 160), 160]
    assert report["students_trained"] == 14
    assert report["reference"] is None  # no committed noise report for the toy task
    conditions = report["conditions"]
    assert isinstance(conditions, dict)
    assert list(conditions) == list(flag.CONDITIONS)
    teacher = conditions["teacher"]
    n_wrong = sum(lab != gold[it.id] for it, lab in pool.pool)
    assert teacher["train_labels_wrong"] == n_wrong
    assert teacher["flags"]["auroc"]["auroc"] == 1.0
    retrain = teacher["retrain"]
    assert set(retrain) == {"filtered", "random", "unfiltered", "filtered_minus_random"}
    assert retrain["filtered"]["n_train"] == 128
    assert retrain["filtered"]["train_labels_wrong"] == 0  # the flagged 32 held every error
    assert "minus_unfiltered" not in retrain["filtered"]
    for condition in ("uniform", "pair"):
        assert conditions[condition]["train_labels_wrong"] == n_wrong
        assert set(conditions[condition]["retrain"]) == {"filtered"}
    assert arrays["proba_teacher"].shape == (160, len(pool.spec.label_names))
    assert arrays["folds"].tolist() == flag.fold_ids(160, 3, 0).tolist()

    # A second run trains nothing: every prediction comes from the cache.
    calls.clear()
    again, _ = flag.run_dataset("toy", settings_, flag.CONDITIONS, 50)
    assert calls == []
    assert again["students_trained"] == 0
    assert again["conditions"] == report["conditions"]

    # The flag analysis re-runs from the saved probabilities alone.
    out = tmp_path / "flag-toy"
    out.mkdir()
    np.savez_compressed(out / "oof.npz", **arrays)  # type: ignore[arg-type]
    (out / "report.json").write_text(json.dumps(report), "utf-8")
    redone = flag.reanalyse(out / "oof.npz", 50)
    assert json.loads(json.dumps(redone["conditions"])) == json.loads(
        json.dumps(report["conditions"])
    )
    assert flag.summary_line(report, out).startswith("toy: teacher AUROC 1.000")

    text = render.render_flag([("flag-toy-qwen2.5-3b", report, {"command": "cmd"})])
    assert "the teacher's labels" in text
    assert "No flagging run yet" in render.render_flag([])


def test_reanalyse_needs_its_report(tmp_path: Path) -> None:
    from anvil.errors import ConfigError

    np.savez_compressed(tmp_path / "oof.npz", gold=np.zeros(3, dtype=np.int64))
    with pytest.raises(ConfigError, match=r"report\.json"):
        flag.reanalyse(tmp_path / "oof.npz", 10)


def test_main_rejects_bad_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    for argv in (["--resamples", "0"], ["--size", "3"]):
        with pytest.raises(SystemExit) as exc:
            flag.main(argv)
        assert exc.value.code == 2


def test_results_dir_names_are_file_safe() -> None:
    assert flag.results_dir("toy", "qwen2.5:3b", False).name == "flag-toy-qwen2.5-3b"
    assert flag.results_dir("toy", "qwen2.5:3b", True).name == "flag-toy-qwen2.5-3b-smoke"
