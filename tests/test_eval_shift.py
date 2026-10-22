"""The shift harness on synthetic scores: metrics, gates, row selection, config, the analysis
and its scores.npz round trip. No model, no network, no dataset download."""

from __future__ import annotations

import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from sklearn.metrics import roc_auc_score

from anvil.calibrate import THRESHOLD_TOLERANCE, softmax
from anvil.errors import ConfigError
from eval import shift
from eval.data import label_name
from eval.shift import (
    HELDOUT,
    SCORES,
    Role,
    Scored,
    SetCfg,
    analyse,
    auroc,
    fit_gaussians,
    fpr_at_tpr,
    gate_threshold,
    histogram,
    knn_similarity,
    load_scores,
    load_shift_config,
    logit_scores,
    mahalanobis_score,
    main,
    report_from,
    save_scores,
    select_rows,
    served,
)

SHA = "0123456789abcdef0123456789abcdef01234567"
K = 5
LABELS = [f"intent_{i}" for i in range(K)]
TEMPERATURE = 1.5
THRESHOLD = 0.8
COST = 0.05
RESAMPLES = 100
ROLES: dict[str, Role] = {
    "test": "in_distribution",
    "far": "out_of_distribution",
    "near": "out_of_distribution",
}

FloatArray = npt.NDArray[np.float64]


def cfg(**kw: object) -> SetCfg:
    base: dict[str, object] = {
        "name": "s",
        "role": "out_of_distribution",
        "repo": "ns/repo",
        "revision": SHA,
        "license": "MIT",
    }
    return SetCfg.model_validate({**base, **kw})


# ------------------------------------------------------------------------------- metrics


def test_auroc_matches_sklearn_with_many_ties() -> None:
    rng = np.random.default_rng(0)
    pos = rng.integers(0, 6, 80).astype(np.float64) + 1
    neg = rng.integers(0, 6, 60).astype(np.float64)
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    assert auroc(pos, neg) == pytest.approx(roc_auc_score(y, np.r_[pos, neg]))


def test_auroc_edge_cases() -> None:
    assert auroc([3, 4, 5], [0, 1, 2]) == 1.0
    assert auroc([0, 1, 2], [3, 4, 5]) == 0.0
    assert auroc([1, 1, 1], [1, 1]) == 0.5
    with pytest.raises(ValueError, match="at least one"):
        auroc([], [1.0])
    with pytest.raises(ValueError, match="at least one"):
        auroc([1.0], [])


def test_fpr_at_tpr_hand_computed() -> None:
    pos = np.arange(1.0, 11.0)
    neg = [0.0, 5.0, 6.0, 7.0, 9.0]
    # tpr 0.95 of 10 positives: cut at the lowest positive (1.0); 4 of 5 negatives reach it.
    assert fpr_at_tpr(pos, neg, 0.95) == pytest.approx(0.8)
    # tpr 0.5: floor(0.5 * 10) = 5, so the cut is the sixth lowest (6.0); 3 of 5 reach it.
    assert fpr_at_tpr(pos, neg, 0.5) == pytest.approx(0.6)


def test_logit_scores() -> None:
    z = np.random.default_rng(1).normal(scale=3.0, size=(20, K))
    out = logit_scores(z, 2.0)
    assert set(out) == {"max_softmax", "max_logit", "energy"}
    m = z.max(axis=1)
    np.testing.assert_allclose(out["energy"], m + np.log(np.exp(z - m[:, None]).sum(axis=1)))
    np.testing.assert_allclose(out["energy"], np.log(np.exp(z).sum(axis=1)))
    np.testing.assert_allclose(out["max_softmax"], softmax(z, 2.0).max(axis=1))
    np.testing.assert_allclose(out["max_logit"], z.max(axis=1))
    assert not np.allclose(out["max_softmax"], softmax(z, 1.0).max(axis=1))


# ------------------------------------------------------------------------- distance scores


def test_knn_similarity_self_and_far_points() -> None:
    rng = np.random.default_rng(2)
    train = rng.normal(size=(50, 6)) + np.array([5.0, 0, 0, 0, 0, 0])
    assert knn_similarity(train, train[7:8], 1) == pytest.approx([1.0])
    inside = train.mean(axis=0, keepdims=True) + rng.normal(scale=0.1, size=(10, 6))
    far = rng.normal(size=(10, 6)) * np.array([0.1, 5, 5, 5, 5, 5])
    assert knn_similarity(train, inside, 5).min() > knn_similarity(train, far, 5).max()


def test_knn_similarity_k_bounds_and_chunking() -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(12, 4))
    x = rng.normal(size=(10, 4))
    with pytest.raises(ValueError, match="k must be"):
        knn_similarity(train, x, 13)
    with pytest.raises(ValueError, match="k must be"):
        knn_similarity(train, x, 0)
    np.testing.assert_allclose(
        knn_similarity(train, x, 3, chunk=3), knn_similarity(train, x, 3, chunk=1024)
    )


def test_mahalanobis_in_cluster_beats_far_away() -> None:
    rng = np.random.default_rng(4)
    centres = np.zeros((2, 8))
    centres[0, 0], centres[1, 0] = 10.0, -10.0
    codes = np.repeat([0, 1], 100)
    train = centres[codes] + rng.normal(size=(200, 8))
    g = fit_gaussians(train, codes)
    assert g.means.shape == (2, 8)
    assert g.precision.shape == (8, 8)
    inside = centres[rng.integers(0, 2, 20)] + rng.normal(size=(20, 8))
    far = rng.normal(size=(20, 8)) + np.array([0.0, 30, 30, 30, 0, 0, 0, 0])
    assert mahalanobis_score(g, inside).min() > mahalanobis_score(g, far).max()
    at_means = mahalanobis_score(g, g.means)
    np.testing.assert_allclose(at_means, 0.0, atol=1e-8)
    assert at_means.min() >= mahalanobis_score(g, inside).max()


# ------------------------------------------------------------------------------- the gate


def test_gate_threshold_drops_floor_cost_n() -> None:
    s = np.arange(1.0, 11.0)
    accepted = np.ones(10, dtype=bool)
    cut = gate_threshold(s, accepted, 0.3)
    assert int((accepted & (s >= cut)).sum()) == 7
    cut = gate_threshold(s, accepted, 0.35)  # floor(3.5) = 3
    assert int((accepted & (s >= cut)).sum()) == 7


def test_gate_threshold_no_drop_is_minus_infinity() -> None:
    s = np.arange(10.0)
    assert gate_threshold(s, np.ones(10, dtype=bool), 0.09) == -math.inf


def test_gate_threshold_ignores_unaccepted_inputs() -> None:
    s = np.array([5.0, 6, 7, 8, 9, 10, 100, -100, 100, -100])
    accepted = np.array([True] * 6 + [False] * 4)
    base = gate_threshold(s, accepted, 0.2)  # floor(2) = 2 of the 6 accepted
    assert base == 7.0
    kept = accepted & (s >= base)
    assert int(kept.sum()) == 4
    assert not kept[6:].any()
    s2 = s.copy()
    s2[6:] = [-1e9, 1e9, -1e9, 1e9]
    assert gate_threshold(s2, accepted, 0.2) == base


def test_gate_threshold_ties_at_the_cut_drop_fewer() -> None:
    s = np.array([1.0, 2, 2, 2, 3, 4, 5, 6, 7, 8])
    accepted = np.ones(10, dtype=bool)
    cut = gate_threshold(s, accepted, 0.3)  # would drop 3, but three inputs tie at 2.0
    assert cut == 2.0
    assert int((accepted & (s >= cut)).sum()) == 9


def test_gate_threshold_drop_everything_is_infinity() -> None:
    s = np.arange(10.0)
    accepted = np.array([True, True] + [False] * 8)
    assert gate_threshold(s, accepted, 0.5) == math.inf


def test_served_tolerance() -> None:
    conf = np.array([0.8, 0.8 - 5e-7, 0.8 - 2 * THRESHOLD_TOLERANCE, 0.9])
    assert served(conf, 0.8).tolist() == [True, True, False, True]


def test_histogram_counts_and_top_edge() -> None:
    conf = np.array([0.0, 0.05, 0.5, 0.99, 1.0, 1.0])
    h = histogram(conf, 10)
    assert len(h) == 10
    assert sum(h) == len(conf)
    assert h[-1] == 3
    assert h[0] == 2


# -------------------------------------------------------------------------- select_rows


def test_select_rows_keep_filters_by_label_name() -> None:
    texts = ["a", "b", "c", "d"]
    labels = ["oos", "other", "oos", "oos"]
    ids, out, gold = select_rows(cfg(name="clinc", keep="OOS"), texts, labels)
    assert ids == ["clinc-0", "clinc-2", "clinc-3"]
    assert out == ["a", "c", "d"]
    assert gold == ["oos"] * 3
    # The keep value is normalised like dataset class names.
    ids, _, gold = select_rows(
        cfg(keep="Card Arrival"), ["x", "y"], [label_name("Card Arrival"), "z"]
    )
    assert ids == ["s-0"]
    assert gold == ["card_arrival"]


def test_select_rows_dedupes_and_drops_blank_texts() -> None:
    texts = ["hello", "  hello ", "", "   ", "world", "hello"]
    labels = ["a", "b", "c", "d", "e", "f"]
    ids, out, gold = select_rows(cfg(), texts, labels)
    assert out == ["hello", "world"]
    assert ids == ["s-0", "s-4"]
    assert gold == ["a", "e"]


def test_select_rows_seeded_size_sample() -> None:
    texts = [f"t{i}" for i in range(50)]
    labels = ["x"] * 50
    a = select_rows(cfg(size=10, seed=3), texts, labels)
    b = select_rows(cfg(size=10, seed=3), texts, labels)
    c = select_rows(cfg(size=10, seed=4), texts, labels)
    assert a == b
    assert len(a[0]) == 10
    assert len(set(a[0])) == 10
    assert a[0] != c[0]
    for i, t in zip(a[0], a[1], strict=True):
        assert t == f"t{i.removeprefix('s-')}"


def test_select_rows_size_larger_than_available() -> None:
    ids, _, _ = select_rows(cfg(size=100), ["a", "a", "b"], ["x", "x", "x"])
    assert len(ids) == 2


def test_select_rows_without_matches_is_a_config_error() -> None:
    with pytest.raises(ConfigError):
        select_rows(cfg(keep="oos"), ["a", "b"], ["x", "y"])
    with pytest.raises(ConfigError):
        select_rows(cfg(), ["", " "], ["x", "y"])


# ---------------------------------------------------------------------------- the config


def test_committed_config_is_valid() -> None:
    config = load_shift_config()
    roles = [s.role for s in config.sets]
    assert roles.count("in_distribution") == 1
    assert config.reference.role == "in_distribution"
    assert "out_of_distribution" in roles
    for s in config.sets:
        assert len(s.revision) == 40
        int(s.revision, 16)
        assert s.license.strip()


def write_config(path: Path, sets: list[dict[str, object]]) -> Path:
    import yaml

    path.write_text(yaml.safe_dump({"dataset": "d", "sets": sets}), encoding="utf-8")
    return path


def set_dict(name: str, role: str, revision: str = SHA) -> dict[str, object]:
    return {"name": name, "role": role, "repo": "ns/r", "revision": revision, "license": "MIT"}


def test_config_rejects_two_in_distribution_sets(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "c.yaml",
        [set_dict("a", "in_distribution"), set_dict("b", "in_distribution")],
    )
    with pytest.raises(ConfigError):
        load_shift_config(path)


def test_config_rejects_short_revision_and_missing_file(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "c.yaml",
        [set_dict("a", "in_distribution"), set_dict("b", "out_of_distribution", "main")],
    )
    with pytest.raises(ConfigError):
        load_shift_config(path)
    with pytest.raises(ConfigError):
        load_shift_config(tmp_path / "missing.yaml")


def test_config_accepts_a_minimal_valid_file(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "c.yaml",
        [set_dict("a", "in_distribution"), set_dict("b", "out_of_distribution")],
    )
    assert load_shift_config(path).reference.name == "a"


# --------------------------------------------------------------------------- the analysis


def confident(rng: np.random.Generator, n: int) -> tuple[FloatArray, npt.NDArray[np.int64]]:
    """Logits that are right 90% of the time, with a margin that varies a little."""
    gold = rng.integers(0, K, n).astype(np.int64)
    z = rng.normal(scale=0.5, size=(n, K))
    wrong = rng.random(n) < 0.1
    peak = np.where(wrong, (gold + 1) % K, gold)
    z[np.arange(n), peak] += rng.uniform(4.0, 8.0, n)
    return z, gold


def scored(name: str, z: FloatArray, knn: FloatArray, mah: FloatArray, gold: object) -> Scored:
    n = len(z)
    g = None if gold is None else np.asarray(gold, dtype=np.int64)
    return Scored(
        [f"{name}-{i}" for i in range(n)], [f"{name} text {i}" for i in range(n)], z, knn, mah, g
    )


def in_dist(name: str, seed: int, n: int) -> Scored:
    rng = np.random.default_rng(seed)
    z, gold = confident(rng, n)
    return scored(name, z, rng.normal(0.8, 0.05, n), rng.normal(-10.0, 2.0, n), gold)


def make_study() -> tuple[dict[str, Scored], Scored]:
    rng = np.random.default_rng(10)
    sets = {
        "test": in_dist("test", 11, 300),
        "far": scored(
            "far",
            rng.normal(scale=0.5, size=(120, K)),
            rng.normal(0.3, 0.05, 120),
            rng.normal(-60.0, 5.0, 120),
            None,
        ),
        "near": scored(
            "near",
            rng.normal(scale=0.7, size=(100, K)),
            rng.normal(0.4, 0.05, 100),
            rng.normal(-40.0, 5.0, 100),
            None,
        ),
    }
    return sets, in_dist(HELDOUT, 12, 200)


def run_analysis(
    sets: dict[str, Scored], heldout: Scored, labels: list[str] = LABELS
) -> dict[str, Any]:
    return analyse(
        sets,
        ROLES,
        heldout,
        labels,
        TEMPERATURE,
        THRESHOLD,
        cost=COST,
        n_bins=10,
        n_resamples=RESAMPLES,
        seed=0,
    )


# Any: the report is nested JSON read back by key; there is no useful static type.
def sub(r: dict[str, Any], key: str) -> dict[str, Any]:
    value = r[key]
    assert isinstance(value, dict)
    return value


def test_analyse_report_shape() -> None:
    sets, heldout = make_study()
    r = run_analysis(sets, heldout)
    assert {"sets", "reference", "separation", "gates", "hypotheses", "criteria"} <= set(r)
    assert r["labels"] == K
    assert set(sub(r, "sets")) == set(ROLES)
    assert set(sub(r, "separation")) == {"far", "near"}
    for name in ("far", "near"):
        by_score = sub(sub(r, "separation"), name)
        assert set(by_score) == set(SCORES)
        for entry in by_score.values():
            assert 0.0 <= entry["auroc"]["estimate"] <= 1.0
    assert set(sub(r, "gates")) == set(SCORES)
    assert set(sub(sub(r, "hypotheses"), "h2")) == {"far", "near"}
    assert set(sub(sub(r, "hypotheses"), "h1")) == {"coverage_change", "kappa_answered", "held"}
    sets_out = sub(r, "sets")
    assert "top_answers" in sub(sets_out, "far")
    assert "top_answers" not in sub(sets_out, "test")
    assert sum(sub(sets_out, "test")["histogram"]) == 300


def test_analyse_separation_and_baseline_difference() -> None:
    sets, heldout = make_study()
    sep = sub(run_analysis(sets, heldout), "separation")
    for name in ("far", "near"):
        by_score = sub(sep, name)
        assert by_score["max_softmax"]["minus_max_softmax"]["estimate"] == 0.0
        assert by_score["max_softmax"]["minus_max_softmax"]["low"] == 0.0
        for score in ("max_softmax", "max_logit", "energy", "knn", "mahalanobis"):
            assert by_score[score]["auroc"]["estimate"] > 0.9


def test_analyse_ood_answered_below_in_distribution_coverage() -> None:
    sets, heldout = make_study()
    r = run_analysis(sets, heldout)
    coverage = sub(sub(r, "reference"), "coverage")["estimate"]
    assert isinstance(coverage, float)
    assert coverage > 0.7
    for name in ("far", "near"):
        answered = sub(sub(r, "sets"), name)["answered"]["estimate"]
        assert answered < coverage
        assert answered < 0.1
    ref = sub(r, "reference")
    assert ref["set"] == "test"
    assert ref["kappa_answered"]["estimate"] > 0.8
    assert ref["ece_calibrated"] != ref["ece_raw"]


def test_analyse_gate_spends_at_most_the_coverage_cost_on_heldout() -> None:
    sets, heldout = make_study()
    r = run_analysis(sets, heldout)
    held_served = float(
        served(logit_scores(heldout.logits, TEMPERATURE)["max_softmax"], THRESHOLD).mean()
    )
    assert sub(r, "reference")["heldout_coverage"] == pytest.approx(held_served)
    for score, gate in sub(r, "gates").items():
        assert isinstance(gate, dict)
        assert gate["heldout_coverage"] >= held_served - COST - 1e-12, score
        assert gate["heldout_coverage"] <= held_served + 1e-12, score
        assert gate["coverage_cost"] >= -1e-12
        for rate in gate["answered"].values():
            assert 0.0 <= rate <= 1.0


def test_analyse_hypothesis_h2_per_ood_set() -> None:
    sets, heldout = make_study()
    h2 = sub(sub(run_analysis(sets, heldout), "hypotheses"), "h2")
    for name in ("far", "near"):
        entry = h2[name]
        assert isinstance(entry, dict)
        assert entry["held"] is False
        assert 0.0 <= entry["answered"] < 0.4


def test_analyse_requires_gold() -> None:
    sets, heldout = make_study()
    bad = Scored(heldout.ids, heldout.texts, heldout.logits, heldout.knn, heldout.mahalanobis)
    with pytest.raises(ValueError, match="gold codes"):
        run_analysis(sets, bad)


# ------------------------------------------------------------------ scores.npz and main()


def make_context() -> dict[str, object]:
    return {
        "source": {"dataset": "toy", "teacher_model": "gold"},
        "bundle": {
            "path": "bundle",
            "variant": "fp32",
            "labels": LABELS,
            "temperature": TEMPERATURE,
            "threshold": THRESHOLD,
        },
        "settings": {
            "knn_k": 5,
            "coverage_cost": COST,
            "n_bins": 10,
            "n_resamples": RESAMPLES,
            "seed": 0,
        },
        "sets": list(ROLES),
        "roles": dict(ROLES),
        "sources": {name: {"repo": f"ns/{name}", "rows": 1} for name in ROLES},
        "overlap": {"test": {"identical_to_sample": 0}},
        "onnx_vs_torch": {"max_abs_prob_diff": {"test": 1e-6}, "note": "n"},
        "seconds": {"student": 0.1},
    }


def rounded(s: Scored) -> Scored:
    return Scored(
        s.ids,
        s.texts,
        s.logits.astype(np.float32).astype(np.float64),
        s.knn,
        s.mahalanobis,
        s.gold,
    )


def test_scores_round_trip_and_report_from(tmp_path: Path) -> None:
    sets, heldout = make_study()
    context = make_context()
    path = tmp_path / "scores.npz"
    save_scores(path, sets, heldout, context)
    loaded, loaded_held, loaded_context = load_scores(path)

    assert list(loaded) == list(ROLES)
    assert loaded_context == context
    for name, original in {**sets, HELDOUT: heldout}.items():
        got = loaded_held if name == HELDOUT else loaded[name]
        want = rounded(original)
        assert got.ids == want.ids
        assert got.texts == want.texts
        np.testing.assert_array_equal(got.logits, want.logits)
        np.testing.assert_array_equal(got.knn, want.knn)
        np.testing.assert_array_equal(got.mahalanobis, want.mahalanobis)
        if want.gold is None:
            assert got.gold is None
        else:
            assert got.gold is not None
            np.testing.assert_array_equal(got.gold, want.gold)

    report = report_from(loaded, loaded_held, loaded_context)
    direct = run_analysis({n: rounded(s) for n, s in sets.items()}, rounded(heldout))
    assert report["reference"] == direct["reference"]
    assert report["gates"] == direct["gates"]
    bundle = sub(report, "bundle")
    assert "labels" not in bundle
    assert bundle["threshold"] == THRESHOLD
    assert sub(sub(report, "sets"), "far")["source"] == {"repo": "ns/far", "rows": 1}
    assert {"source", "settings", "overlap", "onnx_vs_torch", "seconds"} <= set(report)


def test_main_reanalyses_saved_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(shift, "git_provenance", lambda: {"git_sha": "abc", "git_dirty": False})
    sets, heldout = make_study()
    path = tmp_path / "scores.npz"
    save_scores(path, sets, heldout, make_context())

    assert main(["--scores", str(path)]) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert {"sets", "reference", "separation", "gates", "hypotheses", "criteria"} <= set(report)
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["git_sha"] == "abc"
    assert "test coverage" in capsys.readouterr().out

    # A second pass keeps the first meta and records its own command.
    assert main(["--scores", str(path)]) == 0
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["reanalysed"]["git_sha"] == "abc"
    assert meta["command"].startswith("python -m eval.shift")


def test_main_records_what_training_the_student_spent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On an empty workspace `run` labels and trains the student; a Claude teacher pays for
    # that, and the meta (which the spend ledger sums) must say so.
    monkeypatch.setattr(shift, "git_provenance", lambda: {"git_sha": "abc", "git_dirty": False})
    monkeypatch.setattr(shift, "RESULTS_DIR", tmp_path)
    sets, heldout = make_study()
    context = {**make_context(), "spent_usd": 0.42}
    monkeypatch.setattr(shift, "run", lambda config, teacher, settings: (sets, heldout, context))

    assert main(["--teacher", "claude"]) == 0
    meta = json.loads((tmp_path / "shift-banking77-gold" / "meta.json").read_text("utf-8"))
    assert meta["spent_usd"] == 0.42


def test_main_missing_scores_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(shift, "git_provenance", lambda: {"git_sha": "abc"})
    assert main(["--scores", str(tmp_path / "nope.npz")]) == 2
    assert "not found" in capsys.readouterr().err
    assert not (tmp_path / "report.json").exists()


def test_render_shift_states_every_hypothesis(tmp_path: Path) -> None:
    from eval.render import render_shift

    sets, heldout = make_study()
    context = make_context()
    context["source"] = {
        "dataset": "toy",
        "teacher_model": "gold",
        "student_model": "org/Tiny-BERT",
        "student_run": "run-1",
        "train_inputs": 400,
    }
    context["sources"] = {
        name: {"repo": f"ns/{name}", "revision": SHA, "split": "test", "license": "MIT"}
        for name in ROLES
    }
    context["overlap"] = {
        "test": {"identical_to_sample": 0, "near_duplicates_of_sample": 2, "sample_size": 500}
    }
    report = report_from(sets, heldout, context)
    meta = {"command": "python -m eval.shift", "git_sha": "abcdef1234", "cpus": 4}
    text = render_shift([("shift-toy-gold", report, meta)])
    assert "`eval/results/shift-toy-gold/`" in text
    assert f"`ns/far`@`{SHA[:7]}` test" in text
    assert "near-duplicates (word Jaccard >= 0.8): 2" in text
    for score in ("max softmax (served)", "energy", "Mahalanobis to class means"):
        assert f"| {score} |" in text
        assert f"| + {score} |" in text
    assert text.count("- H2 **") == 2  # one H2 line per out-of-distribution set
    assert "- H1 **" in text
    assert "- H3 **" in text
    assert render_shift([]) == "No shift run yet: `make eval-shift`.\n"


def test_load_rows_passes_the_dataset_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    # `datasets` is in the eval group, which CI does not install: stand in for the module.
    datasets = types.ModuleType("datasets")
    monkeypatch.setitem(sys.modules, "datasets", datasets)

    from eval.data import DatasetRef, load_rows

    seen: dict[str, object] = {}

    class Features(dict[str, object]):
        pass

    class Label:
        names = ("Out Of Scope", "balance")

    class Fake(dict[str, list[object]]):
        features = Features(intent=Label())

    def fake_load(repo: str, config: str | None, **kw: object) -> Fake:
        seen.update(repo=repo, config=config, **kw)
        return Fake(text=["a", "b"], intent=[0, 1])

    monkeypatch.setattr(datasets, "load_dataset", fake_load, raising=False)
    ref = DatasetRef(
        "clinc/clinc_oos", SHA, "test", text_field="text", label_field="intent", config="plus"
    )
    assert load_rows(ref) == (["a", "b"], ["out_of_scope", "balance"])
    assert seen == {"repo": "clinc/clinc_oos", "config": "plus", "revision": SHA, "split": "test"}


def test_near_duplicates_get_their_own_reading_and_survive_the_npz(tmp_path: Path) -> None:
    sets, heldout = make_study()
    test = sets["test"]
    near = np.zeros(len(test.ids), dtype=bool)
    near[:40] = True
    sets["test"] = Scored(
        test.ids, test.texts, test.logits, test.knn, test.mahalanobis, test.gold, near
    )
    fresh = sub(sub(run_analysis(sets, heldout), "reference"), "fresh")
    assert fresh["n"] == len(near) - 40
    assert fresh["near_duplicates"] == 40
    assert 0.0 <= fresh["coverage"] <= 1.0
    assert set(fresh["kappa_answered"]) == {"estimate", "low", "high"}

    path = tmp_path / "scores.npz"
    save_scores(path, sets, heldout, make_context())
    loaded, loaded_held, _ = load_scores(path)
    assert loaded["test"].near_duplicate is not None
    np.testing.assert_array_equal(loaded["test"].near_duplicate, near)
    assert loaded_held.near_duplicate is None


def test_near_duplicate_mask_matches_the_count() -> None:
    from eval.data import near_duplicate_mask, near_duplicates

    pool = ["how do i top up my card", "where is my new card"]
    texts = ["how do i top up my card please", "what is the exchange rate", "", "where is my card"]
    mask = near_duplicate_mask(texts, pool)
    assert mask == [True, False, False, True]
    assert near_duplicates(texts, pool) == sum(mask)
