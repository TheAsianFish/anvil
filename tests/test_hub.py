"""`anvil push`: the model card rendered from measured results, and the upload with a fake Hub."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from anvil.cli import app
from anvil.errors import ConfigError, HubError
from anvil.hub import (
    CARD_FILE,
    LabelCost,
    check_repo_id,
    load_card_inputs,
    push,
    render_card,
)
from anvil.merge import MERGED_META
from anvil.runtime import BUNDLE_META, OnnxStudent, read_bundle_meta
from anvil.store import Store
from anvil.teacher.base import Item, Label
from tests.test_export import PROBES, exported

TASK = "ticket-triage"
RUN = "20261021-072428-n200"
TEACHER = "claude-opus-5-5"
PROMPT = "9b2d7959769e52a1"
STUDENT = {
    "task": TASK,
    "teacher_model": TEACHER,
    "prompt_hash": PROMPT,
    "splits": {"heldout_fraction": 0.25, "seed": 0},
    "student_model": "tiny",
    "train": {
        "examples": 184,
        "epochs": 6,
        "best_epoch": 4,
        "stopped_early": True,
        "validation_examples": 16,
    },
}
CALIBRATION = {
    "temperature": 1.5,
    "n": 91,
    "temperature_identified": True,
    "ece_before": 0.0612,
    "ece_after": 0.0234,
    "n_bins": 15,
    "target_kappa": 0.9,
    "rule": "lower",
    "threshold": 0.4,
    "coverage": 0.8571,
    "kappa": 0.9412,
    "kappa_low": 0.9031,
    "teacher_model": TEACHER,
    "student": RUN,
}
REPORT: dict[str, object] = {
    "task": TASK,
    "student": RUN,
    "student_model": "tiny",
    "teacher_model": TEACHER,
    "prompt_hash": PROMPT,
    "labels": ["billing", "bug", "feature"],
    "train_examples": 200,
    "n": 91,
    "kappa": 0.8123,
    "kappa_ci": {"estimate": 0.8123, "low": 0.7311, "high": 0.8877},
    "accuracy": 0.8791,
    "accuracy_ci": {"estimate": 0.8791, "low": 0.8132, "high": 0.9341},
    "majority_label": "bug",
    "majority_accuracy": 0.2088,
    "macro_f1": 0.8654,
    "per_class": [
        {"label": "billing", "support": 21, "precision": 0.9, "recall": 0.8571, "f1": 0.878},
        {"label": "bug", "support": 40, "precision": 0.85, "recall": 0.85, "f1": 0.85},
        {"label": "feature", "support": 30, "precision": 0.875, "recall": 0.9333, "f1": 0.9032},
    ],
}
MERGED_RUN = "20261021-072529-n120"
MERGED: dict[str, object] = {
    "format": 1,
    "exported_at": "2026-10-21T07:25:50+00:00",
    "student": MERGED_RUN,
    "outputs": ["sql"],
    "system": 'Write one SQLite query.\nQuote "columns".',
    "max_length": 384,
    "max_new_tokens": 56,
    "stop_ids": [258],
    "parity": {
        "probes": 52,
        "logit_probes": 8,
        "max_abs_logit_diff": 3.5762786865234375e-07,
        "same_completions": 52,
        "passed": True,
        "gate": "max abs logit diff <= 0.001 and every completion identical",
    },
    "source": {
        "student_model": "Qwen/Qwen2.5-0.5B-Instruct",
        "lora": {"r": 16, "alpha": 32},
        "task": "toy-sql",
        "kind": "generate",
        "teacher_model": "gold",
        "prompt_hash": "57395dd1a087e598",
        "splits": {"heldout_fraction": 0.25, "seed": 0},
        "train_examples": 120,
        "trainable_parameters": 32768,
        "train": {"examples": 103, "epochs": 6, "best_epoch": 6, "validation_examples": 17},
    },
    "spec": {"name": "toy-sql", "kind": "generate"},
}
MERGED_REPORT: dict[str, object] = {
    "task": "toy-sql",
    "student": MERGED_RUN,
    "teacher_model": "gold",
    "prompt_hash": "57395dd1a087e598",
    "train_examples": 120,
    "n": 52,
    "exact_match": {"estimate": 0.6731, "low": 0.5385, "high": 0.7885},
    "field_match": [{"name": "sql", "match_rate": 0.6923, "n": 52}],
    "valid": 49,
    "invalid": 3,
}


class FakeHub:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.error = error

    def create_repo(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool | None = None,
        exist_ok: bool = False,
    ) -> object:
        self.calls.append(
            ("create_repo", {"repo_id": repo_id, "token": token, "private": private,
                             "exist_ok": exist_ok})
        )  # fmt: skip
        if self.error is not None:
            raise self.error
        return f"https://huggingface.co/{repo_id}"

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        commit_message: str | None = None,
        token: str | None = None,
        allow_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
    ) -> object:
        self.calls.append(
            ("upload_folder", {"repo_id": repo_id, "folder_path": folder_path,
                               "commit_message": commit_message, "token": token,
                               "allow_patterns": allow_patterns,
                               "delete_patterns": delete_patterns})
        )  # fmt: skip
        return None


def _bundle_at(src: Path, out: Path, **meta_updates: object) -> Path:
    shutil.copytree(src, out)
    meta = json.loads((out / BUNDLE_META).read_text(encoding="utf-8"))
    meta.update(
        {"student": STUDENT, "calibration": CALIBRATION, "run": RUN, "spec": {"name": TASK}}
    )
    meta.update(meta_updates)
    (out / BUNDLE_META).write_text(json.dumps(meta), encoding="utf-8")
    return out


def _write_report(workspace: Path, run: str, report: dict[str, object]) -> None:
    (workspace / "reports").mkdir(parents=True, exist_ok=True)
    (workspace / "reports" / f"eval-{run}.json").write_text(json.dumps(report), encoding="utf-8")


def _write_store(workspace: Path, cost: float, n: int = 3) -> None:
    with Store(workspace / "store.sqlite", 0, 0.25) as store:
        store.add_inputs(Item(f"t{i}", f"text {i}") for i in range(n))
        store.append_labels(
            [Label(f"t{i}", "bug", TEACHER, PROMPT, cost_usd=cost / n) for i in range(n)]
        )


@pytest.fixture
def home(tmp_path: Path) -> Path:
    return tmp_path / "home"


@pytest.fixture
def bundle(tmp_path_factory: pytest.TempPathFactory, home: Path) -> Path:
    """A real exported bundle under `<home>/<task>/exports/<run>`, with its report and store."""
    ws = home / TASK
    out = _bundle_at(exported(tmp_path_factory).bundle, ws / "exports" / RUN)
    _write_report(ws, RUN, REPORT)
    _write_store(ws, 0.0123)
    return out


@pytest.fixture
def merged(tmp_path: Path) -> Path:
    ws = tmp_path / "eval-ws"
    out = ws / "exports" / f"{MERGED_RUN}-merged"
    out.mkdir(parents=True)
    (out / MERGED_META).write_text(json.dumps(MERGED), encoding="utf-8")
    (out / "model.safetensors").write_bytes(b"weights")
    _write_report(ws, MERGED_RUN, MERGED_REPORT)
    return out


def _front_matter(card: str) -> dict[str, object]:
    assert card.startswith("---\n")
    doc = yaml.safe_load(card.split("---\n")[1])
    assert isinstance(doc, dict)
    return doc


def test_bundle_card_carries_the_measured_numbers(bundle: Path, home: Path) -> None:
    inputs = load_card_inputs(bundle, home)
    assert inputs.kind == "bundle"
    assert inputs.report == REPORT
    assert inputs.cost is not None
    assert inputs.cost.labels == 3
    assert inputs.cost.usd == pytest.approx(0.0123)
    card = render_card(inputs.kind, inputs.meta, inputs.report, inputs.cost, "me/triage")
    for text in [
        "| Held-out inputs | 91 |",
        "| Cohen's kappa [95% CI] | 0.812 [0.731, 0.888] |",
        "| Accuracy (agreement) [95% CI] | 0.879 [0.813, 0.934] |",
        "| Macro F1 | 0.865 |",
        "| Majority baseline accuracy (always `bug`) | 0.209 |",
        "| `feature` | 30 | 0.875 | 0.933 | 0.903 |",
        "| Temperature | 1.500 (fitted on 91 held-out labels) |",
        "| ECE (15 bins) | 0.061 -> 0.023 |",
        "| Escalation threshold | 0.4000",
        "85.7% of held-out inputs answered by the student, at kappa 0.941 (lower bound 0.903)",
        "the 95% bootstrap lower bound of kappa must reach it",
        "| Labels trained on | 200 (184 for gradient steps, 16 held back for early stopping) |",
        "| Epochs | 6 (best epoch 4, stopped early) |",
        f"| Prompt hash | `{PROMPT}` |",
        "cost $0.0123",
        "hf download me/triage --local-dir student",
        "anvil serve student",
        '{"texts": ["your text here"]}',
    ]:
        assert text in card, text
    # The export's own parity row, with the pipes of its gate escaped for the table.
    fp32 = inputs.meta["variants"]["fp32"]["parity"]  # type: ignore[index]
    assert f"| {fp32['max_abs_diff']:.2e} |" in card
    assert "max \\|logit diff\\|" in card
    assert "| int8 | `model.int8.onnx` |" in card
    assert render_card("bundle", inputs.meta, inputs.report, inputs.cost) == render_card(
        "bundle", inputs.meta, inputs.report, inputs.cost
    )


def test_front_matter_is_yaml_with_the_hub_keys(bundle: Path, home: Path) -> None:
    inputs = load_card_inputs(bundle, home)
    front = _front_matter(render_card("bundle", inputs.meta, inputs.report))
    assert front["license"] == "mit"
    assert front["library_name"] == "onnx"
    assert front["pipeline_tag"] == "text-classification"
    tags = front["tags"]
    assert isinstance(tags, list)
    assert {"anvil", "distillation", "text-classification"} <= set(tags)
    assert "base_model" not in front  # `tiny` is not a Hub model

    meta = {**inputs.meta, "student": {**STUDENT, "student_model": "answerdotai/ModernBERT-base"}}
    front = _front_matter(render_card("bundle", meta))
    assert front["base_model"] == "answerdotai/ModernBERT-base"


def test_missing_report_says_not_measured(bundle: Path, home: Path) -> None:
    (home / TASK / "reports" / f"eval-{RUN}.json").unlink()
    (home / TASK / "store.sqlite").unlink()
    inputs = load_card_inputs(bundle, home)
    assert inputs.report is None
    assert inputs.cost is None
    card = render_card(inputs.kind, inputs.meta, inputs.report, inputs.cost)
    assert "Not measured: run `anvil eval` before pushing." in card
    assert "Cohen's kappa" not in card
    assert "Cost\n\nNot recorded" in card


def test_report_for_another_prompt_is_not_used(bundle: Path, home: Path) -> None:
    _write_report(home / TASK, RUN, {**REPORT, "prompt_hash": "0000000000000000"})
    assert load_card_inputs(bundle, home).report is None


def test_report_is_found_next_to_the_export_before_home(
    tmp_path_factory: pytest.TempPathFactory, tmp_path: Path, home: Path
) -> None:
    # The eval harness's workspaces are `.anvil/eval/<task>-<sample>/`, not `<home>/<task>`.
    ws = tmp_path / "eval" / "triage-n500"
    out = _bundle_at(exported(tmp_path_factory).bundle, ws / "exports" / RUN)
    _write_report(ws, RUN, {**REPORT, "kappa": 0.5555})
    _write_store(ws, 0.5)
    _write_report(home / TASK, RUN, REPORT)
    inputs = load_card_inputs(out, home)
    assert inputs.report is not None
    assert inputs.report["kappa"] == 0.5555
    assert inputs.cost is not None
    assert inputs.cost.usd == pytest.approx(0.5)

    (ws / "reports" / f"eval-{RUN}.json").unlink()
    fallback = load_card_inputs(out, home).report
    assert fallback is not None
    assert fallback["kappa"] == 0.8123


def test_explicit_report_wins_and_must_match(bundle: Path, home: Path, tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text(json.dumps({**REPORT, "kappa": 0.4242}), encoding="utf-8")
    report = load_card_inputs(bundle, home, path).report
    assert report is not None
    assert report["kappa"] == 0.4242
    path.write_text(json.dumps({**REPORT, "teacher_model": "other"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="pass this student's report"):
        load_card_inputs(bundle, home, path)
    with pytest.raises(ConfigError, match="does not exist"):
        load_card_inputs(bundle, home, tmp_path / "missing.json")


def test_merged_card(merged: Path, home: Path) -> None:
    inputs = load_card_inputs(merged, home)
    assert inputs.kind == "merged"
    assert inputs.report == MERGED_REPORT
    assert inputs.cost is None  # no store next to the export or under home
    card = render_card(inputs.kind, inputs.meta, inputs.report, LabelCost(172, 1.23456), "me/sql")
    front = _front_matter(card)
    assert front["library_name"] == "transformers"
    assert front["pipeline_tag"] == "text-generation"
    assert front["base_model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    for text in [
        "| Exact match [95% CI] | 0.673 [0.538, 0.788] |",
        "| Field `sql` match | 0.692 |",
        "| Invalid answers | 3 |",
        "| Held-out inputs | 52 |",
        "| Identical greedy completions | 52 of 52 |",
        "| Max abs logit diff | 3.58e-07 (on 8 prompts) |",
        "| LoRA | r 16, alpha 32 |",
        "| Trainable parameters | 32,768 |",
        'repo = "me/sql"',
        # The system prompt as a valid Python string literal.
        json.dumps(MERGED["system"]),
        "max_new_tokens=56",
        "The 172 teacher labels",
        "cost $1.2346",
        "The answer is the bare `sql` value.",
    ]:
        assert text in card, text
    assert "Calibration and the cascade" not in card


def test_offline_teachers_are_named_in_the_limitations(bundle: Path, home: Path) -> None:
    meta = read_bundle_meta(bundle)
    assert "not an LLM" not in render_card("bundle", meta)
    fake = {**meta, "student": {**STUDENT, "teacher_model": "fake-rules-bbf26b5d"}}
    assert "offline keyword-rule fake" in render_card("bundle", fake)
    assert "gold labels stood in for an LLM" in render_card("merged", MERGED)
    assert "not accuracy against ground truth" in render_card("bundle", meta)


def test_push_creates_the_repo_then_uploads(bundle: Path, home: Path) -> None:
    hub = FakeHub()
    r = push(bundle, "me/triage", home=home, token="hf_x", api=hub)
    assert [name for name, _ in hub.calls] == ["create_repo", "upload_folder"]
    assert hub.calls[0][1] == {
        "repo_id": "me/triage", "token": "hf_x", "private": True, "exist_ok": True
    }  # fmt: skip
    upload = hub.calls[1][1]
    assert upload["folder_path"] == bundle
    assert upload["token"] == "hf_x"
    assert upload["allow_patterns"] == r.files
    assert upload["delete_patterns"] == "*"
    assert RUN in str(upload["commit_message"])
    assert r.url == "https://huggingface.co/me/triage"
    assert r.card == bundle / CARD_FILE
    assert {CARD_FILE, BUNDLE_META, "model.onnx", "tokenizer.json"} <= set(r.files)

    push(bundle, "me/triage", home=home, token="hf_x", private=False, api=(hub := FakeHub()))
    assert hub.calls[0][1]["private"] is False


def test_push_needs_a_token_unless_dry_run(bundle: Path, home: Path) -> None:
    hub = FakeHub()
    with pytest.raises(ConfigError, match="HF_TOKEN"):
        push(bundle, "me/triage", home=home, token=None, api=hub)
    r = push(bundle, "me/triage", home=home, token=None, dry_run=True, api=hub)
    assert hub.calls == []
    assert r.dry_run
    assert r.url is None
    assert (bundle / CARD_FILE).read_text(encoding="utf-8").startswith("---\n")


def test_upload_failure_is_a_hub_error(bundle: Path, home: Path) -> None:
    with pytest.raises(HubError, match="push to me/triage failed: OSError: 403"):
        push(bundle, "me/triage", home=home, token="t", api=FakeHub(OSError("403 forbidden")))


@pytest.mark.parametrize("repo", ["triage", "a/b/c", "me/", "/x", "me/-x", "me/x.", "me/a b"])
def test_bad_repo_ids(repo: str) -> None:
    with pytest.raises(ConfigError, match="namespace/name"):
        check_repo_id(repo)


def test_good_repo_id() -> None:
    assert check_repo_id("TheAsianFish/ticket-triage_v1.0") == "TheAsianFish/ticket-triage_v1.0"


def test_non_artifacts_are_refused(tmp_path: Path, home: Path) -> None:
    student = tmp_path / "students" / RUN
    student.mkdir(parents=True)
    (student / "anvil_student.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"trained student directory.*`anvil export`"):
        load_card_inputs(student, home)
    with pytest.raises(ConfigError, match="not a directory"):
        load_card_inputs(tmp_path / "nowhere", home)


def test_bundle_still_loads_with_the_card_in_it(bundle: Path, home: Path) -> None:
    push(bundle, "me/triage", home=home, token=None, dry_run=True)
    assert (bundle / CARD_FILE).is_file()
    assert read_bundle_meta(bundle)["run"] == RUN
    student = OnnxStudent.load(bundle)
    assert len(student.predict(PROBES)) == len(PROBES)


def test_cli_dry_run(bundle: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANVIL_HOME", str(home))
    runner = CliRunner()
    result = runner.invoke(app, ["push", str(bundle), "--repo", "me/triage", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert f"model card {bundle / CARD_FILE}" in result.output
    assert "files would be uploaded to me/triage" in result.output
    assert "Cohen's kappa [95% CI] | 0.812" in (bundle / CARD_FILE).read_text(encoding="utf-8")

    result = runner.invoke(app, ["push", str(bundle), "--repo", "me/triage"])
    assert result.exit_code == 2
    assert "HF_TOKEN" in result.output
