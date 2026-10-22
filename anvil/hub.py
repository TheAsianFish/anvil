"""Publish an exported student to the Hugging Face Hub with a model card rendered from what was
measured: the eval report, the export's calibration and parity checks, and the label cost.

The card is written into the export directory as `README.md` (the file the Hub shows) and
uploaded with it. Every number in it is read from those inputs; a measurement that is missing
is reported as missing, never filled in.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import yaml

from anvil.errors import ConfigError, ExportError, HubError
from anvil.log import get_logger
from anvil.merge import MERGED_FORMAT, MERGED_META
from anvil.pipeline import Workspace
from anvil.runtime import BUNDLE_META, read_bundle_meta
from anvil.store import Store
from anvil.students.encoder import TINY

ArtifactKind = Literal["bundle", "merged"]
CARD_FILE = "README.md"
GOLD_TEACHER = "gold"
# The Hub's own rule, less the length limit: letters, digits, `-`, `_` and `.`, not starting
# or ending with `-` or `.`.
_NAME = r"[A-Za-z0-9](?:[\w.-]*[A-Za-z0-9])?"
_REPO_ID = re.compile(rf"{_NAME}/{_NAME}")
_STUDENT_METAS = ("anvil_student.json", "anvil_generator.json", "anvil_tagger.json")

LOG = get_logger(__name__)


class HubApi(Protocol):
    """The two `huggingface_hub.HfApi` methods a push uses (a fake in tests)."""

    def create_repo(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool | None = None,
        exist_ok: bool = False,
    ) -> object: ...

    def upload_folder(
        self,
        *,
        repo_id: str,
        folder_path: str | Path,
        commit_message: str | None = None,
        token: str | None = None,
        allow_patterns: list[str] | str | None = None,
        delete_patterns: list[str] | str | None = None,
    ) -> object: ...


@dataclass(frozen=True)
class LabelCost:
    """What the teacher's labels for one (teacher model, prompt) key cost, from the store."""

    labels: int
    usd: float


@dataclass(frozen=True)
class CardInputs:
    artifact: Path
    kind: ArtifactKind
    meta: dict[str, object]
    report: dict[str, object] | None
    cost: LabelCost | None


@dataclass(frozen=True)
class PushResult:
    repo_id: str
    url: str | None
    """None on a dry run."""
    files: list[str]
    """Paths relative to the artifact, uploaded (or, on a dry run, that would be)."""
    card: Path
    dry_run: bool


def _dict(value: object) -> dict[str, object]:
    return {str(k): v for k, v in value.items()} if isinstance(value, dict) else {}


def _num(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _int(value: object) -> int | None:
    n = _num(value)
    return int(n) if n is not None and n == int(n) else None


def _str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _f(value: object, spec: str = ".3f") -> str:
    n = _num(value)
    return "not recorded" if n is None else format(n, spec)


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


@dataclass(frozen=True)
class _Facts:
    """The fields the card needs, wherever the bundle or the merged export keeps them."""

    task: str | None
    run: str | None
    teacher: str | None
    prompt_hash: str | None
    student_model: str | None
    splits: dict[str, object]
    train: dict[str, object]
    train_examples: int | None
    labels: list[str]
    outputs: list[str]


def _facts(kind: ArtifactKind, meta: Mapping[str, object]) -> _Facts:
    spec = _dict(meta.get("spec"))
    if kind == "bundle":
        src = _dict(meta.get("student"))
        run = _str(meta.get("run")) or _str(_dict(meta.get("calibration")).get("student"))
        train = _dict(src.get("train"))
        fit, held = _int(train.get("examples")), _int(train.get("validation_examples"))
        examples = fit + (held or 0) if fit is not None else None
    else:
        src = _dict(meta.get("source"))
        run = _str(meta.get("student"))
        train = _dict(src.get("train"))
        examples = _int(src.get("train_examples"))
    labels = meta.get("labels")
    outputs = meta.get("outputs")
    return _Facts(
        task=_str(src.get("task")) or _str(spec.get("name")),
        run=run,
        teacher=_str(src.get("teacher_model")),
        prompt_hash=_str(src.get("prompt_hash")),
        student_model=_str(src.get("student_model")),
        splits=_dict(src.get("splits")),
        train=train,
        train_examples=examples,
        labels=[str(x) for x in labels] if isinstance(labels, list) else [],
        outputs=[str(x) for x in outputs] if isinstance(outputs, list) else [],
    )


def _code(value: str | None) -> str:
    return f"`{value}`" if value else "not recorded"


def _front_matter(kind: ArtifactKind, facts: _Facts) -> str:
    task_tag = "text-classification" if kind == "bundle" else "text-generation"
    front: dict[str, object] = {
        "license": "mit",
        "library_name": "onnx" if kind == "bundle" else "transformers",
        "pipeline_tag": task_tag,
        "tags": ["anvil", "distillation", task_tag, "onnx" if kind == "bundle" else "lora"],
    }
    model = facts.student_model
    if model and model != TINY and _REPO_ID.fullmatch(model):
        front["base_model"] = model
    return "---\n" + yaml.safe_dump(front, sort_keys=False, allow_unicode=True) + "---\n"


def _classify_parity(report: Mapping[str, object]) -> list[str]:
    def ci(key: str) -> str:
        c = _dict(report.get(key))
        return f"[{_f(c.get('low'))}, {_f(c.get('high'))}]" if c else "[not recorded]"

    majority = _str(report.get("majority_label"))
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Held-out inputs | {_f(report.get('n'), '.0f')} |",
        f"| Cohen's kappa [95% CI] | {_f(report.get('kappa'))} {ci('kappa_ci')} |",
        f"| Accuracy (agreement) [95% CI] | {_f(report.get('accuracy'))} {ci('accuracy_ci')} |",
        f"| Macro F1 | {_f(report.get('macro_f1'))} |",
        f"| Majority baseline accuracy (always `{majority or '?'}`) |"
        f" {_f(report.get('majority_accuracy'))} |",
    ]
    per_class = report.get("per_class")
    if isinstance(per_class, list) and per_class:
        lines += ["", "| Label | Support | Precision | Recall | F1 |", "|---|---|---|---|---|"]
        for row in per_class:
            r = _dict(row)
            lines.append(
                f"| `{_cell(str(r.get('label')))}` | {_f(r.get('support'), '.0f')} |"
                f" {_f(r.get('precision'))} | {_f(r.get('recall'))} | {_f(r.get('f1'))} |"
            )
    return lines


def _generate_parity(report: Mapping[str, object]) -> list[str]:
    em = _dict(report.get("exact_match"))
    lines = [
        "| Metric | Value |",
        "|---|---|",
        f"| Held-out inputs | {_f(report.get('n'), '.0f')} |",
        f"| Exact match [95% CI] | {_f(em.get('estimate'))}"
        f" [{_f(em.get('low'))}, {_f(em.get('high'))}] |",
    ]
    fields = report.get("field_match")
    for row in fields if isinstance(fields, list) else []:
        r = _dict(row)
        lines.append(f"| Field `{_cell(str(r.get('name')))}` match | {_f(r.get('match_rate'))} |")
    lines.append(f"| Invalid answers | {_f(report.get('invalid'), '.0f')} |")
    return lines


def _parity(kind: ArtifactKind, facts: _Facts, report: Mapping[str, object] | None) -> list[str]:
    out = ["## Parity with the teacher", ""]
    if report is None:
        return [*out, "Not measured: run `anvil eval` before pushing.", ""]
    teacher = _str(report.get("teacher_model")) or facts.teacher
    trained = _int(report.get("train_examples"))
    trained_on = f" (trained on {trained} labels)" if trained is not None else ""
    out += [
        f"Agreement with {_code(teacher)} of student `{report.get('student')}`{trained_on} on"
        " held-out inputs it never trained on, measured by `anvil eval` (95% intervals are"
        " bootstrap percentiles).",
        "",
    ]
    out += _classify_parity(report) if kind == "bundle" else _generate_parity(report)
    return [*out, ""]


IN_DISTRIBUTION_NOTE = (
    "Coverage was measured in distribution: on held-out inputs drawn from the same data as the"
    " training labels. Inputs unlike that data can clear the threshold too (a confidence"
    " threshold is no out-of-distribution detector), so on other traffic the student may answer"
    " a different share, and answer inputs no label fits; measure it on your own traffic"
    " before relying on the coverage figure."
)


def _calibration(meta: Mapping[str, object]) -> list[str]:
    out = ["## Calibration and the cascade", ""]
    calib = _dict(meta.get("calibration"))
    if not calib:
        return [
            *out,
            "Not calibrated: the bundle serves raw probabilities (temperature 1.0) and has no"
            " escalation threshold, so `anvil serve` needs `--threshold`. Run `anvil calibrate`"
            " and export again.",
            "",
        ]
    identified = calib.get("temperature_identified")
    note = (
        "; not identified, as the held-out NLL has no minimum, so left at 1.0"
        if identified is False
        else ""
    )
    out += [
        "| | |",
        "|---|---|",
        f"| Temperature | {_f(meta.get('temperature'))} (fitted on"
        f" {_f(calib.get('n'), '.0f')} held-out labels{note}) |",
        f"| ECE ({_f(calib.get('n_bins'), '.0f')} bins) | {_f(calib.get('ece_before'))} ->"
        f" {_f(calib.get('ece_after'))} |",
    ]
    threshold = _num(meta.get("threshold"))
    rule = _str(calib.get("rule"))
    rule_text = {
        "lower": "the 95% bootstrap lower bound of kappa must reach it",
        "point": "the kappa estimate must reach it",
    }.get(rule or "", f"rule {_code(rule)}")
    out.append(f"| Target | kappa {_f(calib.get('target_kappa'), 'g')} ({rule_text}) |")
    if threshold is None:
        out += [
            "",
            "No threshold reached the target on the held-out split, so the cascade would send"
            " every input to the teacher.",
            "",
        ]
        return out
    coverage = _num(calib.get("coverage"))
    low = _num(calib.get("kappa_low"))
    out += [
        f"| Escalation threshold | {threshold:.4f} (inputs whose top probability is below it go"
        " to the teacher) |",
        f"| Coverage at the target | {'not recorded' if coverage is None else f'{coverage:.1%}'}"
        f" of held-out inputs answered by the student, at kappa {_f(calib.get('kappa'))}"
        + (f" (lower bound {low:.3f})" if low is not None else "")
        + " |",
        "",
        IN_DISTRIBUTION_NOTE,
        "",
    ]
    return out


def _export_parity(kind: ArtifactKind, meta: Mapping[str, object]) -> list[str]:
    out = ["## Export parity", ""]
    if kind == "merged":
        p = _dict(meta.get("parity"))
        probes = _f(p.get("probes"), ".0f")
        passed = "passed" if p.get("passed") is True else "not passed"
        return [
            *out,
            "The LoRA adapter is merged into the base weights, so this is a plain transformers"
            " checkpoint. The merged model was checked against the adapter model on held-out"
            f" inputs before it was written ({passed}):",
            "",
            "| Check | Value |",
            "|---|---|",
            f"| Identical greedy completions | {_f(p.get('same_completions'), '.0f')} of {probes}"
            " |",
            f"| Max abs logit diff | {_f(p.get('max_abs_logit_diff'), '.2e')} (on"
            f" {_f(p.get('logit_probes'), '.0f')} prompts) |",
            f"| Gate | `{_cell(str(p.get('gate')))}` |",
            "",
        ]
    out += [
        "Each ONNX graph was compared with its reference on held-out inputs at export; a variant"
        " that fails its gate is not served.",
        "",
        "| Variant | File | Reference | Probes | Max abs logit diff | Agreement | Kappa drop"
        " | Size (MB) | Gate | Passed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, info in _dict(meta.get("variants")).items():
        v = _dict(info)
        p = _dict(v.get("parity"))
        drop = _num(p.get("kappa_drop"))
        out.append(
            f"| {name} | `{_cell(str(v.get('file')))}` | {_cell(str(p.get('reference')))} |"
            f" {_f(p.get('n'), '.0f')} | {_f(p.get('max_abs_diff'), '.2e')} |"
            f" {_f(p.get('agreement'))} | {'n/a' if drop is None else f'{drop:+.3f}'} |"
            f" {_f(p.get('size_mb'), '.1f')} | `{_cell(str(p.get('gate')))}` |"
            f" {'yes' if v.get('passed') is True else 'no'} |"
        )
    return [*out, ""]


def _training(kind: ArtifactKind, meta: Mapping[str, object], facts: _Facts) -> list[str]:
    t = facts.train
    fit, held = _int(t.get("examples")), _int(t.get("validation_examples"))
    split = (
        f" ({fit} for gradient steps, {held} held back for early stopping)"
        if fit is not None and held
        else ""
    )
    best = _int(t.get("best_epoch"))
    epochs = _f(t.get("epochs"), ".0f")
    if best is not None:
        epochs += f" (best epoch {best}" + (", stopped early)" if t.get("stopped_early") else ")")
    frac, seed = _num(facts.splits.get("heldout_fraction")), _int(facts.splits.get("seed"))
    rows = [
        ("Teacher", _code(facts.teacher)),
        ("Prompt hash", _code(facts.prompt_hash)),
        ("Labels trained on", f"{_f(facts.train_examples, '.0f')}{split}"),
        ("Epochs", epochs),
        ("Student base model", _code(facts.student_model)),
        (
            "Held-out split",
            f"{frac:.0%} of inputs by content hash, seed {seed}"
            if frac is not None and seed is not None
            else "not recorded",
        ),
    ]
    if kind == "merged":
        src = _dict(meta.get("source"))
        lora = _dict(src.get("lora"))
        if lora:
            rows.append(
                ("LoRA", f"r {_f(lora.get('r'), '.0f')}, alpha {_f(lora.get('alpha'), '.0f')}")
            )
        params = _int(src.get("trainable_parameters"))
        if params is not None:
            rows.append(("Trainable parameters", f"{params:,}"))
    rows += [
        ("Student run", _code(facts.run)),
        ("Exported at", _str(meta.get("exported_at")) or "not recorded"),
    ]
    return [
        "## Training",
        "",
        "| | |",
        "|---|---|",
        *(f"| {k} | {_cell(v)} |" for k, v in rows),
        "",
    ]


def _cost(cost: LabelCost | None) -> list[str]:
    out = ["## Cost", ""]
    if cost is None:
        return [*out, "Not recorded: no label store was found for this task.", ""]
    return [
        *out,
        f"The {cost.labels} teacher labels in the store for this teacher and prompt (training and"
        f" held-out, plus any billed failed calls) cost ${cost.usd:.4f}. This counts every label"
        " in the store for this prompt, so it can include labels the student was not trained"
        " on.",
        "",
    ]


def _use(kind: ArtifactKind, meta: Mapping[str, object], repo_id: str | None) -> list[str]:
    repo = repo_id or "<namespace>/<name>"
    if kind == "bundle":
        return [
            "## Use it",
            "",
            "Serve it with Anvil (ONNX Runtime, no torch); inputs below the threshold escalate to"
            " the teacher:",
            "",
            "```bash",
            f"hf download {repo} --local-dir student",
            "anvil serve student",
            "curl -s localhost:8000/predict -H 'content-type: application/json' \\",
            '  -d \'{"texts": ["your text here"]}\'',
            "```",
            "",
        ]
    system = _str(meta.get("system")) or ""
    new_tokens = _int(meta.get("max_new_tokens"))
    outputs = meta.get("outputs")
    fields = [str(x) for x in outputs] if isinstance(outputs, list) else []
    answer = (
        f"The answer is the bare `{fields[0]}` value."
        if len(fields) == 1
        else f"The answer is a JSON object with the fields {', '.join(f'`{x}`' for x in fields)}."
    )
    return [
        "## Use it",
        "",
        "Prompt it with the system prompt it was trained with and decode greedily. " + answer,
        "",
        "```python",
        "from transformers import AutoModelForCausalLM, AutoTokenizer",
        "",
        f"repo = {json.dumps(repo)}",
        "tokenizer = AutoTokenizer.from_pretrained(repo)",
        "model = AutoModelForCausalLM.from_pretrained(repo)",
        "messages = [",
        f'    {{"role": "system", "content": {json.dumps(system, ensure_ascii=False)}}},',
        '    {"role": "user", "content": "your input here"},',
        "]",
        "inputs = tokenizer.apply_chat_template(",
        '    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True',
        ")",
        f"out = model.generate(**inputs, max_new_tokens={new_tokens or 128}, do_sample=False)",
        'print(tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))',
        "```",
        "",
    ]


def _limitations(kind: ArtifactKind, facts: _Facts) -> list[str]:
    out = [
        "## Limitations",
        "",
        "- Every parity number is agreement with the teacher on held-out inputs drawn like the"
        " training inputs, not accuracy against ground truth: where the teacher is wrong, a"
        " student that agrees with it is wrong too, and inputs unlike the training pool may"
        " agree less.",
    ]
    teacher = facts.teacher or ""
    if teacher.startswith("fake-"):
        out.append(
            f"- The teacher was `{teacher}`, Anvil's offline keyword-rule fake used for demos and"
            " tests, not an LLM: these numbers say nothing about agreement with a real model."
        )
    elif teacher == GOLD_TEACHER:
        out.append(
            "- The teacher was `gold`: the dataset's gold labels stood in for an LLM, so parity"
            " here is agreement with those labels, not with an LLM."
        )
    if facts.student_model == TINY:
        out.append(
            "- The student is `tiny`, a small random-init model Anvil uses for tests and demos,"
            " not a pretrained one."
        )
    if kind == "bundle":
        out.append(
            "- The student only answers with the labels it was trained on: "
            + ", ".join(f"`{x}`" for x in facts.labels)
            + "."
        )
    return [*out, ""]


def render_card(
    kind: ArtifactKind,
    meta: Mapping[str, object],
    report: Mapping[str, object] | None = None,
    cost: LabelCost | None = None,
    repo_id: str | None = None,
) -> str:
    """The Hugging Face model card (README.md) for an export, from its meta, its eval report
    and what its labels cost. Deterministic: the same inputs give the same text."""
    facts = _facts(kind, meta)
    what = "an ONNX text classifier" if kind == "bundle" else "a text generator (merged LoRA)"
    does = (
        "It labels a text with one of " + ", ".join(f"`{x}`" for x in facts.labels) + "."
        if kind == "bundle"
        else "It writes " + ", ".join(f"`{x}`" for x in facts.outputs) + " for an input."
    )
    lines = [
        _front_matter(kind, facts),
        f"# {facts.task or 'Anvil student'}",
        "",
        f"A student distilled from {_code(facts.teacher)} for the task {_code(facts.task)}:"
        f" {what} built on {_code(facts.student_model)} with Anvil (`anvil-distil`). {does}",
        "",
        *_parity(kind, facts, report),
        *(_calibration(meta) if kind == "bundle" else []),
        *_export_parity(kind, meta),
        *_training(kind, meta, facts),
        *_cost(cost),
        *_use(kind, meta, repo_id),
        *_limitations(kind, facts),
    ]
    return "\n".join(lines).rstrip("\n") + "\n"


def _read_json(path: Path) -> dict[str, object]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExportError(
            f"{path} is unreadable ({exc}); export again with `anvil export`"
        ) from exc
    if not isinstance(data, dict):
        raise ExportError(f"{path} is not a JSON object; export again with `anvil export`")
    return _dict(data)


def _artifact(artifact: Path) -> tuple[ArtifactKind, dict[str, object]]:
    if not artifact.is_dir():
        raise ConfigError(
            f"{artifact} is not a directory; pass the directory `anvil export` wrote"
        )
    if (artifact / BUNDLE_META).is_file():
        return "bundle", read_bundle_meta(artifact)
    if (artifact / MERGED_META).is_file():
        meta = _read_json(artifact / MERGED_META)
        if meta.get("format") != MERGED_FORMAT:
            raise ExportError(
                f"{artifact} has merged export format {meta.get('format')!r}; export it again"
                " with this version of Anvil"
            )
        return "merged", meta
    hint = (
        " (it is a trained student directory)"
        if any((artifact / name).is_file() for name in _STUDENT_METAS)
        else ""
    )
    raise ConfigError(
        f"{artifact} is not an exported student (no {BUNDLE_META} or {MERGED_META}){hint};"
        " export it first with `anvil export`"
    )


def _workspaces(artifact: Path, home: Path, task: str | None) -> list[Workspace]:
    """Where the export's task state can be: two levels up (`<workspace>/exports/<run>`, which
    also finds the eval harness's workspaces), then `<home>/<task>`."""
    roots = [artifact.resolve().parent.parent]
    if task is not None:
        roots.append((home / task).resolve())
    return [Workspace(r) for r in dict.fromkeys(roots)]


def _matches(report: Mapping[str, object], facts: _Facts) -> bool:
    """The report measures this student: same teacher and prompt, and the same run when both
    name one (a report for a sibling run of the same task would put its numbers on this card)."""
    same_run = facts.run is None or report.get("student") in (None, facts.run)
    return same_run and (report.get("teacher_model"), report.get("prompt_hash")) == (
        facts.teacher,
        facts.prompt_hash,
    )


def _find_report(
    artifact: Path, workspaces: Sequence[Workspace], facts: _Facts
) -> dict[str, object] | None:
    # Bundles exported before the meta named their run are found by calibration or directory.
    runs = [r for r in (facts.run, artifact.resolve().name) if r]
    for ws in workspaces:
        for run in dict.fromkeys(runs):
            path = ws.reports / f"eval-{run}.json"
            if not path.is_file():
                continue
            report = _read_json(path)
            if not _matches(report, facts):
                LOG.warning(
                    "eval report is for another teacher or prompt; not used",
                    extra={"report": str(path)},
                )
                continue
            return report
    return None


def _label_cost(workspaces: Sequence[Workspace], facts: _Facts) -> LabelCost | None:
    if facts.teacher is None or facts.prompt_hash is None:
        return None
    for ws in workspaces:
        if not ws.db.is_file():
            continue
        seed = _int(facts.splits.get("seed")) or 0
        frac = _num(facts.splits.get("heldout_fraction")) or 0.25
        with Store(ws.db, seed, frac) as store:
            stats = store.stats(facts.teacher, facts.prompt_hash)
        return LabelCost(labels=stats.labels, usd=stats.cost_usd)
    return None


def load_card_inputs(artifact: Path, home: Path, report: Path | None = None) -> CardInputs:
    """Read an export (an ONNX bundle or a merged generator), its eval report and its label
    cost. The report is `report` when given, else `eval-<run>.json` in the reports of the
    first workspace that has one: the export's own (`<workspace>/exports/<run>`), then
    `<home>/<task>`. The label cost comes from the first of those with a store."""
    kind, meta = _artifact(artifact)
    facts = _facts(kind, meta)
    workspaces = _workspaces(artifact, home, facts.task)
    if report is not None:
        if not report.is_file():
            raise ConfigError(f"--report {report} does not exist; run `anvil eval` first")
        found: dict[str, object] | None = _read_json(report)
        if found is not None and not _matches(found, facts):
            raise ConfigError(
                f"{report} measures student {found.get('student')!r} with teacher"
                f" {found.get('teacher_model')!r} and prompt {found.get('prompt_hash')!r}, but"
                f" {artifact} is student {facts.run!r} trained on {facts.teacher!r} and"
                f" {facts.prompt_hash!r}; pass this student's report"
            )
    else:
        found = _find_report(artifact, workspaces, facts)
    return CardInputs(artifact, kind, meta, found, _label_cost(workspaces, facts))


def check_repo_id(repo_id: str) -> str:
    if not _REPO_ID.fullmatch(repo_id) or len(repo_id.split("/")[1]) > 96:
        raise ConfigError(
            f"--repo {repo_id!r} is not a Hub model id; use namespace/name (letters, digits,"
            " '-', '_' and '.')"
        )
    return repo_id


def artifact_files(artifact: Path) -> list[str]:
    """Files a push uploads: everything in the export except hidden files."""
    return sorted(
        p.relative_to(artifact).as_posix()
        for p in artifact.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(artifact).parts)
    )


def push(
    artifact: Path,
    repo_id: str,
    *,
    home: Path,
    token: str | None,
    private: bool = True,
    dry_run: bool = False,
    report: Path | None = None,
    api: HubApi | None = None,
) -> PushResult:
    """Render the card into `artifact/README.md` (from `report` or the eval report
    `load_card_inputs` finds) and upload the export to the Hub model repo
    `repo_id` (created if missing). The repo ends up holding exactly the export: files a
    previous push left that this export lacks are deleted. A dry run stops after the card."""
    check_repo_id(repo_id)
    if not dry_run and not token:
        raise ConfigError(
            "pushing needs a Hugging Face token with write access: set HF_TOKEN (or use"
            " --dry-run to only render the card)"
        )
    inputs = load_card_inputs(artifact, home, report)
    card = artifact / CARD_FILE
    card.write_text(
        render_card(inputs.kind, inputs.meta, inputs.report, inputs.cost, repo_id),
        encoding="utf-8",
    )
    files = artifact_files(artifact)
    if dry_run:
        return PushResult(repo_id, None, files, card, dry_run=True)
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    facts = _facts(inputs.kind, inputs.meta)
    message = (
        f"Upload {facts.task or 'Anvil'} student {facts.run or artifact.name} ({inputs.kind})"
    )
    try:
        url = api.create_repo(repo_id, token=token, private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            folder_path=artifact,
            commit_message=message,
            token=token,
            allow_patterns=files,
            delete_patterns="*",
        )
    except Exception as exc:  # huggingface_hub, httpx and OSError all surface here
        raise HubError(f"push to {repo_id} failed: {type(exc).__name__}: {exc}") from exc
    repo_url = str(url) if isinstance(url, str) else f"https://huggingface.co/{repo_id}"
    LOG.info(
        "pushed",
        extra={"repo": repo_id, "files": len(files), "private": private, "kind": inputs.kind},
    )
    return PushResult(repo_id, repo_url, files, card, dry_run=False)
