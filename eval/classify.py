"""`make eval-classify`: student-vs-teacher parity on a real classification dataset.

One run: draw a seeded sample of a pinned dataset, label the held-out split and the pool with
the teacher, label a subset of the held-out split a second time under a paraphrased prompt
(the teacher's self-agreement ceiling), train a fresh student on the pool labels, and measure
its agreement with the teacher on the held-out split (kappa with a bootstrap CI, majority and
TF-IDF baselines, per-class scores, confusions, slices by confidence and length). Gold labels
give the secondary table: teacher vs gold and student vs gold.

Teachers:
- `claude`: live labels (needs a key). They are written to `eval/labels/` so every later run
  can replay them for free, together with the inputs the teacher failed on (a refusal, a
  truncated or invalid answer). A recorded failure is not asked again; delete its row to retry.
- `replay`: the committed labels from an earlier `claude` run; fully offline. Recorded failures
  replay as failures, so the held-out and pool sets match the live run's.
- `gold`: the dataset's own labels as the teacher. Runs offline without a key and checks the
  whole harness and the student on real data, but it is not a parity number: its rows say
  `gold` and the ceiling is not applicable.
- `auto` (the default): `replay` when recorded labels exist (the main labels and, when the
  ceiling is measured, the paraphrase labels), else `claude` when a key is set, else an error
  that says what to do.

Raw output goes to `eval/results/classify-<dataset>-<teacher>/`: `report.json` (every number),
`meta.json` (command, git SHA, models, seed, wall time, dollars), `train-metrics.jsonl`.
`python -m eval.render` turns those into RESULTS.md tables. A run with `--student`, `--epochs`,
`--size` or a non-default `--resamples` is a smoke run: it writes to `...-smoke/`, which the
renderer ignores, so it can never overwrite the numbers RESULTS.md is built from (`--out` is
refused for a smoke run unless it names a `...-smoke` directory too).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from anvil import __version__, log
from anvil.errors import AnvilError, ConfigError
from anvil.evaluate.bootstrap import bootstrap_ci, bootstrap_difference_ci
from anvil.evaluate.metrics import Agreement, accuracy, self_agreement, top_confusions
from anvil.pipeline import Workspace, evaluate, label, train
from anvil.settings import Settings
from anvil.spec import TaskSpec, TeacherSpec
from anvil.store import Split
from anvil.students.encoder import EncoderClassifier
from anvil.teacher import Teacher, build_teacher
from anvil.teacher.base import Failure, Label
from anvil.teacher.labelfile import (
    GOLD,
    NOT_RECORDED,
    LabelFileTeacher,
    RecordedThenLiveTeacher,
    read_label_file,
    write_label_file,
)
from eval.data import (
    NEAR_DUPLICATE_JACCARD,
    DatasetRef,
    Sample,
    load_rows,
    near_duplicates,
    sample,
    write_inputs,
)
from eval.render import SEED_SUFFIX, SMOKE_SUFFIX

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "eval" / "classify"
LABELS_DIR = ROOT / "eval" / "labels"
RESULTS_DIR = ROOT / "eval" / "results"
TeacherKind = Literal["auto", "claude", "ollama", "replay", "gold"]
LIVE_KINDS = ("claude", "ollama")
"""Teacher kinds that label with a model and record what they label under `eval/labels/`."""
DEFAULT_RESAMPLES = 1000

logger = log.get_logger("anvil.eval.classify")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetCfg(_Strict):
    name: str
    repo: str
    revision: str
    split: str = "train"
    license: str = ""

    def ref(self) -> DatasetRef:
        return DatasetRef(self.repo, self.revision, self.split, license=self.license)


class SampleCfg(_Strict):
    size: int = Field(ge=10)
    pool_labels: int = Field(ge=1)
    heldout_labels: int = Field(ge=1)
    seed: int = 0


class CeilingCfg(_Strict):
    n: int = Field(default=200, ge=0)
    instructions: str


class OllamaCfg(_Strict):
    """The local teacher `--teacher ollama` uses in place of the spec's Claude model; the
    prompt (and so the prompt hash) is the spec's own, so the two teachers' labels compare."""

    model: str = "qwen2.5:3b"
    digest: str | None = None


class EvalConfig(_Strict):
    dataset: DatasetCfg
    sample: SampleCfg
    ceiling: CeilingCfg
    spec: TaskSpec
    ollama: OllamaCfg = OllamaCfg()

    def with_local_teacher(self) -> EvalConfig:
        """The same eval with the Ollama model as the teacher."""
        teacher = TeacherSpec(
            provider="ollama",
            model=self.ollama.model,
            effort=self.spec.teacher.effort,
            max_tokens=self.spec.teacher.max_tokens,
            digest=self.ollama.digest,
        )
        return self.model_copy(update={"spec": self.spec.model_copy(update={"teacher": teacher})})

    def paraphrase_spec(self) -> TaskSpec:
        """The same task under the paraphrased instructions (a different prompt hash)."""
        return self.spec.model_copy(update={"instructions": self.ceiling.instructions})


class _Dataset(Protocol):
    @property
    def name(self) -> str: ...


class _Ceiling(Protocol):
    @property
    def n(self) -> int: ...


class LabelledEval(Protocol):
    """What the teacher helpers below need from an eval config; `eval.extract` has its own
    config class and shares them."""

    @property
    def dataset(self) -> _Dataset: ...

    @property
    def ceiling(self) -> _Ceiling: ...

    @property
    def spec(self) -> TaskSpec: ...

    def paraphrase_spec(self) -> TaskSpec: ...


def load_config(path: Path) -> EvalConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"eval config {path} not found") from exc
    config = EvalConfig.model_validate(raw)
    # Validate the spec through the same path `anvil.yaml` takes (pydantic re-runs checks).
    TaskSpec.model_validate(config.spec.model_dump())
    return config


def file_safe(model: str) -> str:
    """A model id as a file-name part: a local tag's `:` (`qwen2.5:3b`) is not portable."""
    return re.sub(r"[^A-Za-z0-9._-]", "-", model)


def label_file(dataset: str, model: str, prompt_hash: str) -> Path:
    return LABELS_DIR / dataset / f"{file_safe(model)}-{prompt_hash}.jsonl"


def _shown(path: Path) -> str:
    """A path relative to the repo when it is inside it (error messages stay short)."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def record_labels(
    path: Path,
    labels: Sequence[Label],
    failures: Sequence[Failure] = (),
    key: tuple[str, str] = ("", ""),
) -> int:
    """Merge `labels` and `failures` into the recorded label file at `path` and return its new
    row count.

    `failures` are under `key`, (teacher model, prompt hash) (`Store.failure_records`: the
    latest failure per input that has no label). New rows replace old ones with the same
    input id and every other recorded row is kept: a small run (a `--size 100` smoke test)
    must add to a full recording, never truncate it. A label always wins over a failure for
    the same input, whichever is newer: a failure is recorded only so that a replay can
    reproduce it, and a paid answer is never dropped for one.
    """
    if failures and not all(key):
        raise ValueError("record_labels needs the (teacher model, prompt hash) of the failures")
    merged_labels: dict[str, Label] = {}
    merged_failures: dict[str, tuple[Failure, str, str]] = {}
    if path.exists():
        for r in read_label_file(path):
            if r.label is not None:
                merged_labels[r.input_id] = Label(
                    r.input_id,
                    r.label,
                    r.teacher_model,
                    r.prompt_hash,
                    r.input_tokens,
                    r.output_tokens,
                    r.cost_usd,
                    compute_seconds=r.compute_seconds,
                )
            else:
                failure = Failure(
                    r.input_id,
                    r.reason or "",
                    r.input_tokens,
                    r.output_tokens,
                    r.cost_usd,
                    compute_seconds=r.compute_seconds,
                )
                merged_failures[r.input_id] = (failure, r.teacher_model, r.prompt_hash)
    for lb in labels:
        merged_labels[lb.input_id] = lb
    for f in failures:
        merged_failures[f.input_id] = (f, *key)
    return write_label_file(path, merged_labels.values(), merged_failures.values())


def resolve_teacher_kind(kind: TeacherKind, config: LabelledEval) -> TeacherKind:
    """`auto` replays only when every label file the run reads is recorded; a missing
    paraphrase file would otherwise fail the ceiling halfway through an offline run."""
    if kind != "auto":
        return kind
    spec, para = config.spec, config.paraphrase_spec()
    needed = [label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash)]
    if config.ceiling.n > 0:
        needed.append(label_file(config.dataset.name, para.teacher.model, para.prompt_hash))
    missing = [p for p in needed if not p.exists()]
    if not missing:
        return "replay"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    raise ConfigError(
        f"no recorded labels at {', '.join(_shown(p) for p in missing)} and no"
        " ANTHROPIC_API_KEY; set a key to label with Claude, or pass --teacher gold to run"
        " the harness offline with the dataset's own labels as the teacher"
    )


def make_teacher(
    kind: TeacherKind, config: LabelledEval, spec: TaskSpec, data: Sample, settings: Settings
) -> Teacher:
    if kind == "gold":
        return LabelFileTeacher.gold(data.gold, spec.prompt_hash)
    if kind == "replay":
        path = label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash)
        if not path.exists():
            raise ConfigError(f"no recorded labels at {_shown(path)}; label with --teacher claude")
        return LabelFileTeacher.from_file(path, spec.teacher.model, spec.prompt_hash)
    if kind in LIVE_KINDS:
        live = build_teacher(spec, settings, provider="ollama" if kind == "ollama" else "claude")
        path = label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash)
        if path.exists():  # pay only for inputs the committed file does not cover
            recorded = LabelFileTeacher.from_file(path, spec.teacher.model, spec.prompt_hash)
            return RecordedThenLiveTeacher(recorded, live)
        return live
    raise ConfigError(f"unresolved teacher kind {kind!r}")


def require_complete(
    kind: TeacherKind, failures: Sequence[Failure], config: LabelledEval, spec: TaskSpec
) -> None:
    """Recorded labels must cover every input the run labels: a partial file (a smoke run
    recorded before a full one) would otherwise score a student on a fraction of the sample
    and write that as a result. A recorded failure (the teacher refused when the recording
    was made) is part of the recording, not a hole in it: the live run had no label for that
    input either, so the replay leaves it out of the held-out and pool sets just as it did."""
    missing = sum(1 for f in failures if f.reason == NOT_RECORDED)
    if missing and kind in ("replay", "gold"):
        path = label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash)
        raise ConfigError(
            f"{missing} inputs have no recorded label in {_shown(path)}; the recording is"
            " incomplete, so re-label with --teacher claude (only the missing inputs are paid"
            " for) or run with --teacher gold"
        )


def _tfidf_baseline(
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    student_predicted: list[str],
    n_resamples: int,
    seed: int,
) -> tuple[dict[str, object], list[str]]:
    """A TF-IDF + logistic regression student on the same teacher labels: what the encoder
    adds over a bag of words. Both students are scored on the same held-out items, so the
    comparison is a paired bootstrap of the kappa difference (student minus TF-IDF), not a
    look at whether two separate intervals overlap. Returns the stats and the predictions."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if len(set(train_labels)) < 2:
        return {"skipped": "fewer than two classes in the training labels"}, []
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    x_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train, train_labels)
    predicted = [str(p) for p in clf.predict(vec.transform(test_texts))]
    ci = bootstrap_ci(test_labels, predicted, "kappa", n_resamples=n_resamples, seed=seed)
    diff = bootstrap_difference_ci(
        test_labels, student_predicted, predicted, "kappa", n_resamples=n_resamples, seed=seed
    )
    return {
        "kappa": ci.estimate,
        "kappa_ci": asdict(ci),
        "accuracy": accuracy(test_labels, predicted),
        "student_minus_tfidf_kappa": asdict(diff),
    }, predicted


def _pair_stats(a: list[str], b: list[str], n_resamples: int, seed: int) -> dict[str, object]:
    ci = bootstrap_ci(a, b, "kappa", n_resamples=n_resamples, seed=seed)
    return {"n": len(a), "kappa": ci.estimate, "kappa_ci": asdict(ci), "accuracy": accuracy(a, b)}


def teacher_errors(
    gold: Sequence[str],
    teacher: Sequence[str],
    student: Sequence[str],
    n_resamples: int,
    seed: int,
    k: int = 10,
) -> dict[str, object]:
    """What the student does with a fallible teacher's mistakes, on the held-out split.

    - `student_minus_teacher_vs_gold`: the paired bootstrap interval of kappa(gold, student)
      minus kappa(gold, teacher). Both are scored on the same items, so this, not two
      overlapping intervals, says whether the student agrees with gold more than its teacher.
    - `on_teacher_errors`: of the items the teacher got wrong, how many the student labels as
      the teacher did (`copied`), as gold does (`corrected`) or as neither (`other`); and on
      the items the teacher got right, how many the student gets wrong (`broken`).
    - `confusions`: the teacher's top (gold -> teacher) confusions with how often the student
      makes the same (gold -> student) one, then the student's own top confusions.
    """
    kappa_diff = bootstrap_difference_ci(
        gold, student, teacher, "kappa", n_resamples=n_resamples, seed=seed
    )
    acc_diff = bootstrap_difference_ci(
        gold, student, teacher, "accuracy", n_resamples=n_resamples, seed=seed
    )
    copied = corrected = other = broken = right = 0
    for g, t, st in zip(gold, teacher, student, strict=True):
        if t == g:
            right += 1
            broken += st != g
        elif st == t:
            copied += 1
        elif st == g:
            corrected += 1
        else:
            other += 1
    student_cells = Counter((g, st) for g, st in zip(gold, student, strict=True) if g != st)
    teacher_top = top_confusions(gold, teacher, k)
    return {
        "student_minus_teacher_vs_gold_kappa": asdict(kappa_diff),
        "student_minus_teacher_vs_gold_accuracy": asdict(acc_diff),
        "on_teacher_errors": {
            "n": copied + corrected + other,
            "copied": copied,
            "corrected": corrected,
            "other": other,
        },
        "on_teacher_correct": {"n": right, "broken": broken},
        "confusions": {
            "teacher": [
                {
                    "gold": c.teacher,
                    "label": c.student,
                    "teacher": c.count,
                    "student": student_cells.get((c.teacher, c.student), 0),
                }
                for c in teacher_top
            ],
            "student": [
                {"gold": c.teacher, "label": c.student, "student": c.count}
                for c in top_confusions(gold, student, k)
            ],
        },
    }


def on_ceiling_items(
    first: dict[str, str],
    repeat: dict[str, str],
    student: dict[str, str],
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """Student-vs-teacher on the items the ceiling was measured on, so the two kappas share
    their items, and split by whether the teacher gave the same label under both prompts
    (`stable`) or not: disagreement with a consistent teacher is the student's error,
    disagreement where the teacher is inconsistent is mostly the teacher's."""
    ids = sorted(first.keys() & repeat.keys() & student.keys())
    a = [first[i] for i in ids]
    b = [student[i] for i in ids]
    stable = [i for i in ids if first[i] == repeat[i]]
    unstable = [i for i in ids if first[i] != repeat[i]]

    def agree(sub: list[str]) -> dict[str, object]:
        hits = sum(first[i] == student[i] for i in sub)
        return {"n": len(sub), "agree": hits, "rate": hits / len(sub) if sub else None}

    return {
        "student_vs_teacher": _pair_stats(a, b, n_resamples, seed),
        "stable": agree(stable),
        "unstable": agree(unstable),
    }


def compute_seconds(rows: Sequence[Label | Failure]) -> dict[str, object]:
    """A local teacher's server seconds over the labels it produced (none for Claude)."""
    seconds = sorted(r.compute_seconds for r in rows if r.compute_seconds is not None)
    if not seconds:
        return {}
    return {
        "total": round(sum(seconds), 1),
        "median_per_label": round(seconds[len(seconds) // 2], 3),
        "labels": len(seconds),
    }


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def git_provenance() -> dict[str, object]:
    """The commit and whether the tree differs from it: a modified tracked file or an untracked
    one (ignored files such as `.anvil/` do not count), since a harness edit that has not been
    committed yet is exactly what a run's SHA must not hide. Taken once per invocation, before
    any result is written: a multi-dataset run would otherwise see its own earlier output as
    uncommitted changes."""
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
    }


def workspace(settings: Settings, spec: TaskSpec, data: Sample) -> Workspace:
    """One workspace per sample, so a smoke run never mixes its inputs into a full run's store."""
    return Workspace(settings.home / "eval" / f"{spec.name}-{data.sha256[:12]}")


def run(
    config: EvalConfig,
    data: Sample,
    kind: TeacherKind,
    out_dir: Path,
    settings: Settings,
    n_resamples: int = 1000,
    command: str = "",
    provenance: dict[str, object] | None = None,
    label_only: bool = False,
) -> dict[str, object]:
    """One full eval; writes `report.json`, `meta.json` and `train-metrics.jsonl` to
    `out_dir` and returns the report. `provenance` (from `git_provenance`) is taken now when
    not given. With `label_only` it stops once the labels (and the paraphrase labels) are
    recorded and returns their counts: a slow local teacher can label on one machine and the
    student train later, or elsewhere, from the committed files."""
    started = time.perf_counter()
    prov = provenance if provenance is not None else git_provenance()
    kind = resolve_teacher_kind(kind, config)
    spec = config.spec
    seed = spec.student.seed
    ws = workspace(settings, spec, data)
    inputs = ws.root / "inputs.jsonl"
    write_inputs(inputs, data.items)

    teacher = make_teacher(kind, config, spec, data, settings)
    # `label` adds N *more* pool labels; an eval wants N in total, so a re-run (or one that
    # was interrupted) tops up to the configured count instead of growing past it.
    with ws.open_store(spec) as store:
        pool_done = len(store.labelled(teacher.model_id, spec.prompt_hash, Split.POOL))
    labelled = label(
        spec,
        ws,
        teacher,
        inputs,
        max(0, config.sample.pool_labels - pool_done),
        config.sample.heldout_labels,
    )
    spent = labelled.spent_usd
    failures = len(labelled.failures)
    require_complete(kind, labelled.failures, config, spec)

    with ws.open_store(spec) as store:
        heldout = store.labelled(teacher.model_id, spec.prompt_hash, Split.HELDOUT)
        pool = store.labelled(teacher.model_id, spec.prompt_hash, Split.POOL)
        if kind in LIVE_KINDS:
            record_labels(
                label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash),
                store.label_records(spec.teacher.model, spec.prompt_hash),
                store.failure_records(spec.teacher.model, spec.prompt_hash),
                (spec.teacher.model, spec.prompt_hash),
            )

    ceiling: dict[str, object]
    para = config.paraphrase_spec()
    first: dict[str, str] = {}
    repeat: dict[str, str] = {}
    if kind == "gold":
        ceiling = {"skipped": "gold labels are fixed, so the teacher agrees with itself"}
    elif config.ceiling.n == 0:
        ceiling = {"skipped": "ceiling.n is 0"}
    else:
        first = {it.id: lab for it, lab in heldout[: config.ceiling.n]}
        second = make_teacher(kind, config, para, data, settings)
        with ws.open_store(para) as store:
            # Label only what an earlier run has not, so re-runs do not pile up duplicates.
            done = store.labels(second.model_id, para.prompt_hash)
            todo = [it for it, _ in heldout[: config.ceiling.n] if it.id not in done]
            if todo:
                batch = second.label(todo)
                spent += batch.spent_usd
                failures += len(batch.failures)
                require_complete(kind, batch.failures, config, para)
                store.append_labels(batch.labels)
                store.append_failures(batch.failures, second.model_id, para.prompt_hash)
            repeated = store.labels(second.model_id, para.prompt_hash)
            if kind in LIVE_KINDS:
                record_labels(
                    label_file(config.dataset.name, para.teacher.model, para.prompt_hash),
                    store.label_records(para.teacher.model, para.prompt_hash),
                    store.failure_records(para.teacher.model, para.prompt_hash),
                    (para.teacher.model, para.prompt_hash),
                )
        repeat = {i: lab for i, lab in repeated.items() if i in first}
        if not repeat:
            path = label_file(config.dataset.name, para.teacher.model, para.prompt_hash)
            raise ConfigError(
                f"the paraphrase labels ({_shown(path)}) cover none of the {len(first)}"
                " held-out inputs the ceiling is measured on; re-label with --teacher claude"
            )
        c = self_agreement(first, repeat, n_resamples=n_resamples, seed=seed)
        ceiling = {"paraphrase_prompt_hash": para.prompt_hash, **asdict(c)}

    if label_only:
        return {
            "dataset": {"name": config.dataset.name},
            "teacher": {
                "model": teacher.model_id,
                "pool_labels": len(pool),
                "heldout_labels": len(heldout),
                "failures": failures,
            },
            "ceiling": ceiling,
            "label_only": True,
        }

    trained = train(spec, ws, teacher.model_id)
    evaluated = evaluate(spec, ws, trained.path, n_resamples=n_resamples, seed=seed)
    if not isinstance(evaluated.agreement, Agreement):
        raise ConfigError(f"eval.classify needs a classify task, {spec.name!r} is {spec.kind}")

    texts = [it.text for it, _ in heldout]
    teacher_labels = [lab for _, lab in heldout]
    student = EncoderClassifier.load(trained.path)
    probs = student.predict_proba(texts)
    predicted = [student.labels[i] for i in probs.argmax(axis=1)]
    gold_labels = [data.gold[it.id] for it, _ in heldout]
    baseline, tfidf_predicted = _tfidf_baseline(
        [it.text for it, _ in pool],
        [lab for _, lab in pool],
        texts,
        teacher_labels,
        predicted,
        n_resamples,
        seed,
    )
    gold: dict[str, object] = {
        "student_vs_gold": _pair_stats(gold_labels, predicted, n_resamples, seed),
    }
    if kind != "gold":
        gold["teacher_vs_gold"] = _pair_stats(gold_labels, teacher_labels, n_resamples, seed)
        gold["errors"] = teacher_errors(gold_labels, teacher_labels, predicted, n_resamples, seed)
    if repeat:
        by_id = {it.id: p for (it, _), p in zip(heldout, predicted, strict=True)}
        ceiling["on_same_items"] = on_ceiling_items(first, repeat, by_id, n_resamples, seed)

    with ws.open_store(spec) as store:
        stats = store.stats(teacher.model_id, spec.prompt_hash)
        failure_rows = store.failure_records(teacher.model_id, spec.prompt_hash)
        stored_failures = {f.input_id for f in failure_rows}
        seconds = compute_seconds(
            [*store.label_records(teacher.model_id, spec.prompt_hash), *failure_rows]
        )
    if repeat:
        with ws.open_store(para) as store:
            para_seconds = compute_seconds(
                [
                    *store.label_records(teacher.model_id, para.prompt_hash),
                    *store.failure_records(teacher.model_id, para.prompt_hash),
                ]
            )
        if para_seconds:
            seconds["paraphrase_total"] = para_seconds["total"]
    # A replayed refusal is not stored (it would be counted once per run), but it was billed
    # when recorded and the live run's label cost includes it, so it is added back here.
    replayed_failure_cost = sum(
        f.cost_usd for f in labelled.failures if f.cached and f.input_id not in stored_failures
    )
    wall = time.perf_counter() - started
    report: dict[str, object] = {
        "dataset": {
            "name": config.dataset.name,
            "repo": config.dataset.repo,
            "revision": config.dataset.revision,
            "split": config.dataset.split,
            "license": config.dataset.license,
            "source_rows": data.source_rows,
            "sampled": len(data.items),
            "sample_seed": config.sample.seed,
            "sample_sha256": data.sha256,
            "near_duplicate_heldout": near_duplicates(texts, [it.text for it, _ in pool]),
            "near_duplicate_jaccard": NEAR_DUPLICATE_JACCARD,
        },
        "teacher": {
            "kind": kind,
            "model": teacher.model_id,
            "prompt_hash": spec.prompt_hash,
            "provider": spec.teacher.provider if kind != "gold" else None,
            "digest": spec.teacher.digest if kind != "gold" else None,
            # A local model has no effort setting; only a Claude teacher's matters.
            "effort": (
                spec.teacher.effort
                if kind != "gold" and spec.teacher.provider == "claude"
                else None
            ),
            "pool_labels": len(pool),
            "heldout_labels": len(heldout),
            "failures": failures,
            "label_cost_usd": stats.cost_usd + replayed_failure_cost,
            "compute_seconds": seconds,
        },
        "student": {
            "model": spec.student.model,
            "run": trained.run_id,
            "settings": spec.student.model_dump(),
            "train": asdict(trained.result),
        },
        "agreement": asdict(evaluated.agreement),
        "baselines": {
            "majority": {
                "label": evaluated.agreement.majority_label,
                "accuracy": evaluated.agreement.majority_accuracy,
                "kappa": 0.0,
            },
            "tfidf_logreg": baseline,
        },
        "ceiling": ceiling,
        "gold": gold,
        # Per-item labels, so any later comparison (another baseline, a paired test, an error
        # review) can be done from the committed raw output without the student or the store.
        "predictions": {
            "input_id": [it.id for it, _ in heldout],
            "teacher": teacher_labels,
            "student": predicted,
            "student_confidence": [round(float(c), 4) for c in probs.max(axis=1)],
            "tfidf": tfidf_predicted,
            "gold": gold_labels,
        },
        "spent_usd": spent,
        "wall_seconds": wall,
    }
    meta = {
        "command": command,
        "git_sha": prov["git_sha"],
        "git_dirty": prov["git_dirty"],
        "anvil": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpus": os.cpu_count(),
        "versions": _versions(),
        "seed": seed,
        "n_resamples": n_resamples,
        "teacher_model": teacher.model_id,
        "teacher_digest": spec.teacher.digest if kind != "gold" else None,
        "student_model": spec.student.model,
        "wall_seconds": round(wall, 1),
        "spent_usd": spent,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    metrics = trained.path / "metrics.jsonl"
    if metrics.exists():
        shutil.copyfile(metrics, out_dir / "train-metrics.jsonl")
    logger.info(
        "eval finished",
        extra={"dataset": spec.name, "teacher": teacher.model_id, "wall_s": round(wall, 1)},
    )
    return report


def _versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for pkg in ("torch", "transformers", "numpy", "scikit-learn", "datasets", "anthropic"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            continue
    return out


def results_dir(
    dataset: str, teacher_model: str, smoke: bool = False, seed: int | None = None
) -> Path:
    """`classify-<dataset>-<teacher>`, plus `-seed<N>` for a training-seed repeat (rendered
    in the seed-spread table) or `-smoke` for a run with overridden settings (never rendered)."""
    suffix = SMOKE_SUFFIX if smoke else (f"{SEED_SUFFIX}{seed}" if seed is not None else "")
    return RESULTS_DIR / f"classify-{dataset}-{file_safe(teacher_model)}{suffix}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.classify", description=__doc__)
    parser.add_argument("datasets", nargs="+", help="config names under eval/classify/")
    parser.add_argument(
        "--teacher", choices=["auto", "claude", "ollama", "replay", "gold"], default="auto"
    )
    parser.add_argument(
        "--label-only",
        action="store_true",
        help="label and record the labels under eval/labels/, then stop (no training)",
    )
    parser.add_argument("--student", help="override student.model (e.g. tiny for a smoke run)")
    parser.add_argument("--epochs", type=int, help="override student.epochs")
    parser.add_argument("--size", type=int, help="override sample.size (smoke runs)")
    parser.add_argument(
        "--seed",
        type=int,
        help="repeat with another student seed (init, data order, validation slice)",
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"bootstrap resamples (default {DEFAULT_RESAMPLES}; any other value is a smoke run)",
    )
    parser.add_argument("--out", type=Path, help="results directory (one dataset only)")
    args = parser.parse_args(argv)
    if args.out is not None and len(args.datasets) > 1:
        parser.error("--out takes one dataset; several would overwrite each other")

    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.classify " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    # Any override makes a smoke run: it must not land where RESULTS.md is rendered from.
    smoke = (
        any(v is not None for v in (args.student, args.epochs, args.size))
        or args.resamples != DEFAULT_RESAMPLES
    )
    if smoke and args.out is not None and not args.out.name.endswith(SMOKE_SUFFIX):
        parser.error(
            f"--out {args.out} with overrides (a smoke run) must name a directory ending in"
            f" {SMOKE_SUFFIX}: the renderer skips those, so a smoke run can never replace a"
            " result RESULTS.md is built from"
        )
    try:
        for name in args.datasets:
            config = load_config(CONFIG_DIR / f"{name}.yaml")
            if args.teacher == "ollama":
                config = config.with_local_teacher()
            config = _override(config, args.student, args.epochs, args.size)
            seed = args.seed if args.seed != config.spec.student.seed else None
            if seed is not None:
                student = config.spec.student.model_copy(update={"seed": seed})
                config = config.model_copy(
                    update={"spec": config.spec.model_copy(update={"student": student})}
                )
            rows = load_rows(config.dataset.ref())
            data = sample(name, *rows, n=config.sample.size, seed=config.sample.seed)
            kind = resolve_teacher_kind(args.teacher, config)
            model = GOLD if kind == "gold" else config.spec.teacher.model
            out = args.out or results_dir(name, model, smoke=smoke, seed=seed)
            report = run(
                config,
                data,
                kind,
                out,
                settings,
                args.resamples,
                command,
                provenance=provenance,
                label_only=args.label_only,
            )
            print(summary_line(report, out))  # noqa: T201 - the final summary
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    return 0


def _override(
    config: EvalConfig, student: str | None, epochs: int | None, size: int | None
) -> EvalConfig:
    s_update: dict[str, object] = {}
    if student is not None:
        s_update["model"] = student
    if epochs is not None:
        s_update["epochs"] = epochs
    spec = config.spec.model_copy(
        update={"student": config.spec.student.model_copy(update=s_update)}
    )
    smp = config.sample
    if size is not None:
        ratio = size / smp.size
        smp = smp.model_copy(
            update={
                "size": size,
                "pool_labels": max(1, round(smp.pool_labels * ratio)),
                "heldout_labels": max(1, round(smp.heldout_labels * ratio)),
            }
        )
    # `model_copy` does not validate, so the result goes through the validators again: a
    # negative epoch count or sample size must fail here, not halfway through a run.
    try:
        return EvalConfig.model_validate(
            config.model_copy(update={"spec": spec, "sample": smp}).model_dump()
        )
    except ValidationError as exc:
        raise ConfigError(f"invalid override: {exc}") from exc


def summary_line(report: dict[str, object], out: Path) -> str:
    """The one line `main` prints per dataset (the CLI's final summary)."""
    teacher = report["teacher"]
    dataset = report["dataset"]
    assert isinstance(teacher, dict)
    assert isinstance(dataset, dict)
    if report.get("label_only"):
        return (
            f"{dataset['name']}: {teacher['model']} labelled {teacher['heldout_labels']}"
            f" held-out + {teacher['pool_labels']} pool inputs ({teacher['failures']} failed);"
            " recorded under eval/labels/"
        )
    ag = report["agreement"]
    assert isinstance(ag, dict)
    assert isinstance(teacher, dict)
    assert isinstance(dataset, dict)
    ci = ag["kappa_ci"]
    return (
        f"{dataset['name']}: {teacher['model']} -> student: kappa {ag['kappa']:.3f}"
        f" [{ci['low']:.3f}, {ci['high']:.3f}] on {ag['n']} held-out,"
        f" {teacher['pool_labels']} train labels; raw output {out}"
    )


if __name__ == "__main__":
    raise SystemExit(main())
