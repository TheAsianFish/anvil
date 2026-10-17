"""`make eval-classify`: student-vs-teacher parity on a real classification dataset.

One run: draw a seeded sample of a pinned dataset, label the held-out split and the pool with
the teacher, label a subset of the held-out split a second time under a paraphrased prompt
(the teacher's self-agreement ceiling), train a fresh student on the pool labels, and measure
its agreement with the teacher on the held-out split (kappa with a bootstrap CI, majority and
TF-IDF baselines, per-class scores, confusions, slices by confidence and length). Gold labels
give the secondary table: teacher vs gold and student vs gold.

Teachers:
- `claude`: live labels (needs a key). They are written to `eval/labels/` so every later run
  can replay them for free.
- `replay`: the committed labels from an earlier `claude` run; fully offline.
- `gold`: the dataset's own labels as the teacher. Runs offline without a key and checks the
  whole harness and the student on real data, but it is not a parity number: its rows say
  `gold` and the ceiling is not applicable.
- `auto` (the default): `replay` when recorded labels exist (the main labels and, when the
  ceiling is measured, the paraphrase labels), else `claude` when a key is set, else an error
  that says what to do.

Raw output goes to `eval/results/classify-<dataset>-<teacher>/`: `report.json` (every number),
`meta.json` (command, git SHA, models, seed, wall time, dollars), `train-metrics.jsonl`.
`python -m eval.render` turns those into RESULTS.md tables. A run with `--student`, `--epochs`
or `--size` is a smoke run: it writes to `...-smoke/`, which the renderer ignores, so it can
never overwrite the numbers RESULTS.md is built from.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from anvil import __version__, log
from anvil.errors import AnvilError, ConfigError
from anvil.evaluate.bootstrap import bootstrap_ci
from anvil.evaluate.metrics import accuracy, self_agreement
from anvil.pipeline import Workspace, evaluate, label, train
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.store import Split
from anvil.students.encoder import EncoderClassifier
from anvil.teacher import Teacher, build_teacher
from anvil.teacher.base import Label
from anvil.teacher.labelfile import GOLD, LabelFileTeacher, read_label_file, write_label_file
from eval.data import DatasetRef, Sample, load_rows, sample, write_inputs
from eval.render import SEED_SUFFIX, SMOKE_SUFFIX

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "eval" / "classify"
LABELS_DIR = ROOT / "eval" / "labels"
RESULTS_DIR = ROOT / "eval" / "results"
TeacherKind = Literal["auto", "claude", "replay", "gold"]

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


class EvalConfig(_Strict):
    dataset: DatasetCfg
    sample: SampleCfg
    ceiling: CeilingCfg
    spec: TaskSpec

    def paraphrase_spec(self) -> TaskSpec:
        """The same task under the paraphrased instructions (a different prompt hash)."""
        return self.spec.model_copy(update={"instructions": self.ceiling.instructions})


def load_config(path: Path) -> EvalConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"eval config {path} not found") from exc
    config = EvalConfig.model_validate(raw)
    # Validate the spec through the same path `anvil.yaml` takes (pydantic re-runs checks).
    TaskSpec.model_validate(config.spec.model_dump())
    return config


def label_file(dataset: str, model: str, prompt_hash: str) -> Path:
    return LABELS_DIR / dataset / f"{model}-{prompt_hash}.jsonl"


def _shown(path: Path) -> str:
    """A path relative to the repo when it is inside it (error messages stay short)."""
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def record_labels(path: Path, labels: Sequence[Label]) -> int:
    """Merge `labels` into the recorded label file at `path` and return its new row count.

    New rows replace old ones with the same input id and every other recorded row is kept: a
    small run (a `--size 100` smoke test) must add to a full recording, never truncate it.
    """
    merged: dict[str, Label] = {}
    if path.exists():
        for r in read_label_file(path):
            merged[r.input_id] = Label(
                r.input_id,
                r.label,
                r.teacher_model,
                r.prompt_hash,
                r.input_tokens,
                r.output_tokens,
                r.cost_usd,
            )
    for lb in labels:
        merged[lb.input_id] = lb
    return write_label_file(path, merged.values())


def resolve_teacher_kind(kind: TeacherKind, config: EvalConfig) -> TeacherKind:
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
    kind: TeacherKind, config: EvalConfig, spec: TaskSpec, data: Sample, settings: Settings
) -> Teacher:
    if kind == "gold":
        return LabelFileTeacher.gold(data.gold, spec.prompt_hash)
    if kind == "replay":
        path = label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash)
        if not path.exists():
            raise ConfigError(f"no recorded labels at {_shown(path)}; label with --teacher claude")
        return LabelFileTeacher.from_file(path, spec.teacher.model, spec.prompt_hash)
    if kind == "claude":
        return build_teacher(spec, settings, provider="claude")
    raise ConfigError(f"unresolved teacher kind {kind!r}")


def _tfidf_baseline(
    train_texts: list[str],
    train_labels: list[str],
    test_texts: list[str],
    test_labels: list[str],
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    """A TF-IDF + logistic regression student on the same teacher labels: what the encoder
    adds over a bag of words."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    if len(set(train_labels)) < 2:
        return {"skipped": "fewer than two classes in the training labels"}
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    x_train = vec.fit_transform(train_texts)
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(x_train, train_labels)
    predicted = [str(p) for p in clf.predict(vec.transform(test_texts))]
    ci = bootstrap_ci(test_labels, predicted, "kappa", n_resamples=n_resamples, seed=seed)
    return {
        "kappa": ci.estimate,
        "kappa_ci": asdict(ci),
        "accuracy": accuracy(test_labels, predicted),
    }


def _pair_stats(a: list[str], b: list[str], n_resamples: int, seed: int) -> dict[str, object]:
    ci = bootstrap_ci(a, b, "kappa", n_resamples=n_resamples, seed=seed)
    return {"n": len(a), "kappa": ci.estimate, "kappa_ci": asdict(ci), "accuracy": accuracy(a, b)}


def _git(*args: str) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip()


def git_provenance() -> dict[str, object]:
    """The commit and whether tracked files differ from it. Taken once per invocation, before
    any result is written: a multi-dataset run would otherwise see its own earlier output as
    uncommitted changes."""
    return {
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
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
) -> dict[str, object]:
    """One full eval; writes `report.json`, `meta.json` and `train-metrics.jsonl` to
    `out_dir` and returns the report. `provenance` (from `git_provenance`) is taken now when
    not given."""
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

    with ws.open_store(spec) as store:
        heldout = store.labelled(teacher.model_id, spec.prompt_hash, Split.HELDOUT)
        pool = store.labelled(teacher.model_id, spec.prompt_hash, Split.POOL)
        if kind == "claude":
            record_labels(
                label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash),
                store.label_records(spec.teacher.model, spec.prompt_hash),
            )

    ceiling: dict[str, object]
    para = config.paraphrase_spec()
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
                store.append_labels(batch.labels)
                store.append_failures(batch.failures, second.model_id, para.prompt_hash)
            repeated = store.labels(second.model_id, para.prompt_hash)
            if kind == "claude":
                record_labels(
                    label_file(config.dataset.name, para.teacher.model, para.prompt_hash),
                    store.label_records(para.teacher.model, para.prompt_hash),
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

    trained = train(spec, ws, teacher.model_id)
    evaluated = evaluate(spec, ws, trained.path, n_resamples=n_resamples, seed=seed)

    texts = [it.text for it, _ in heldout]
    teacher_labels = [lab for _, lab in heldout]
    student = EncoderClassifier.load(trained.path)
    predicted = student.predict(texts)
    gold_labels = [data.gold[it.id] for it, _ in heldout]
    baseline = _tfidf_baseline(
        [it.text for it, _ in pool],
        [lab for _, lab in pool],
        texts,
        teacher_labels,
        n_resamples,
        seed,
    )
    gold: dict[str, object] = {
        "student_vs_gold": _pair_stats(gold_labels, predicted, n_resamples, seed),
    }
    if kind != "gold":
        gold["teacher_vs_gold"] = _pair_stats(gold_labels, teacher_labels, n_resamples, seed)

    with ws.open_store(spec) as store:
        stats = store.stats(teacher.model_id, spec.prompt_hash)
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
        },
        "teacher": {
            "kind": kind,
            "model": teacher.model_id,
            "prompt_hash": spec.prompt_hash,
            "effort": spec.teacher.effort if kind != "gold" else None,
            "pool_labels": len(pool),
            "heldout_labels": len(heldout),
            "failures": failures,
            "label_cost_usd": stats.cost_usd,
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
    return RESULTS_DIR / f"classify-{dataset}-{teacher_model}{suffix}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.classify", description=__doc__)
    parser.add_argument("datasets", nargs="+", help="config names under eval/classify/")
    parser.add_argument("--teacher", choices=["auto", "claude", "replay", "gold"], default="auto")
    parser.add_argument("--student", help="override student.model (e.g. tiny for a smoke run)")
    parser.add_argument("--epochs", type=int, help="override student.epochs")
    parser.add_argument("--size", type=int, help="override sample.size (smoke runs)")
    parser.add_argument(
        "--seed",
        type=int,
        help="repeat with another student seed (init, data order, validation slice)",
    )
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", type=Path, help="results directory (one dataset only)")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.classify " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    # Any override makes a smoke run: it must not land where RESULTS.md is rendered from.
    smoke = any(v is not None for v in (args.student, args.epochs, args.size))
    try:
        for name in args.datasets:
            config = load_config(CONFIG_DIR / f"{name}.yaml")
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
                config, data, kind, out, settings, args.resamples, command, provenance=provenance
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
    if student:
        s_update["model"] = student
    if epochs:
        s_update["epochs"] = epochs
    spec = config.spec.model_copy(
        update={"student": config.spec.student.model_copy(update=s_update)}
    )
    smp = config.sample
    if size:
        ratio = size / smp.size
        smp = smp.model_copy(
            update={
                "size": size,
                "pool_labels": max(1, round(smp.pool_labels * ratio)),
                "heldout_labels": max(1, round(smp.heldout_labels * ratio)),
            }
        )
    return config.model_copy(update={"spec": spec, "sample": smp})


def summary_line(report: dict[str, object], out: Path) -> str:
    """The one line `main` prints per dataset (the CLI's final summary)."""
    ag = report["agreement"]
    teacher = report["teacher"]
    dataset = report["dataset"]
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
