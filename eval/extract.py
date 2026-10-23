"""`make eval-extract`: student-vs-teacher parity on a real span extraction dataset.

One run: draw a seeded sample of a pinned word-tokenised NER dataset (MIT Restaurant), label
the held-out split and the pool with the teacher (mentions with character offsets), label a
subset of the held-out split again under a paraphrased prompt (the self-agreement ceiling),
align every answer to its input and to the student's tokens (refusing to train above the
unaligned-rate limit), train a fresh `EncoderTagger` on the pool, and score it against the
teacher on the held-out split: exact-match entity F1 and word-level tag kappa, both with
95% bootstrap intervals that resample inputs, per-type scores, an error breakdown and length
slices. Baselines: tag everything O (kappa 0) and a gazetteer of the teacher's training
spans (what memorising surface forms gets you), compared with a paired interval.

The alignment stress study takes the held-out gold answers, corrupts their offsets and
text the ways a language model gets them wrong (off by one, a few characters off, a word
index instead of a character offset, offset 0, title case, a trailing full stop), and counts
how many spans plain verification (the exact offset or nothing) and the full repair ladder
recover. It needs no teacher and no training, so it runs on every invocation.

Teachers are as in `eval.classify` (`claude`, `replay`, `gold`, `auto`); labels a live run
pays for go to `eval/labels/<dataset>/`. Raw output goes to
`eval/results/extract-<dataset>-<teacher>/` (`report.json`, `meta.json`,
`train-metrics.jsonl`); `python -m eval.render` builds the RESULTS.md table. Overrides
(`--student`, `--epochs`, `--size`, `--resamples`) make a smoke run under `...-smoke/`.

`--predictions <report.json>` skips all of that and runs the cascade study
(`eval.extract_cascade`, M12) on the held-out predictions a run committed, with no model:
coverage against F1, the thresholds at F1 0.9 and 0.95, the split study and which errors the
confidence detects. A full run also runs it on its own report, into
`eval/results/cascade-extract-<dataset>-<teacher>/`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import shutil
import string
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from anvil import __version__, log
from anvil.align import MATCH_RULES, align_mentions
from anvil.errors import AnvilError, ConfigError
from anvil.evaluate.extraction import (
    entity_f1_ci,
    entity_f1_difference_ci,
    entity_scores,
    extraction_self_agreement,
    per_type,
    tag_agreement,
    token_kappa_ci,
)
from anvil.extraction import answer_spans, score_tagger
from anvil.pipeline import Workspace, label, train
from anvil.settings import Settings
from anvil.spans import Mention, Span, decode_answer
from anvil.spec import TaskSpec
from anvil.store import Split
from anvil.students.tagger import EncoderTagger
from anvil.teacher.base import Item
from eval import extract_cascade
from eval.classify import (
    DEFAULT_RESAMPLES,
    CeilingCfg,
    SampleCfg,
    TeacherKind,
    _versions,
    git_provenance,
    label_file,
    make_teacher,
    record_labels,
    require_complete,
    resolve_teacher_kind,
)
from eval.data import (
    NEAR_DUPLICATE_JACCARD,
    Sample,
    TaggedRef,
    load_tagged_rows,
    near_duplicates,
    sample_tagged,
    write_inputs,
)
from eval.render import SMOKE_SUFFIX

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "eval" / "extract"
RESULTS_DIR = ROOT / "eval" / "results"

logger = log.get_logger("anvil.eval.extract")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetCfg(_Strict):
    name: str
    repo: str
    revision: str
    filename: str
    label_map: str
    license: str = ""

    def ref(self) -> TaggedRef:
        return TaggedRef(self.repo, self.revision, self.filename, self.label_map, self.license)


class EvalConfig(_Strict):
    dataset: DatasetCfg
    sample: SampleCfg
    ceiling: CeilingCfg
    spec: TaskSpec

    def paraphrase_spec(self) -> TaskSpec:
        return self.spec.model_copy(update={"instructions": self.ceiling.instructions})


def load_config(path: Path) -> EvalConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigError(f"eval config {path} not found") from exc
    config = EvalConfig.model_validate(raw)
    TaskSpec.model_validate(config.spec.model_dump())
    if config.spec.kind != "extract":
        raise ConfigError(f"{path} must hold an extract task, not {config.spec.kind}")
    return config


# --- gazetteer baseline -----------------------------------------------------------------


def gazetteer(
    train_texts: Sequence[str],
    train_spans: Sequence[Sequence[Span]],
    test_texts: Sequence[str],
    types: Sequence[str],
) -> list[list[Span]]:
    """Tag every test occurrence of a surface form the teacher tagged in training.

    The dictionary maps each lower-cased training span text to its most frequent type (ties
    go to the type listed first in the spec). Matching is case-insensitive on word
    boundaries, longest phrase first, non-overlapping: what a student that memorised the
    training entities, and understood nothing about context, would answer.
    """
    rank = {t: i for i, t in enumerate(types)}
    votes: dict[str, Counter[str]] = {}
    for text, spans in zip(train_texts, train_spans, strict=True):
        for s in spans:
            votes.setdefault(s.text_in(text).lower(), Counter())[s.type] += 1
    if not votes:
        return [[] for _ in test_texts]
    best = {
        phrase: min(c, key=lambda t: (-c[t], rank[t])) for phrase, c in votes.items() if phrase
    }
    phrases = sorted(best, key=lambda p: (-len(p), p))
    # One group per phrase, looked up by group number: under IGNORECASE a match can differ
    # from its phrase by more than `lower()` undoes (a long s matches "s").
    pattern = re.compile(
        r"(?<!\w)(?:" + "|".join(f"({re.escape(p)})" for p in phrases) + r")(?!\w)",
        re.IGNORECASE,
    )
    out: list[list[Span]] = []
    for text in test_texts:
        out.append(
            [
                Span(m.start(), m.end(), best[phrases[(m.lastindex or 1) - 1]])
                for m in pattern.finditer(text)
            ]
        )
    return out


# --- alignment stress study -------------------------------------------------------------

Corruption = Callable[[str, Mention, random.Random], Mention]


def _shift(by: Sequence[int]) -> Corruption:
    def apply(text: str, m: Mention, rng: random.Random) -> Mention:
        delta = rng.choice(by)
        # A shift below 0 is mirrored, not clamped: clamping a span at 0 would hand back
        # the true offset and count a corruption that never happened.
        return Mention(
            m.text, m.type, m.start + delta if m.start + delta >= 0 else m.start - delta
        )

    return apply


def _word_index(text: str, m: Mention, rng: random.Random) -> Mention:
    return Mention(m.text, m.type, len(text[: m.start].split()))


def _zero(text: str, m: Mention, rng: random.Random) -> Mention:
    return Mention(m.text, m.type, 0)


def _title(text: str, m: Mention, rng: random.Random) -> Mention:
    return Mention(m.text.title(), m.type, m.start)


def _full_stop(text: str, m: Mention, rng: random.Random) -> Mention:
    return Mention(m.text + ".", m.type, m.start)


CORRUPTIONS: dict[str, Corruption] = {
    "none": lambda text, m, rng: m,
    "off_by_one": _shift((-1, 1)),
    "off_by_2_to_5": _shift((-5, -4, -3, -2, 2, 3, 4, 5)),
    "word_index_as_offset": _word_index,
    "offset_zero": _zero,
    "title_case": _title,
    "trailing_full_stop": _full_stop,
}
"""Each corruption is applied to every mention of every answer, one corruption at a time."""


def alignment_stress(
    texts: Sequence[str],
    gold: Sequence[Sequence[Span]],
    types: Sequence[str],
    seed: int,
) -> dict[str, object]:
    """Spans recovered by plain verification and by the repair ladder, per corruption.

    Only spans the corruption actually changes are scored: a span at offset 0 is untouched
    by "offset zero" and an already title-cased one by "title case", and counting those as
    recovered would credit plain verification with a repair it never had to make. Each
    corrupted mention is aligned within its sentence next to the other (equally corrupted)
    mentions. A changed span is `recovered` when alignment returns it exactly, `misplaced`
    when its mention was aligned somewhere else (another occurrence of the same words, or
    an exact match of a corrupted text such as "pizza." in "pizza."), and `lost` otherwise.
    The attribution is one-to-one (`_attribute`): an aligned span accounts for at most one
    gold span, so two mentions of the same words cannot both claim one aligned span.
    """
    total = sum(len(g) for g in gold)
    out: dict[str, object] = {"spans": total, "inputs": len(texts), "seed": seed}
    rows: dict[str, object] = {}
    for name, corrupt in CORRUPTIONS.items():
        rng = random.Random(f"{seed}:{name}")
        corrupted: list[tuple[str, list[Span], list[Mention], set[int]]] = []
        changed = 0
        for text, spans in zip(texts, gold, strict=True):
            exact = [Mention(s.text_in(text), s.type, s.start) for s in spans]
            mentions = [corrupt(text, m, rng) for m in exact]
            moved = {i for i, (a, b) in enumerate(zip(exact, mentions, strict=True)) if a != b}
            changed += len(moved)
            corrupted.append((text, list(spans), mentions, moved))
        row: dict[str, object] = {"changed": changed, "unchanged": total - changed}
        for mode, rules in (("verify_only", ("exact",)), ("repair", MATCH_RULES)):
            recovered = misplaced = 0
            for text, spans, mentions, moved in corrupted:
                if not moved:
                    continue
                got = align_mentions(text, mentions, types, rules=rules).spans
                r, m = _attribute(text, spans, moved, got)
                recovered += r
                misplaced += m
            row[mode] = {
                "recovered": recovered,
                "misplaced": misplaced,
                "lost": changed - recovered - misplaced,
                "recovered_rate": recovered / changed if changed else 0.0,
            }
        rows[name] = row
    out["corruptions"] = rows
    return out


_EDGE = string.punctuation + string.whitespace


def _words(text: str, span: Span) -> str:
    return span.text_in(text).strip(_EDGE).lower()


def _attribute(
    text: str, gold: Sequence[Span], moved: set[int], got: Sequence[Span]
) -> tuple[int, int]:
    """(recovered, misplaced) among the changed gold spans `moved`, one aligned span each.

    Aligned spans equal to a gold span are consumed first: by the changed ones (recovered)
    and by the unchanged ones (not scored, but no longer free to explain another span). Each
    remaining changed span then takes the first unconsumed aligned span of its type that
    overlaps it or covers the same words, ignoring case and edge punctuation (misplaced);
    one with nothing left to take was lost.
    """
    free = list(got)
    recovered = 0
    pending: list[Span] = []
    for i, span in enumerate(gold):
        if span in free:
            free.remove(span)
            recovered += i in moved
        elif i in moved:
            pending.append(span)
    misplaced = 0
    for span in pending:
        words = _words(text, span)
        for j, s in enumerate(free):
            if s.type == span.type and (s.overlaps(span) or _words(text, s) == words):
                del free[j]
                misplaced += 1
                break
    return recovered, misplaced


# --- the run ----------------------------------------------------------------------------


def workspace(settings: Settings, spec: TaskSpec, data: Sample) -> Workspace:
    return Workspace(settings.home / "eval" / f"{spec.name}-{data.sha256[:12]}")


def _record(config: EvalConfig, spec: TaskSpec, ws: Workspace) -> None:
    with ws.open_store(spec) as store:
        record_labels(
            label_file(config.dataset.name, spec.teacher.model, spec.prompt_hash),
            store.label_records(spec.teacher.model, spec.prompt_hash),
            store.failure_records(spec.teacher.model, spec.prompt_hash),
            (spec.teacher.model, spec.prompt_hash),
        )


def _ceiling(
    kind: TeacherKind,
    config: EvalConfig,
    data: Sample,
    heldout: Sequence[tuple[Item, str]],
    ws: Workspace,
    settings: Settings,
    n_resamples: int,
    seed: int,
) -> tuple[dict[str, object], float, int]:
    """The teacher's self-agreement on `ceiling.n` held-out inputs; (stats, spent, failures)."""
    if kind == "gold":
        return {"skipped": "gold labels are fixed, so the teacher agrees with itself"}, 0.0, 0
    if config.ceiling.n == 0:
        return {"skipped": "ceiling.n is 0"}, 0.0, 0
    spec, para = config.spec, config.paraphrase_spec()
    subset = list(heldout[: config.ceiling.n])
    second = make_teacher(kind, config, para, data, settings)
    spent, failures = 0.0, 0
    with ws.open_store(para) as store:
        done = store.labels(second.model_id, para.prompt_hash)
        todo = [it for it, _ in subset if it.id not in done]
        if todo:
            batch = second.label(todo)
            spent, failures = batch.spent_usd, len(batch.failures)
            require_complete(kind, batch.failures, config, para)
            store.append_labels(batch.labels)
            store.append_failures(batch.failures, second.model_id, para.prompt_hash)
        repeated = store.labels(second.model_id, para.prompt_hash)
    if kind == "claude":
        _record(config, para, ws)
    pairs_second = [(it, repeated[it.id]) for it, _ in subset if it.id in repeated]
    if not pairs_second:
        raise ConfigError("the paraphrase labels cover none of the ceiling inputs; re-label")
    first_spans, _ = answer_spans(spec, subset)
    second_spans, _ = answer_spans(para, pairs_second)
    first = {it.id: s for (it, _), s in zip(subset, first_spans, strict=True)}
    second_by_id = {it.id: s for (it, _), s in zip(pairs_second, second_spans, strict=True)}
    texts = {it.id: it.text for it, _ in subset}
    c = extraction_self_agreement(texts, first, second_by_id, n_resamples=n_resamples, seed=seed)
    return {"paraphrase_prompt_hash": para.prompt_hash, **asdict(c)}, spent, failures


def _pair(
    texts: Sequence[str],
    reference: Sequence[Sequence[Span]],
    other: Sequence[Sequence[Span]],
    n_resamples: int,
    seed: int,
) -> dict[str, object]:
    scores = entity_scores(reference, other)
    ci = entity_f1_ci(reference, other, n_resamples=n_resamples, seed=seed)
    words = tag_agreement(texts, reference, other)
    kappa_ci = token_kappa_ci(texts, reference, other, n_resamples=n_resamples, seed=seed)
    return {
        "entity": asdict(scores),
        "entity_f1_ci": asdict(ci),
        "token_kappa": words.kappa,
        "token_kappa_ci": asdict(kappa_ci),
    }


def _spans_json(spans: Sequence[Sequence[Span]]) -> list[list[list[object]]]:
    return [[[s.start, s.end, s.type] for s in row] for row in spans]


def run(
    config: EvalConfig,
    data: Sample,
    kind: TeacherKind,
    out_dir: Path,
    settings: Settings,
    n_resamples: int = DEFAULT_RESAMPLES,
    command: str = "",
    provenance: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """One full eval; writes `report.json`, `meta.json` and `train-metrics.jsonl`."""
    started = time.perf_counter()
    prov = dict(provenance) if provenance is not None else git_provenance()
    kind = resolve_teacher_kind(kind, config)
    spec = config.spec
    seed = spec.student.seed
    ws = workspace(settings, spec, data)
    inputs = ws.root / "inputs.jsonl"
    write_inputs(inputs, data.items)

    teacher = make_teacher(kind, config, spec, data, settings)
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
    spent, failures = labelled.spent_usd, len(labelled.failures)
    require_complete(kind, labelled.failures, config, spec)
    if kind == "claude":
        _record(config, spec, ws)
    with ws.open_store(spec) as store:
        heldout = store.labelled(teacher.model_id, spec.prompt_hash, Split.HELDOUT)
        pool = store.labelled(teacher.model_id, spec.prompt_hash, Split.POOL)
        stats = store.stats(teacher.model_id, spec.prompt_hash)
    if not heldout:
        raise ConfigError(
            f"the sample of {len(data.items)} sentences put none in the held-out split, so"
            " there is nothing to score; raise --size (or sample.size in the config)"
        )

    ceiling, c_spent, c_failures = _ceiling(
        kind, config, data, heldout, ws, settings, n_resamples, seed
    )
    spent += c_spent
    failures += c_failures

    trained = train(spec, ws, teacher.model_id)
    student = EncoderTagger.load(trained.path)
    scored = score_tagger(spec, student, heldout, n_resamples=n_resamples, seed=seed)
    texts = [it.text for it, _ in heldout]

    pool_texts = [it.text for it, _ in pool]
    pool_spans, _ = answer_spans(spec, pool)
    gaz = gazetteer(pool_texts, pool_spans, texts, spec.label_names)
    gaz_scores = _pair(texts, scored.teacher, gaz, n_resamples, seed)
    gaz_scores["per_type"] = [asdict(r) for r in per_type(scored.teacher, gaz, spec.label_names)]
    gaz_scores["student_minus_gazetteer_f1"] = asdict(
        entity_f1_difference_ci(
            scored.teacher, scored.predicted, gaz, n_resamples=n_resamples, seed=seed
        )
    )

    gold_spans = [
        align_mentions(it.text, decode_answer(data.gold[it.id]), spec.label_names).spans
        for it, _ in heldout
    ]
    gold: dict[str, object] = {
        "student_vs_gold": _pair(texts, gold_spans, scored.predicted, n_resamples, seed)
    }
    if kind != "gold":
        gold["teacher_vs_gold"] = _pair(texts, gold_spans, scored.teacher, n_resamples, seed)

    alignment = student.meta.get("alignment", {})
    wall = time.perf_counter() - started
    report: dict[str, object] = {
        "dataset": {
            "name": config.dataset.name,
            "repo": config.dataset.repo,
            "revision": config.dataset.revision,
            "file": config.dataset.filename,
            "license": config.dataset.license,
            "source_rows": data.source_rows,
            "sampled": len(data.items),
            "sample_seed": config.sample.seed,
            "sample_sha256": data.sha256,
            "near_duplicate_heldout": near_duplicates(texts, pool_texts),
            "near_duplicate_jaccard": NEAR_DUPLICATE_JACCARD,
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
        "alignment": {"pool": alignment, "heldout": scored.alignment.as_dict()},
        "agreement": asdict(scored.agreement),
        "baselines": {
            "all_o": {
                "entity_f1": 0.0,
                "token_kappa": 0.0,
                "token_accuracy": scored.agreement.all_o_accuracy,
            },
            "gazetteer": gaz_scores,
        },
        "ceiling": ceiling,
        "gold": gold,
        "alignment_stress": alignment_stress(texts, gold_spans, spec.label_names, seed),
        "predictions": {
            "input_id": [it.id for it, _ in heldout],
            "teacher": _spans_json(scored.teacher),
            "student": _spans_json(scored.predicted),
            "student_confidence": [round(float(c), 4) for c in scored.confidence],
            "gazetteer": _spans_json(gaz),
            "gold": _spans_json(gold_spans),
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
    logger.info("eval finished", extra={"dataset": spec.name, "wall_s": round(wall, 1)})
    return report


def results_dir(dataset: str, teacher_model: str, smoke: bool = False) -> Path:
    return RESULTS_DIR / f"extract-{dataset}-{teacher_model}{SMOKE_SUFFIX if smoke else ''}"


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


def summary_line(report: Mapping[str, object], out: Path) -> str:
    ag, teacher, dataset = report["agreement"], report["teacher"], report["dataset"]
    assert isinstance(ag, dict)
    assert isinstance(teacher, dict)
    assert isinstance(dataset, dict)
    f1, ci, kci = ag["entity"]["f1"], ag["entity_f1_ci"], ag["token_kappa_ci"]
    return (
        f"{dataset['name']}: {teacher['model']} -> student: entity F1 {f1:.3f}"
        f" [{ci['low']:.3f}, {ci['high']:.3f}], token kappa {ag['token_kappa']:.3f}"
        f" [{kci['low']:.3f}, {kci['high']:.3f}] on {ag['n']} held-out,"
        f" {teacher['pool_labels']} train labels; raw output {out}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.extract", description=__doc__)
    parser.add_argument("datasets", nargs="*", help="config names under eval/extract/")
    parser.add_argument(
        "--predictions",
        type=Path,
        help="an eval.extract report.json: run the cascade study on its predictions, no model",
    )
    parser.add_argument("--splits", type=int, default=extract_cascade.DEFAULT_SPLITS)
    parser.add_argument("--teacher", choices=["auto", "claude", "replay", "gold"], default="auto")
    parser.add_argument("--student", help="override student.model (e.g. tiny for a smoke run)")
    parser.add_argument("--epochs", type=int, help="override student.epochs")
    parser.add_argument("--size", type=int, help="override sample.size (smoke runs)")
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    parser.add_argument("--out", type=Path, help="results directory (one dataset only)")
    args = parser.parse_args(argv)
    if args.predictions is not None and args.datasets:
        parser.error("--predictions takes no dataset: the report names its own")
    if args.predictions is None and not args.datasets:
        parser.error("name a dataset under eval/extract/, or pass --predictions <report.json>")
    if args.out is not None and len(args.datasets) > 1:
        parser.error("--out takes one dataset; several would overwrite each other")

    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.extract " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    if args.predictions is not None:
        return _cascade_main(args, command, provenance)
    smoke = (
        any(v is not None for v in (args.student, args.epochs, args.size))
        or args.resamples != DEFAULT_RESAMPLES
    )
    if smoke and args.out is not None and not args.out.name.endswith(SMOKE_SUFFIX):
        parser.error(f"--out with overrides (a smoke run) must end in {SMOKE_SUFFIX}")
    try:
        for name in args.datasets:
            config = _override(
                load_config(CONFIG_DIR / f"{name}.yaml"), args.student, args.epochs, args.size
            )
            rows = load_tagged_rows(config.dataset.ref())
            data = sample_tagged(
                name,
                rows,
                n=config.sample.size,
                seed=config.sample.seed,
                types=config.spec.label_names,
            )
            kind = resolve_teacher_kind(args.teacher, config)
            model = "gold" if kind == "gold" else config.spec.teacher.model
            out = args.out or results_dir(name, model, smoke=smoke)
            report = run(config, data, kind, out, settings, args.resamples, command, provenance)
            print(summary_line(report, out))  # noqa: T201 - the final summary
            if not smoke:
                cascade_out = extract_cascade.results_dir(report)
                if args.splits != extract_cascade.DEFAULT_SPLITS:
                    cascade_out = cascade_out.with_name(cascade_out.name + SMOKE_SUFFIX)
                cascade = extract_cascade.run(
                    out / "report.json",
                    cascade_out,
                    n_splits=args.splits,
                    command=f"{command} (cascade study on {out / 'report.json'})",
                    provenance=provenance,
                )
                print(extract_cascade.summary_line(cascade, cascade_out))  # noqa: T201
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    return 0


def _cascade_main(args: argparse.Namespace, command: str, provenance: Mapping[str, object]) -> int:
    try:
        report = json.loads(args.predictions.read_text(encoding="utf-8"))
        smoke = args.splits != extract_cascade.DEFAULT_SPLITS or (
            args.resamples != extract_cascade.DEFAULT_RESAMPLES
        )
        if smoke and args.out is not None and not args.out.name.endswith(SMOKE_SUFFIX):
            print(  # noqa: T201 - the CLI's error line
                f"error: --out with overrides (a smoke run) must end in {SMOKE_SUFFIX}",
                file=sys.stderr,
            )
            return 2
        out = args.out or extract_cascade.results_dir(report)
        if smoke and args.out is None:
            out = out.with_name(out.name + SMOKE_SUFFIX)
        cascade = extract_cascade.run(
            args.predictions,
            out,
            n_splits=args.splits,
            n_resamples=args.resamples,
            command=command,
            provenance=provenance,
        )
    except FileNotFoundError:
        print(f"error: no report at {args.predictions}", file=sys.stderr)  # noqa: T201
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {args.predictions} is not JSON: {exc}", file=sys.stderr)  # noqa: T201
        return 2
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    print(extract_cascade.summary_line(cascade, out))  # noqa: T201 - the final summary
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
