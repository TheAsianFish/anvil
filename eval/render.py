"""Render RESULTS.md tables from the raw eval output under `eval/results/`.

Numbers in RESULTS.md are never typed by hand: each generated block sits between
`<!-- BEGIN generated:<name> -->` and `<!-- END generated:<name> -->` markers and is rebuilt
from the `report.json` and `meta.json` files of every run. Everything outside the markers is
prose and stays as written.

    uv run python -m eval.render          # rewrite RESULTS.md
    uv run python -m eval.render --check  # exit 1 if RESULTS.md is stale (used in tests)
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RESULTS_MD = ROOT / "RESULTS.md"
RESULTS_DIR = ROOT / "eval" / "results"

# Any: report.json is arbitrary JSON written by eval/classify.py.
Report = Mapping[str, Any]


MIN_SUPPORT = 5
"""Held-out items a class needs before it is listed among the weakest classes."""


def _f(x: object, digits: int = 3) -> str:
    return f"{x:.{digits}f}" if isinstance(x, int | float) else "n/a"


def _ci(stats: Mapping[str, Any] | None) -> str:
    if not stats or "kappa" not in stats:
        return "n/a"
    ci = stats.get("kappa_ci") or {}
    return f"{_f(stats['kappa'])} [{_f(ci.get('low'))}, {_f(ci.get('high'))}]"


def _diff(stats: Mapping[str, Any] | None) -> str:
    """A signed difference with its interval, `+0.180 [0.139, 0.222]`."""
    if not stats or "estimate" not in stats:
        return "n/a"
    return f"{stats['estimate']:+.3f} [{_f(stats.get('low'))}, {_f(stats.get('high'))}]"


def _student_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


SMOKE_SUFFIX = "-smoke"
"""Results directories ending in this hold smoke runs (`eval.classify` with overrides)."""


SEED_SUFFIX = "-seed"
"""Results directories ending in `-seed<N>` repeat a run with another student seed."""
_SEED_DIR = re.compile(rf"^(?P<base>.+){SEED_SUFFIX}(?P<seed>\d+)$")


def load_runs(
    results_dir: Path, prefix: str = "classify-", seeds: bool = False
) -> list[tuple[str, Report, Report]]:
    """Every run under `results_dir`, except smoke runs (`...-smoke/`): those used overridden
    settings (another student, fewer epochs, a smaller sample) and are not results. Seed
    repeats (`...-seed<N>/`) are returned only with `seeds=True`."""
    runs = []
    for path in sorted(results_dir.glob(f"{prefix}*/report.json")):
        name = path.parent.name
        if name.endswith(SMOKE_SUFFIX) or (not seeds and _SEED_DIR.match(name)):
            continue
        meta_path = path.parent / "meta.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        runs.append((path.parent.name, report, meta))
    return runs


def render_classify(runs: list[tuple[str, Report, Report]]) -> str:
    if not runs:
        return "No classification runs yet: run `make eval-classify`.\n"
    lines = [
        "| Dataset | Teacher | Student | Teacher labels | Held-out | Student kappa [95% CI]"
        " | Accuracy | Macro F1 | Majority acc. | TF-IDF kappa [95% CI]"
        " | Student minus TF-IDF kappa [95% paired CI] | Teacher ceiling kappa"
        " | Label cost | Wall time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r, meta in runs:
        ag, t, s = r["agreement"], r["teacher"], r["student"]
        ceiling = r.get("ceiling") or {}
        ceiling_cell = _ci(ceiling) if "kappa" in ceiling else "n/a"
        minutes = meta.get("wall_seconds", r["wall_seconds"]) / 60
        lines.append(
            f"| {r['dataset']['name']} | {t['model']} | {_student_name(s['model'])}"
            f" | {t['pool_labels']} | {ag['n']} | {_ci(ag)} | {_f(ag['accuracy'])}"
            f" | {_f(ag.get('macro_f1'))} | {_f(r['baselines']['majority']['accuracy'])}"
            f" | {_ci(r['baselines']['tfidf_logreg'])}"
            f" | {_diff(r['baselines']['tfidf_logreg'].get('student_minus_tfidf_kappa'))}"
            f" | {ceiling_cell}"
            f" | ${t['label_cost_usd']:.2f} | {minutes:.0f} min |"
        )
    lines.append("")
    for name, r, meta in runs:
        lines += _details(name, r, meta)
    return "\n".join(lines)


def _details(name: str, r: Report, meta: Report) -> list[str]:
    ag, t, s = r["agreement"], r["teacher"], r["student"]
    train = s.get("train") or {}
    out = [
        f"#### {r['dataset']['name']}, teacher `{t['model']}`, student"
        f" `{_student_name(s['model'])}`",
        "",
        f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`,"
        " `train-metrics.jsonl`)",
        f"- Command: `{meta.get('command', '')}` at `{str(meta.get('git_sha', ''))[:7]}`"
        f"{' (dirty tree)' if meta.get('git_dirty') else ''}; seed {meta.get('seed')},"
        f" {meta.get('n_resamples')} bootstrap resamples; {meta.get('cpus')} CPUs;"
        f" wall {_f(meta.get('wall_seconds'), 0)} s; spent ${r['spent_usd']:.4f}",
        f"- Data: {r['dataset']['repo']}@{str(r['dataset']['revision'])[:7]}"
        f" ({r['dataset']['license']}), {r['dataset']['sampled']} inputs sampled"
        f" (seed {r['dataset']['sample_seed']}, sha256 {str(r['dataset']['sample_sha256'])[:12]})",
        f"- Training: {train.get('examples')} examples and"
        f" {train.get('validation_examples')} held back to pick the epoch, out of"
        f" {t['pool_labels']} teacher labels; {train.get('epochs')} epochs,"
        f" {train.get('steps')} steps, best epoch {train.get('best_epoch')},"
        f" {_f(train.get('seconds'), 0)} s",
    ]
    if t.get("failures"):
        out.append(f"- Teacher failures: {t['failures']} inputs got no label")
    dup = r["dataset"].get("near_duplicate_heldout")
    if dup is not None:
        out.append(
            f"- Near-duplicates: {dup} of {ag['n']} held-out inputs share at least"
            f" {100 * r['dataset']['near_duplicate_jaccard']:.0f}% of their words with a labelled"
            " pool input (identical texts never straddle the split; these are the ones a hash"
            " cannot catch)"
        )
    if t["kind"] == "gold":
        out.append(
            "- The teacher here is the dataset's gold labels (no API key in this run), so this"
            " row measures the harness and how well the student learns from"
            f" {t['pool_labels']:,} clean labels; it is not a parity-with-Claude number."
        )
    out += _teacher_lines(t)
    gold = r.get("gold") or {}
    if "teacher_vs_gold" in gold:
        out.append(
            f"- Against gold: teacher {_ci(gold['teacher_vs_gold'])},"
            f" student {_ci(gold['student_vs_gold'])}"
        )
    out += _error_lines(gold.get("errors"), ag["n"])
    out += _same_items_lines((r.get("ceiling") or {}).get("on_same_items"), r.get("ceiling"))
    # Classes with a handful of held-out items have F1 of 0 or 1 by chance; rank the rest.
    worst = sorted(
        (c for c in ag.get("per_class", []) if c.get("support", 0) >= MIN_SUPPORT),
        key=lambda c: (c["f1"], c["label"]),
    )[:5]
    if worst:
        out.append(
            f"- Weakest classes with at least {MIN_SUPPORT} held-out items (F1 vs teacher,"
            " support): "
            + ", ".join(f"`{c['label']}` {_f(c['f1'], 2)} ({c['support']})" for c in worst)
        )
    confusions = ag.get("top_confusions", [])[:5]
    if confusions:
        out.append(
            "- Top confusions (teacher -> student): "
            + ", ".join(f"`{c['teacher']}` -> `{c['student']}` {c['count']}" for c in confusions)
        )
    for key, title in (("by_confidence", "confidence"), ("by_length", "length (words)")):
        slices = ag.get(key, [])
        if slices:
            cells = ", ".join(
                f"{_f(sl['lo'], 2 if key == 'by_confidence' else 0)}-"
                f"{_f(sl['hi'], 2 if key == 'by_confidence' else 0)}: {_f(sl['agreement'], 2)}"
                f" (n={sl['n']})"
                for sl in slices
            )
            out.append(f"- Agreement by {title}: {cells}")
    out.append("")
    return out


def _share(part: int, whole: int) -> str:
    return f"{part} ({100 * part / whole:.0f}%)" if whole else f"{part}"


def _teacher_lines(t: Mapping[str, Any]) -> list[str]:
    """How a local teacher was run: the pinned weights and the time its labels took."""
    if t.get("provider") != "ollama":
        return []
    seconds = t.get("compute_seconds") or {}
    digest = str(t.get("digest") or "")
    line = f"- Teacher: `{t['model']}` through Ollama"
    if digest:
        line += f", weights sha256 `{digest[:12]}` (pinned in the eval config)"
    if seconds:
        para = seconds.get("paraphrase_total")
        line += (
            f"; {seconds['labels']} labels took {_f(seconds['total'] / 3600, 2)} h of server"
            f" time (median {_f(seconds['median_per_label'], 2)} s per label)"
            + (f", the ceiling's second pass {_f(para / 60, 0)} min more" if para else "")
            + ", $0"
        )
    return [line]


def _error_lines(errors: Mapping[str, Any] | None, n: int) -> list[str]:
    """What the student did with the teacher's mistakes (`classify.teacher_errors`)."""
    if not errors:
        return []
    diff = errors["student_minus_teacher_vs_gold_kappa"]
    acc = errors["student_minus_teacher_vs_gold_accuracy"]
    wrong, right = errors["on_teacher_errors"], errors["on_teacher_correct"]
    out = [
        f"- Student minus teacher, agreement with gold on the same {n} inputs (paired"
        f" bootstrap): kappa {_diff(diff)}, accuracy {_diff(acc)}",
        f"- On the {wrong['n']} held-out inputs the teacher got wrong, the student gave the"
        f" teacher's label on {_share(wrong['copied'], wrong['n'])}, the gold label on"
        f" {_share(wrong['corrected'], wrong['n'])} and another on"
        f" {_share(wrong['other'], wrong['n'])}; on the {right['n']} it got right, the student"
        f" is wrong on {_share(right['broken'], right['n'])}",
    ]
    cells = errors["confusions"]["teacher"][:6]
    if cells:
        out.append(
            "- The teacher's top confusions (gold -> teacher label: how often the teacher /"
            " the student makes it): "
            + ", ".join(
                f"`{c['gold']}` -> `{c['label']}` {c['teacher']} / {c['student']}" for c in cells
            )
        )
    return out


def _same_items_lines(
    same: Mapping[str, Any] | None, ceiling: Mapping[str, Any] | None
) -> list[str]:
    """Student-vs-teacher on the ceiling's own items, split by whether the teacher repeated
    its label under the paraphrased prompt."""
    if not same or not ceiling:
        return []
    stable, unstable = same["stable"], same["unstable"]

    def rate(part: Mapping[str, Any]) -> str:
        return f"{_pct(part['rate'])} of {part['n']}" if part["n"] else "none"

    return [
        f"- On the ceiling's {ceiling['n']} inputs: teacher vs itself {_ci(ceiling)}, student"
        f" vs teacher {_ci(same['student_vs_teacher'])}; the student agrees with the teacher on"
        f" {rate(stable)} inputs where the teacher repeated its label and on {rate(unstable)}"
        " where it did not",
    ]


def render_fallible(runs: list[tuple[str, Report, Report]]) -> str:
    """One row per classify run whose teacher is a model rather than gold: how the teacher,
    its own repeat, and the student trained on it each agree with the dataset's gold labels."""
    rows = [(name, r) for name, r, _ in runs if r["teacher"]["kind"] != "gold"]
    if not rows:
        return "No run with a model as the teacher yet: `make eval-classify TEACHER=ollama`.\n"
    # The gold run's held-out ids too: its kappa is only comparable on the same inputs, which
    # a teacher that failed some of them (they leave its held-out set) would break.
    clean = {
        r["dataset"]["name"]: (r["agreement"], _heldout_ids(r))
        for _, r, _ in runs
        if r["teacher"]["kind"] == "gold" and r["student"]["model"]
    }
    lines = [
        "| Dataset | Teacher | Student | Teacher vs gold | Teacher vs itself (ceiling)"
        " | Student vs teacher | Student vs teacher, ceiling inputs | Student vs gold"
        " | Student minus teacher vs gold [95% paired CI] | Same student on gold labels |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in rows:
        t, s, gold = r["teacher"], r["student"], r.get("gold") or {}
        ceiling = r.get("ceiling") or {}
        same = ceiling.get("on_same_items") or {}
        errors = gold.get("errors") or {}
        reference, ids = clean.get(r["dataset"]["name"], (None, None))
        if reference is not None and ids != _heldout_ids(r):
            reference = None
        lines.append(
            f"| {r['dataset']['name']} | {t['model']} | {_student_name(s['model'])}"
            f" | {_ci(gold.get('teacher_vs_gold'))}"
            f" | {_ci(ceiling) if 'kappa' in ceiling else 'n/a'}"
            f" | {_ci(r['agreement'])} | {_ci(same.get('student_vs_teacher'))}"
            f" | {_ci(gold.get('student_vs_gold'))}"
            f" | {_diff(errors.get('student_minus_teacher_vs_gold_kappa'))}"
            f" | {_ci(reference) if reference else 'n/a (other held-out inputs)'} |"
        )
    lines += [
        "",
        "All kappas are on the same held-out inputs (the ceiling and its column on the first"
        f" {rows[0][1].get('ceiling', {}).get('n', 'n/a')} of them, which the teacher labelled"
        ' under two prompts). "Same student on gold labels" is the gold-teacher run of the'
        " table above: the same sample, split and student trained on clean labels.",
        "",
    ]
    return "\n".join(lines)


def _heldout_ids(report: Report) -> list[str] | None:
    ids = (report.get("predictions") or {}).get("input_id")
    return sorted(ids) if isinstance(ids, list) else None


def _f1_ci(stats: Mapping[str, Any] | None, key: str = "entity_f1_ci") -> str:
    """`0.812 [0.780, 0.841]` from a block holding `entity` (or `entity_f1`) and its CI."""
    if not stats:
        return "n/a"
    value = (stats.get("entity") or {}).get("f1", stats.get("entity_f1"))
    ci = stats.get(key) or {}
    return f"{_f(value)} [{_f(ci.get('low'))}, {_f(ci.get('high'))}]"


def _kappa_ci(stats: Mapping[str, Any] | None) -> str:
    if not stats or "token_kappa" not in stats:
        return "n/a"
    ci = stats.get("token_kappa_ci") or {}
    return f"{_f(stats['token_kappa'])} [{_f(ci.get('low'))}, {_f(ci.get('high'))}]"


def _pct(x: object) -> str:
    return f"{100 * x:.1f}%" if isinstance(x, int | float) else "n/a"


def render_extract(runs: list[tuple[str, Report, Report]]) -> str:
    if not runs:
        return "No extraction runs yet: run `make eval-extract`.\n"
    lines = [
        "| Dataset | Teacher | Student | Teacher labels | Held-out | Entity F1 [95% CI]"
        " | Precision | Recall | Token kappa [95% CI] | All-O token acc."
        " | Gazetteer F1 [95% CI] | Student minus gazetteer F1 [95% paired CI]"
        " | Teacher ceiling F1 | Unaligned (pool) | Label cost | Wall time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r, meta in runs:
        ag, t, s = r["agreement"], r["teacher"], r["student"]
        gaz = r["baselines"]["gazetteer"]
        ceiling = r.get("ceiling") or {}
        ceiling_cell = _f1_ci(ceiling) if "entity_f1" in ceiling else "n/a"
        pool = (r.get("alignment") or {}).get("pool") or {}
        minutes = meta.get("wall_seconds", r["wall_seconds"]) / 60
        lines.append(
            f"| {r['dataset']['name']} | {t['model']} | {_student_name(s['model'])}"
            f" | {t['pool_labels']} | {ag['n']} | {_f1_ci(ag)}"
            f" | {_f(ag['entity']['precision'])} | {_f(ag['entity']['recall'])}"
            f" | {_kappa_ci(ag)} | {_f(ag['all_o_accuracy'])} | {_f1_ci(gaz)}"
            f" | {_diff(gaz.get('student_minus_gazetteer_f1'))} | {ceiling_cell}"
            f" | {_pct(pool.get('unaligned_rate'))} | ${t['label_cost_usd']:.2f}"
            f" | {minutes:.0f} min |"
        )
    lines.append("")
    for name, r, meta in runs:
        lines += _extract_details(name, r, meta)
    return "\n".join(lines)


def _alignment_line(label: str, a: Mapping[str, Any]) -> str:
    matched = ", ".join(f"{k} {v}" for k, v in (a.get("matched") or {}).items() if v)
    dropped = ", ".join(f"{k} {v}" for k, v in (a.get("dropped") or {}).items() if v)
    return (
        f"- Alignment, {label}: {a.get('mentions')} teacher mentions; matched {matched or 'none'};"
        f" dropped {dropped or 'none'} (unaligned {_pct(a.get('unaligned_rate'))});"
        f" {a.get('duplicates', 0)} duplicates; {a.get('snapped', 0)} snapped to a token boundary"
    )


def _extract_details(name: str, r: Report, meta: Report) -> list[str]:
    ag, t, s = r["agreement"], r["teacher"], r["student"]
    train = s.get("train") or {}
    out = [
        f"#### {r['dataset']['name']}, teacher `{t['model']}`, student"
        f" `{_student_name(s['model'])}`",
        "",
        f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`,"
        " `train-metrics.jsonl`)",
        f"- Command: `{meta.get('command', '')}` at `{str(meta.get('git_sha', ''))[:7]}`"
        f"{' (dirty tree)' if meta.get('git_dirty') else ''}; seed {meta.get('seed')},"
        f" {meta.get('n_resamples')} bootstrap resamples (by input); {meta.get('cpus')} CPUs;"
        f" wall {_f(meta.get('wall_seconds'), 0)} s; spent ${r['spent_usd']:.4f}",
        f"- Data: {r['dataset']['repo']}@{str(r['dataset']['revision'])[:7]}"
        f" `{r['dataset'].get('file')}` ({r['dataset']['license']}), {r['dataset']['sampled']}"
        f" sentences sampled (seed {r['dataset']['sample_seed']}, sha256"
        f" {str(r['dataset']['sample_sha256'])[:12]})",
        f"- Training: {train.get('examples')} examples and"
        f" {train.get('validation_examples')} held back to pick the epoch (entity F1), out of"
        f" {t['pool_labels']} teacher labels; {train.get('epochs')} epochs,"
        f" {train.get('steps')} steps, best epoch {train.get('best_epoch')},"
        f" {_f(train.get('seconds'), 0)} s",
    ]
    alignment = r.get("alignment") or {}
    for key in ("pool", "heldout"):
        if alignment.get(key):
            out.append(_alignment_line(key, alignment[key]))
    if t.get("failures"):
        out.append(f"- Teacher failures: {t['failures']} inputs got no label")
    dup = r["dataset"].get("near_duplicate_heldout")
    if dup is not None:
        out.append(
            f"- Near-duplicates: {dup} of {ag['n']} held-out sentences share at least"
            f" {100 * r['dataset']['near_duplicate_jaccard']:.0f}% of their words with a labelled"
            " pool sentence"
        )
    if t["kind"] == "gold":
        out.append(
            "- The teacher here is the dataset's gold annotation (no API key in this run), so"
            " this row measures the harness and how well the tagger learns from"
            f" {t['pool_labels']:,} clean labels; it is not a parity-with-Claude number."
        )
    gold = r.get("gold") or {}
    if "teacher_vs_gold" in gold:
        out.append(
            f"- Against gold (entity F1): teacher {_f1_ci(gold['teacher_vs_gold'])},"
            f" student {_f1_ci(gold['student_vs_gold'])}"
        )
    types = [c for c in ag.get("per_type", []) if c.get("support", 0) >= MIN_SUPPORT]
    if types:
        gaz_types = {c["type"]: c for c in r["baselines"]["gazetteer"].get("per_type", [])}
        out.append(
            f"- Entity F1 by type (student / gazetteer, held-out support, types with at least"
            f" {MIN_SUPPORT}): "
            + ", ".join(
                f"`{c['type']}` {_f(c['f1'], 2)} / {_f(gaz_types.get(c['type'], {}).get('f1'), 2)}"
                f" ({c['support']})"
                for c in sorted(types, key=lambda c: (c["f1"], c["type"]))
            )
        )
    e = ag.get("errors") or {}
    if e:
        out.append(
            "- Teacher entities by outcome: "
            + ", ".join(
                f"{k.replace('_', ' ')} {e[k]}"
                for k in ("correct", "wrong_boundary", "wrong_type", "wrong_both", "missed")
            )
            + f"; spurious student entities {e.get('spurious')}"
        )
    slices = ag.get("by_length", [])
    if slices:
        cells = ", ".join(
            f"{_f(sl['lo'], 0)}-{_f(sl['hi'], 0)}: {_f(sl['agreement'], 2)} (n={sl['n']})"
            for sl in slices
        )
        out.append(f"- Entity F1 by length (words): {cells}")
    stress = r.get("alignment_stress") or {}
    rows = stress.get("corruptions") or {}
    if rows:
        out += [
            "",
            f"Alignment stress on the {stress.get('spans')} held-out gold spans: every mention"
            " corrupted the same way; only the spans a corruption changes are scored, and a"
            " span counts when alignment returns it exactly:",
            "",
            "| Corruption | Spans changed | Verify only | Repair ladder | Repair: misplaced |",
            "|---|---|---|---|---|",
        ]
        for cname, row in rows.items():
            if not row.get("changed"):
                continue
            v, rep = row["verify_only"], row["repair"]
            out.append(
                f"| {cname} | {row['changed']} | {v['recovered']} ({_pct(v['recovered_rate'])})"
                f" | {rep['recovered']} ({_pct(rep['recovered_rate'])}) | {rep['misplaced']} |"
            )
    out.append("")
    return out


_KIND_NAMES = {
    "wrong_boundary": "boundary (same type, other extent)",
    "wrong_type": "type (same extent, other type)",
    "wrong_both": "both",
    "missed": "missed (no overlapping span)",
    "spurious": "spurious (overlaps no teacher span)",
}


def _auc(stats: Mapping[str, Any] | None) -> str:
    if not stats or "auroc" not in stats:
        return "n/a"
    a = stats["auroc"]
    return f"{_f(a['estimate'])} [{_f(a['low'])}, {_f(a['high'])}]"


def _split_met(entry: Mapping[str, Any]) -> str:
    if entry.get("met") is None:
        return "n/a"
    ci = entry.get("met_interval") or [None, None]
    return (
        f"{entry['met_splits']}/{entry['found_splits']} ({_pct(entry['met'])};"
        f" {_pct(ci[0])}-{_pct(ci[1])})"
    )


def render_extract_cascade(runs: list[tuple[str, Report, Report]]) -> str:
    """The tagger in the cascade (M12): thresholds on entity F1, the split study, the error
    kinds either side of the threshold and how well the confidence detects each kind."""
    if not runs:
        return "No extraction cascade study yet: `make eval-extract-cascade`.\n"
    out: list[str] = []
    for name, r, meta in runs:
        src, ss, served = r["source"], r["split_study"], r.get("served_rule", "lower")
        out += [
            f"#### {src['dataset']}, teacher `{src['teacher_model']}`, student"
            f" `{_student_name(str(src['student_model']))}`: the tagger in the cascade",
            "",
            f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`), from"
            f" `{src['report']}` (sha256 `{str(src['sha256'])[:12]}`, student run"
            f" `{src['student_run']}`)",
            f"- Command: `{meta.get('command', '?')}` at {_commit(meta)}; {r['n']} held-out"
            f" sentences, entity F1 {_f(r['f1'])} against the teacher; confidence: the"
            f" {src['confidence']} ({r['distinct_confidences']} distinct values); at least"
            f" {ss['min_accepted']} answered sentences; {ss['n_resamples']} sentence"
            f" resamples; wall {_f(meta.get('wall_seconds'), 0)} s; no model, spent $0.00",
            "",
            "Threshold each rule serves on all held-out sentences (the cascade F1 counts"
            " escalated sentences as answered by the teacher):",
            "",
            "| Target F1 | Rule | Threshold | Sentences answered | F1 on answered"
            " | Lower bound | F1 on escalated | Cascade F1 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for target, rules in r["thresholds"].items():
            for rule, c in rules.items():
                label = _RULE_NAMES.get(rule, rule)
                if c is None:
                    out.append(f"| {target} | {label} | none | 0 (0.0%) | n/a | n/a | n/a | n/a |")
                    continue
                out.append(
                    f"| {target} | {label} | {_f(c['threshold'], 4)} | {c['accepted']}"
                    f" ({_pct(c['coverage'])}) | {_f(c['value'])} | {_f(c['low'])}"
                    f" | {_f(c['below'].get('f1'))} | {_f(c['system_f1'])} |"
                )
        out += [
            "",
            f"Unseen halves: {ss['splits']} random splits into {ss['fit_size']} sentences to"
            f' fit the threshold on and {ss["test_size"]} to test it on. "Kept" counts the'
            " splits whose unseen half's answered sentences reach the target among those"
            ' where a threshold was found (exact 95% interval); "missed" is the share of'
            " all splits that served a cascade missing the target.",
            "",
            "| Target F1 | Rule | Threshold found | Unseen half: kept the target"
            " | Missed (all splits) | Unseen half: coverage | Unseen half: F1 on answered |",
            "|---|---|---|---|---|---|---|",
        ]
        for target, rules in ss["targets"].items():
            for rule, e in rules.items():
                out.append(
                    f"| {target} | {_RULE_NAMES.get(rule, rule)} | {_pct(e['found'])}"
                    f" | {_split_met(e)} | {_pct(e['missed_share'])}"
                    f" | {_qpct(e.get('coverage'))} | {_q(e.get('f1'))} |"
                )
        cuts = [(t, rules.get(served)) for t, rules in r["thresholds"].items()]
        cuts = [(t, c) for t, c in cuts if c is not None]
        if cuts:
            head = " | ".join(f"Above F1 {t} cut | Below" for t, _ in cuts)
            out += [
                "",
                f"Error kinds of M4's matcher either side of the {_RULE_NAMES.get(served, served)}"
                " threshold (counts of teacher spans, spurious ones of student spans; each"
                " row adds up to the committed report's breakdown):",
                "",
                f"| | All | {head} |",
                "|---|---|" + "---|---|" * len(cuts),
            ]

            def row(label: str, total: object, cells: list[str]) -> str:
                return f"| {label} | {total} | " + " | ".join(cells) + " |"

            out.append(
                row(
                    "Sentences",
                    r["n"],
                    [f"{c[s]['sentences']}" for _, c in cuts for s in ("above", "below")],
                )
            )
            total_entities = sum(v for k, v in r["errors"].items() if k != "spurious")
            out.append(
                row(
                    "Teacher entities",
                    total_entities,
                    [
                        f"{c[s].get('teacher_entities', 0)}"
                        f" ({_rate(c[s].get('teacher_entities', 0), total_entities)})"
                        for _, c in cuts
                        for s in ("above", "below")
                    ],
                )
            )
            out.append(
                row(
                    "Entity F1",
                    _f(r["f1"]),
                    [_f(c[s].get("f1")) for _, c in cuts for s in ("above", "below")],
                )
            )
            for kind, label in _KIND_NAMES.items():
                total = r["errors"][kind]
                cells = []
                for _, c in cuts:
                    for s in ("above", "below"):
                        n = (c[s].get("errors") or {}).get(kind, 0)
                        cells.append(f"{n} ({_rate(n, total)})")
                out.append(row(label, total, cells))
        det = r["detection"]
        out += [
            "",
            'Does low confidence find each kind? AUROC of the confidence for "this sentence'
            ' holds an error of this kind", against the clean sentences (no false positive'
            f" or negative; {det['clean_sentences']} of them) and against every other"
            " sentence, with 95% sentence-resampled intervals. A sentence can hold several kinds"
            f" ({det['boundary_and_type']} hold both a boundary and a type error).",
            "",
            "| Error kind | Sentences | AUROC against clean"
            " | AUROC against every other sentence |",
            "|---|---|---|---|",
        ]
        clean, rest = det["clean"]["by_kind"], det["rest"]["by_kind"]
        for kind, label in {"any_error": "any error", **_KIND_NAMES}.items():
            if kind not in clean:
                continue
            out.append(
                f"| {label} | {clean[kind]['sentences']} | {_auc(clean[kind])}"
                f" | {_auc(rest.get(kind))} |"
            )
        out.append(
            f"| boundary minus type (paired) | | {_diff(det['clean'].get('boundary_minus_type'))}"
            f" | {_diff(det['rest'].get('boundary_minus_type'))} |"
        )
        out += [
            "",
            "F1 against coverage on all held-out sentences:",
            "",
            "| Coverage | Threshold | F1 on answered | Cascade F1 |",
            "|---|---|---|---|",
        ]
        for p in r["curve"]:
            out.append(
                f"| {_pct(p['coverage'])} | {_f(p['threshold'], 4)} | {_f(p['value'])}"
                f" | {_f(p['system_f1'])} |"
            )
        if src.get("teacher_model") == "gold":
            out += [
                "",
                "The teacher here is the dataset's gold spans (no API key in this run): this"
                " measures the tagger's confidence against clean labels, not against Claude.",
            ]
        out.append("")
    return "\n".join(out)


def _em(stats: Mapping[str, Any] | None, key: str = "execution_match") -> str:
    """Execution match with its interval, `0.712 [0.660, 0.763]`."""
    if not stats or key not in stats:
        return "n/a"
    ci = stats[key]
    return f"{_f(ci.get('estimate'))} [{_f(ci.get('low'))}, {_f(ci.get('high'))}]"


def render_generate(runs: list[tuple[str, Report, Report]]) -> str:
    if not runs:
        return "No generation runs yet: run `make eval-generate`.\n"
    lines = [
        "| Dataset | Teacher | Student | Teacher labels | Held-out | Execution match [95% CI]"
        " | Exec. match, rows doubled too | Exact SQL | Exec. match, non-empty reference"
        " | Zero-shot base [95% CI]"
        " | Student minus zero-shot [95% paired CI] | Teacher ceiling | Merged export"
        " | Label cost | Wall time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r, meta in runs:
        ag, t, s = r["agreement"], r["teacher"], r["student"]
        zero = (r.get("baselines") or {}).get("zero_shot") or {}
        ceiling = r.get("ceiling") or {}
        parity = (r.get("export") or {}).get("parity") or {}
        merged = (
            f"{parity['same_completions']}/{parity['probes']} identical,"
            f" max diff {_sci(parity['max_abs_logit_diff'])}"
            if parity
            else "n/a"
        )
        minutes = meta.get("wall_seconds", r["wall_seconds"]) / 60
        lines.append(
            f"| {r['dataset']['name']} | {t['model']} | {_student_name(s['model'])}"
            f" | {t['pool_labels']} | {ag['n']} | {_em(ag)}"
            f" | {_em(ag, 'execution_match_doubled')} | {_f(ag['exact_match'])}"
            f" | {_f(ag.get('execution_match_nonempty'))} | {_em(zero)}"
            f" | {_diff(zero.get('student_minus_zero_shot'))} | {_em(ceiling)}"
            f" | {merged} | ${t['label_cost_usd']:.2f} | {minutes:.0f} min |"
        )
    lines.append("")
    for name, r, meta in runs:
        lines += _generate_details(name, r, meta)
    return "\n".join(lines)


def _errors(stats: Mapping[str, Any]) -> str:
    errors = stats.get("candidate_errors") or {}
    return ", ".join(f"{k} {v}" for k, v in sorted(errors.items())) or "none"


def _rescored(meta: Report) -> list[str]:
    """The line saying a run's stored queries were scored again by a later scorer, if so."""
    again = meta.get("rescored")
    if not again:
        return []
    return [
        f"- Rescored: `{again.get('command', '')}` at `{str(again.get('git_sha', ''))[:7]}`"
        f"{' (dirty tree)' if again.get('git_dirty') else ''} (the stored queries scored"
        " again by the current scorer; training and decoding not repeated)"
    ]


def _generate_details(name: str, r: Report, meta: Report) -> list[str]:
    ag, t, s = r["agreement"], r["teacher"], r["student"]
    train = s.get("train") or {}
    chat = s.get("chat") or {}
    settings = s.get("settings") or {}
    lora = settings.get("lora") or {}
    d = r["dataset"]
    out = [
        f"#### {d['name']}, teacher `{t['model']}`, student `{_student_name(s['model'])}`",
        "",
        f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`,"
        " `train-metrics.jsonl`)",
        f"- Command: `{meta.get('command', '')}` at `{str(meta.get('git_sha', ''))[:7]}`"
        f"{' (dirty tree)' if meta.get('git_dirty') else ''}; seed {meta.get('seed')},"
        f" {meta.get('n_resamples')} bootstrap resamples (by question); {meta.get('cpus')} CPUs"
        f" ({meta.get('device', 'cpu')}); wall {_f(meta.get('wall_seconds'), 0)} s; spent"
        f" ${r['spent_usd']:.4f}",
        *_rescored(meta),
        f"- Data: {d['repo']}@{str(d['revision'])[:7]} `{d.get('file')}` ({d['license']}),"
        f" {d['sampled']} questions sampled (seed {d['sample_seed']}, tables of at most"
        f" {d.get('max_columns')} columns, sha256 {str(d['sample_sha256'])[:12]}); held-out"
        f" inputs (column names and question) sharing at least 80% of their words with a pool"
        f" input: {d.get('near_duplicate_heldout')}",
        f"- Training: LoRA r={lora.get('r')}, alpha={lora.get('alpha')} on"
        f" {', '.join(lora.get('target_modules') or [])}; {train.get('examples')} examples and"
        f" {train.get('validation_examples')} held back to pick the epoch (greedy exact match),"
        f" {train.get('epochs')} epochs, {train.get('steps')} steps, best epoch"
        f" {train.get('best_epoch')}, {_f(train.get('seconds'), 0)} s; chat template:"
        f" {chat.get('examples')} answers aligned, {sum((chat.get('dropped') or {}).values())}"
        f" dropped, longest sequence {chat.get('longest')} tokens",
        f"- Decoding: greedy, {_f(s.get('generate_ms_per_input'), 0)} ms per held-out question"
        f" (batches of 16, length-sorted)",
        f"- Execution match on the {ag.get('nonempty_reference')} questions whose reference"
        f" finds rows: {_f(ag.get('execution_match_nonempty'))}; references that find nothing"
        f" (no rows, an aggregate over no matching rows: NULL, or 0 from COUNT):"
        f" {ag.get('reference_empty')} of {ag['n']} (any query that also finds nothing matches"
        " them, and doubling the rows cannot tell them apart)",
        f"- Coincidental matches: {ag.get('coincidental_matches')} queries match on the table"
        " but not once every row is doubled (a missing aggregate over one row, a count that"
        " equals a cell)",
        f"- Execution vs exact match: {ag.get('executed_match_not_exact')} queries match by"
        f" result but differ as SQL, {ag.get('exact_but_not_executed_match')} the other way;"
        f" with string literals compared case-insensitively (as the sandbox compares text)"
        f" exact SQL is {_f(ag.get('exact_match_casefold_literals'))} and"
        f" {ag.get('executed_match_not_exact_casefold_literals')} differ; student execution"
        f" errors: {_errors(ag)}",
    ]
    if t["kind"] == "gold":
        out.append(
            "- The teacher here is WikiSQL's own query for each question (no API key in this"
            " run), so this row measures the harness and how well the student learns from"
            f" {t['pool_labels']:,} clean labels; it is not a parity-with-Claude number."
        )
    zero = (r.get("baselines") or {}).get("zero_shot") or {}
    if "execution_match" in zero:
        out.append(
            f"- Zero-shot base model (no adapter, same prompt): execution match {_em(zero)},"
            f" exact SQL {_f(zero.get('exact_match'))}, {zero.get('fenced_completions')}"
            f" answers wrapped in a code fence; errors: {_errors(zero)}"
        )
    gold = r.get("gold") or {}
    if "teacher_vs_gold" in gold:
        out.append(
            f"- Against WikiSQL's queries: teacher {_em(gold['teacher_vs_gold'])}, student"
            f" {_em(gold['student_vs_gold'])}"
        )
    sl = r.get("slices") or {}
    for key, label in (
        ("aggregate", "aggregate"),
        ("conditions", "conditions"),
        ("length", "words"),
    ):
        rows = sl.get(key) or []
        if rows:
            cells = ", ".join(
                f"{row[label]}: {_f(row['execution_match'], 2)} (n={row['n']})" for row in rows
            )
            title = {"aggregate": "by aggregate", "conditions": "by WHERE conditions"}.get(
                key, "by input length (words)"
            )
            out.append(f"- Execution match {title}: {cells}")
    judge = r.get("judge") or {}
    if "vs_execution_match" in judge:
        j = judge["vs_execution_match"]
        out.append(
            f"- Judge `{judge.get('model')}` on {j.get('n')} sampled pairs ({j.get('failed')}"
            f" failed): equivalent {_pct(j.get('equivalent_rate'))}; both say match"
            f" {j.get('both_match')}, judge only {j.get('judge_only')}, execution only"
            f" {j.get('exec_only')}, neither {j.get('neither')}; kappa {_f(j.get('kappa'))};"
            f" spent ${judge.get('spent_usd', 0.0):.4f}"
        )
    elif judge.get("skipped"):
        out.append(f"- Judge sample: skipped ({judge['skipped']})")
    export = r.get("export") or {}
    parity = export.get("parity") or {}
    if parity:
        out.append(
            f"- Merged export: {parity.get('same_completions')}/{parity.get('probes')} held-out"
            f" completions identical to the adapter's, max |logit diff|"
            f" {_sci(parity.get('max_abs_logit_diff'))} on {parity.get('logit_probes')} prompts;"
            f" {_f(export.get('size_mb'), 0)} MB, {export.get('adapter_parameters'):,} adapter"
            " parameters folded in"
        )
    out.append("")
    return out


BLOCK = re.compile(
    r"(<!-- BEGIN generated:(?P<name>[a-z-]+) -->\n)(?P<body>.*?)"
    r"(<!-- END generated:(?P=name) -->)",
    re.DOTALL,
)


def render(text: str, blocks: Mapping[str, str]) -> str:
    def sub(m: re.Match[str]) -> str:
        name = m.group("name")
        if name not in blocks:
            return m.group(0)
        return m.group(1) + blocks[name].rstrip("\n") + "\n" + m.group(4)

    return BLOCK.sub(sub, text)


def render_seed_spread(runs: list[tuple[str, Report, Report]]) -> str:
    """How far held-out kappa moves when only the student seed changes (weight init, data
    order, validation slice), next to the width of one run's bootstrap interval. The bootstrap
    covers which items landed in held-out; it says nothing about training randomness."""
    groups: dict[str, list[Report]] = {}
    for name, report, _ in runs:
        m = _SEED_DIR.match(name)
        groups.setdefault(m.group("base") if m else name, []).append(report)
    lines = [
        "| Dataset | Teacher | Student | Seeds | Kappa per seed | Mean | Range (max - min)"
        " | Mean 95% CI width |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows = 0
    for _, reports in sorted(groups.items()):
        if len(reports) < 2:
            continue
        reports = sorted(reports, key=lambda r: int(r["student"]["settings"]["seed"]))
        kappas = [float(r["agreement"]["kappa"]) for r in reports]
        widths = [
            float(r["agreement"]["kappa_ci"]["high"]) - float(r["agreement"]["kappa_ci"]["low"])
            for r in reports
        ]
        first = reports[0]
        per_seed = ", ".join(
            f"{r['student']['settings']['seed']}: {_f(k)}"
            for r, k in zip(reports, kappas, strict=True)
        )
        lines.append(
            f"| {first['dataset']['name']} | {first['teacher']['model']}"
            f" | {_student_name(first['student']['model'])} | {len(reports)} | {per_seed}"
            f" | {_f(sum(kappas) / len(kappas))} | {_f(max(kappas) - min(kappas))}"
            f" | {_f(sum(widths) / len(widths))} |"
        )
        rows += 1
    if not rows:
        return "No seed repeats yet: run `python -m eval.classify <dataset> --seed N`.\n"
    return "\n".join(lines) + "\n"


SELECT_DIR = "select-synthetic"
"""Where `python -m eval.select_study` writes, under the results directory."""
SELECT_AT = (50, 100, 200, 400)
"""Label counts whose mean test accuracy the selection table shows."""


def load_select(results_dir: Path) -> tuple[Report, Report] | None:
    path = results_dir / SELECT_DIR / "report.json"
    if not path.exists():
        return None
    meta_path = path.parent / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    return json.loads(path.read_text(encoding="utf-8")), meta


def _reach(stats: Mapping[str, Any]) -> str:
    """`98.5 [60-150]`: mean labels to a target over the seeds that reached it, and its range."""
    if stats.get("mean") is None:
        return "never"
    return f"{stats['mean']:g} [{stats['min']}-{stats['max']}]"


def render_select(run: tuple[Report, Report] | None) -> str:
    """The synthetic selection study: per setting, labels to target and accuracy by labels."""
    if run is None:
        return "No selection study yet: run `make eval-select`.\n"
    report, meta = run
    config = report["config"]
    max_labels = config["max_labels"]
    seeds = config["seeds"]
    lines = [
        f"- Raw output: `eval/results/{SELECT_DIR}/` (`report.json`, `meta.json`)",
        f"- Command: `{meta.get('command', '')}` at `{str(meta.get('git_sha', ''))[:7]}`"
        f"{' (dirty tree)' if meta.get('git_dirty') else ''}; {seeds} seeds, data seed"
        f" {report['data']['data_seed']}; {meta.get('cpus')} CPUs;"
        f" wall {_f(meta.get('wall_seconds', report.get('wall_seconds')), 0)} s;"
        f" spent ${report.get('spent_usd', 0):.2f}",
        "",
    ]
    for name, setting in report["settings"].items():
        targets = setting["targets"]
        lines += [
            f"**{name}**: {setting['pool_size']:,} pool points"
            f" ({setting['distinct_pool_points']:,} distinct), {setting['test_size']:,} test"
            f" points; a student trained on the whole pool scores"
            f" {_f(setting['full_pool_accuracy'])} test accuracy (95% of it:"
            f" {_f(targets['relative'])}).",
            "",
            f"| Strategy | Labels to {targets['accuracy']:.2f} accuracy, mean [min-max]"
            f" | Seeds not reaching {targets['accuracy']:.2f} by {max_labels}"
            f" | Labels to {_f(targets['relative'])} (95% of full pool), mean [min-max]"
            f" | Seeds not reaching it | "
            + " | ".join(f"Accuracy at {n}" for n in SELECT_AT)
            + " |",
            "|---|---|---|---|---|" + "---|" * len(SELECT_AT),
        ]
        for strategy, result in setting["strategies"].items():
            if result.get("status") != "ok":
                lines.append(
                    f"| {strategy} | not implemented yet: run `make eval-select` once it is"
                    " written | | | | " + " | ".join("" for _ in SELECT_AT) + " |"
                )
                continue
            accuracy = dict(zip(result["label_counts"], result["accuracy"]["mean"], strict=True))
            to = result["labels_to_target"]
            lines.append(
                f"| {strategy} | {_reach(to['accuracy'])} | {to['accuracy']['never']} of {seeds}"
                f" | {_reach(to['relative'])} | {to['relative']['never']} of {seeds} | "
                + " | ".join(_f(accuracy.get(n)) for n in SELECT_AT)
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def load_frontiers(results_dir: Path) -> list[tuple[str, Report, Report]]:
    """Every `frontier-*` run except smoke runs."""
    return load_runs(results_dir, prefix="frontier-")


def _target(value: object, budgets: list[int]) -> str:
    return str(value) if isinstance(value, int) else f"> {max(budgets)}"


def _heldout(dataset: Report, settings: Report) -> str:
    """The held-out inputs a frontier scores: the hash split's share of the sample, capped."""
    cap = int(settings["heldout_labels"])
    inputs = dataset.get("heldout_inputs")
    if not isinstance(inputs, int) or inputs <= cap:
        return str(inputs if isinstance(inputs, int) else cap)
    return f"{cap} (of {inputs})"


def _resumed(name: str, meta: Report) -> str:
    """Training plus selection time summed over the rows, flagged when it exceeds the wall:
    the run then finished an earlier invocation's loops (`anvil loop` resumes by id)."""
    path = RESULTS_DIR / name / "frontier.jsonl"
    compute = 0.0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                compute += float(row.get("train_seconds", 0)) + float(row.get("select_seconds", 0))
    note = f" (training + selection {_f(compute / 60, 0)} min"
    if compute > float(meta.get("wall_seconds", 0)):
        note += ", so this invocation resumed an earlier one"
    return note + ")"


def render_frontier(runs: list[tuple[str, Report, Report]]) -> str:
    """Held-out kappa per label budget and strategy (mean and range over seeds), and the labels
    each target kappa needs, from `eval/frontier.py` runs."""
    if not runs:
        return "No frontier runs yet: run `make eval-frontier`.\n"
    out: list[str] = []
    for name, r, meta in runs:
        settings = r["settings"]
        budgets = [int(b) for b in settings["budgets"]]
        targets = [f"{float(t):g}" for t in settings["targets"]]
        d, t = r["dataset"], r["teacher"]
        out += [
            f"#### {d['name']}, teacher `{t['model']}`, student"
            f" `{_student_name(str(r['student']['model']))}`",
            "",
            f"- Raw output: `eval/results/{name}/` (`frontier.jsonl`, `report.json`, `meta.json`)",
            f"- Command: `{meta.get('command', '')}` at `{str(meta.get('git_sha', ''))[:7]}`"
            f"{' (dirty tree)' if meta.get('git_dirty') else ''}; seeds"
            f" {', '.join(str(x) for x in settings['seeds'])}; {meta.get('cpus')} CPUs;"
            f" wall {_f(meta.get('wall_seconds', 0) / 60, 0)} min"
            f"{_resumed(name, meta)}; spent ${r['spent_usd']:.4f}",
            f"- Data: {d['repo']}@{str(d['revision'])[:7]}, {d['sampled']} inputs sampled"
            f" (sha256 {str(d['sample_sha256'])[:12]}): a pool of {d['pool']} to select from,"
            f" {_heldout(d, settings)} held-out inputs scored every round",
            "",
            "| Strategy | "
            + " | ".join(f"{b} labels" for b in budgets)
            + " | "
            + " | ".join(f"Labels to kappa {x} (mean / every seed's CI)" for x in targets)
            + f" | Label cost at {budgets[-1]} |",
            "|---|" + "---|" * (len(budgets) + len(targets) + 1),
        ]
        for strategy, status in r["status"].items():
            if status != "measured":
                out.append(
                    f"| {strategy} | "
                    + " | ".join("not implemented yet" for _ in budgets)
                    + " | "
                    + " | ".join("n/a" for _ in targets)
                    + " | n/a |"
                )
                continue
            s = r["strategies"][strategy]
            by_budget = {int(pt["budget"]): pt for pt in s["points"]}
            cells = []
            for b in budgets:
                pt = by_budget.get(b)
                cells.append(
                    "n/a"
                    if pt is None
                    else f"{_f(pt['kappa_mean'])} [{_f(pt['kappa_min'])}, {_f(pt['kappa_max'])}]"
                )
            reach = [
                f"{_target(s['labels_to_target'][x]['mean_kappa'], budgets)}"
                f" / {_target(s['labels_to_target'][x]['every_seed_ci'], budgets)}"
                for x in targets
            ]
            last = by_budget.get(budgets[-1])
            cost = f"${last['label_cost_usd_mean']:.2f}" if last else "n/a"
            out.append(f"| {strategy} | " + " | ".join([*cells, *reach, cost]) + " |")
        measured = [k for k, v in r["status"].items() if v == "measured"]
        if measured:
            pts = r["strategies"][measured[0]]["points"]
            out += [
                "",
                f"Classes seen in the training labels ({measured[0]}, mean over seeds): "
                + ", ".join(f"{_f(pt['classes_covered_mean'], 1)} at {pt['budget']}" for pt in pts)
                + "; training time per round: "
                + ", ".join(f"{_f(pt['train_seconds_mean'], 0)} s" for pt in pts)
                + ".",
            ]
        if t["kind"] == "gold":
            out += [
                "",
                "The teacher here is the dataset's gold labels (no API key in this run): the"
                " curve measures how the student learns from clean labels, not parity with"
                " Claude.",
            ]
        out.append("")
    return "\n".join(out)


_CONDITION_NAMES = {
    "gold": "gold labels",
    "teacher": "the teacher's labels",
    "uniform": "gold with uniform noise",
    "pair": "gold with the teacher's confusions",
}


def _range(stats: Mapping[str, Any] | None, digits: int = 3) -> str:
    """`mean (min-max)` over seeds."""
    if not stats:
        return "n/a"
    return f"{_f(stats['mean'], digits)} ({_f(stats['min'], digits)}-{_f(stats['max'], digits)})"


def render_noise(runs: list[tuple[str, Report, Report]]) -> str:
    """The noise control: the same student trained on gold, on the teacher's labels and on
    gold corrupted at the teacher's error rate in two ways (`eval.noise`)."""
    if not runs:
        return "No noise-control run yet: `make eval-noise`.\n"
    out: list[str] = []
    for name, r, meta in runs:
        pool, held, summary = r["pool"], r["heldout"], r["summary"]
        out += [
            f"#### {r['dataset']['name']}, teacher `{r['teacher']['model']}`, student"
            f" `{_student_name(r['student']['model'])}`",
            "",
            f"- Raw output: `eval/results/{name}/` (`report.json` with every run's held-out"
            " predictions, `meta.json`)",
            f"- Command: `{meta.get('command', '?')}` at {_commit(meta)}; seeds"
            f" {', '.join(str(x) for x in r['seeds'])}; {meta.get('cpus')} CPUs; wall"
            f" {_duration(float(meta.get('wall_seconds', 0)))}; $0",
            f"- Training labels: {pool['n']} pool inputs; the teacher got"
            f" {_share(pool['teacher_wrong'], pool['n'])} wrong (kappa with gold"
            f" {_ci(pool['teacher_vs_gold'])}), so each corrupted set has exactly that many"
            " wrong labels",
            f"- Scored on {held['n']} held-out inputs, where the teacher's own kappa with gold is"
            f" {_ci(held['teacher_vs_gold'])}",
            "",
            "| Trained on | Student vs gold kappa, mean (range) | Student vs teacher kappa"
            " | Teacher's held-out errors the student repeats |",
            "|---|---|---|---|",
        ]
        for condition in r["conditions"]:
            entry = summary.get(condition)
            if not entry:
                continue
            out.append(
                f"| {_CONDITION_NAMES.get(condition, condition)}"
                f" | {_range(entry['kappa_vs_gold'])} | {_range(entry['kappa_vs_teacher'])}"
                f" | {_range({k: 100 * v for k, v in entry['copied_rate'].items()}, 0)}% |"
            )
        paired = summary.get("paired_vs_gold") or {}
        for key, cells in paired.items():
            other = key.removeprefix("teacher_minus_")
            out.append("")
            out.append(
                "- Kappa with gold, the student trained on the teacher's labels minus the one"
                f" trained on {_CONDITION_NAMES.get(other, other)} (paired bootstrap, per"
                " seed): " + ", ".join(f"seed {c['seed']} {_diff(c)}" for c in cells)
            )
        out.append("")
    return "\n".join(out)


def _auroc_cell(stats: Mapping[str, Any] | None) -> str:
    if not stats:
        return "n/a"
    return f"{_f(stats['auroc'])} [{_f(stats['low'])}, {_f(stats['high'])}]"


def render_flag(runs: list[tuple[str, Report, Report]]) -> str:
    """Out-of-fold flagging of the training labels (`eval.flag`): how well the student's
    probability of each given label finds the wrong ones, and what dropping the flagged share
    and retraining does to the student's kappa with gold."""
    if not runs:
        return "No flagging run yet: `make eval-flag`.\n"
    out: list[str] = []
    for name, r, meta in runs:
        pool, held = r["pool"], r["heldout"]
        share = _pct(r["flag_fraction"])
        target = r["target_precision"]
        out += [
            f"#### {r['dataset']['name']}, teacher `{r['teacher']['model']}`, student"
            f" `{_student_name(r['student']['model'])}`",
            "",
            f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`, `oof.npz` with"
            " every out-of-fold probability)",
            f"- Command: `{meta.get('command', '?')}` at {_commit(meta)}; {pool['n']} pool"
            f" labels, {pool['folds']} folds (seed {pool['fold_seed']}); {r['students_trained']}"
            f" students trained in this invocation; {meta.get('cpus')} CPUs; wall"
            f" {_duration(float(meta.get('wall_seconds', 0)))}; $0",
            "",
            "| Training labels | Wrong | Out-of-fold AUROC for wrong labels"
            f" | Lowest {share} flagged: precision | Recall (of all wrong)"
            f" | Recall at precision >= {target} (share flagged) | Flagged wrong labels where"
            " the student's top class is gold |",
            "|---|---|---|---|---|---|---|",
        ]
        for condition, entry in r["conditions"].items():
            f = entry["flags"]
            at, best = f["at_fraction"], f["at_precision"]
            out.append(
                f"| {_CONDITION_NAMES.get(condition, condition)}"
                f" | {_share(f['wrong'], f['n'])} | {_auroc_cell(f['auroc'])}"
                f" | {_f(at['precision'])} | {_f(at['recall'])} (at most"
                f" {_f(f['best_possible_recall'])})"
                f" | {_f(best['recall'])} ({_pct(best['flagged_share'])})"
                f" | {_pct(f['caught_argmax_is_gold'])} |"
            )
        out += [
            "",
            f"Retrained on what was not flagged, scored against gold on {held['n']} held-out"
            " inputs (seed 0; the reference is the committed seed-0 `eval-noise` student trained"
            " on the same labels unfiltered):",
            "",
            "| Training labels | Trained on | Wrong among them | Kappa with gold"
            " | Minus the unfiltered reference (paired) |",
            "|---|---|---|---|---|",
        ]
        variants = {
            "filtered": f"the {_pct(1 - float(r['flag_fraction']))} not flagged",
            "random": "a random share of the same size",
            "unfiltered": "everything (the reference, retrained here)",
        }
        for condition, entry in r["conditions"].items():
            retrain = entry["retrain"]
            ref = retrain.get("reference_vs_gold")
            if ref:
                out.append(
                    f"| {_CONDITION_NAMES.get(condition, condition)} | everything (committed"
                    f" reference) | {entry['train_labels_wrong']} | {_ci(ref)} | |"
                )
            for key, label in variants.items():
                row = retrain.get(key)
                if not row:
                    continue
                out.append(
                    f"| {_CONDITION_NAMES.get(condition, condition)} | {label} ({row['n_train']})"
                    f" | {row['train_labels_wrong']} | {_ci(row['vs_gold'])}"
                    f" | {_diff(row.get('minus_unfiltered'))} |"
                )
            if retrain.get("filtered_minus_random"):
                out += [
                    "",
                    f"- {_CONDITION_NAMES.get(condition, condition).capitalize()}: kappa with"
                    " gold, filtered minus random drop (paired):"
                    f" {_diff(retrain['filtered_minus_random'])}",
                ]
            repro = retrain.get("reproduces_reference")
            if repro:
                out.append(
                    "- The unfiltered student retrained here predicts the same label as the"
                    f" committed reference on {repro['same_predictions']} of {repro['n']}"
                    " held-out inputs"
                )
            if retrain.get("filtered_minus_random") or repro:
                out.append("")
        out.append("")
    return "\n".join(out)


def _q(stats: Mapping[str, Any] | None, digits: int = 3) -> str:
    """`median [p5, p95]` of a split-study quantile summary."""
    if not stats:
        return "n/a"
    return f"{_f(stats['median'], digits)} [{_f(stats['p5'], digits)}, {_f(stats['p95'], digits)}]"


def _qpct(stats: Mapping[str, Any] | None) -> str:
    """`_q` for a share, in percent."""
    if not stats:
        return "n/a"
    return f"{_pct(stats['median'])} [{_pct(stats['p5'])}, {_pct(stats['p95'])}]"


def _commit(meta: Mapping[str, Any]) -> str:
    sha = str(meta.get("git_sha") or "")[:7]
    return f"`{sha}`" + (" (uncommitted changes)" if meta.get("git_dirty") else "") if sha else "?"


_RULE_NAMES = {
    "point": "point estimate",
    "lower": "95% percentile lower bound",
    "bca": "95% BCa lower bound",
    "exact": "95% exact (Clopper-Pearson) count bound",
    "wilson": "95% Wilson count bound",
}


_METHOD_NAMES = {
    "point": "point estimate, calibrated",
    "lower": "95% percentile lower bound, calibrated",
    "bca": "95% BCa lower bound, calibrated",
    "exact": "95% exact count bound, calibrated",
    "wilson": "95% Wilson count bound, calibrated",
    "point-raw": "point estimate, uncalibrated",
}


def render_calibration(runs: list[tuple[str, Report, Report]]) -> str:
    """Temperature scaling, the served threshold, and the split study of whether a threshold
    chosen on one half of the held-out split keeps its kappa on the other half."""
    if not runs:
        return "No calibration run yet: `make eval-calibration`.\n"
    out = [
        "| Dataset | Teacher | Student | Held-out | Kappa (all inputs) | Temperature"
        " | ECE before -> after (all) | ECE before -> after (unseen half, median)"
        " | Coverage at kappa 0.9 | Coverage at kappa 0.95 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r, _meta in runs:
        src, full, ss = r["source"], r["full"], r["split_study"]
        f9, f95 = full.get("0.9", {}), full.get("0.95", {})
        out.append(
            f"| {src['dataset']} | {src['teacher_model']} | {_student_name(src['student_model'])}"
            f" | {r['n']} | {_f(r['kappa'])} | {_f(f9.get('temperature'), 2)}"
            f" | {_f(f9.get('ece_before'))} -> {_f(f9.get('ece_after'))}"
            f" | {_f(ss['ece_before']['median'])} -> {_f(ss['ece_after']['median'])}"
            f" | {_pct(f9.get('coverage'))} | {_pct(f95.get('coverage'))} |"
        )
    out.append("")
    for name, r, meta in runs:
        src, ss = r["source"], r["split_study"]
        out += [
            f"#### {src['dataset']}, teacher `{src['teacher_model']}`: does the threshold hold"
            " on inputs it was not chosen on?",
            "",
            f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`, `logits.npz`)",
            f"- Command: `{meta.get('command', '?')}` at {_commit(meta)}; student run"
            f" `{src['student_run']}`; {ss['splits']} random splits into {ss['fit_size']}"
            f" inputs to fit on and {ss['test_size']} to test on; at least {ss['min_accepted']}"
            f" answered inputs; {ss['n_resamples']} bootstrap resamples; wall"
            f" {_f(meta.get('wall_seconds'), 0)} s; spent $0.00",
            f"- Fitted temperature over the splits: {_q(ss['temperature'], 2)}",
            "",
            "| Target kappa | Threshold rule | Threshold found"
            " | Unseen half: kappa reaches target | Unseen half: coverage"
            " | Unseen half: kappa on answered | Unseen half: cascade kappa |",
            "|---|---|---|---|---|---|---|",
        ]
        for target, methods in ss["targets"].items():
            for method, m in methods.items():
                out.append(
                    f"| {target} | {_METHOD_NAMES.get(method, method)} | {_pct(m['found'])}"
                    f" | {_pct(m.get('met'))} | {_qpct(m.get('coverage'))} | {_q(m.get('kappa'))}"
                    f" | {_q(m.get('system_kappa'))} |"
                )
        if r.get("by_rule"):
            out += [
                "",
                "Threshold each rule serves on all held-out inputs:",
                "",
                "| Target kappa | Rule | Threshold | Coverage | Kappa on answered | Lower bound |",
                "|---|---|---|---|---|---|",
            ]
            for rule, rule_name in _RULE_NAMES.items():
                for target, row in (r["by_rule"].get(rule) or {}).items():
                    row = row or {}
                    threshold = row.get("threshold")
                    out.append(
                        f"| {target} | {rule_name}"
                        f" | {'none' if threshold is None else _f(threshold)}"
                        f" | {_pct(row.get('coverage'))} | {_f(row.get('kappa'))}"
                        f" | {_f(row.get('kappa_low'))} |"
                    )
        out += [
            "",
            "Coverage-vs-parity curve on all held-out inputs (calibrated confidence; the cascade"
            " kappa counts escalated inputs as answered by the teacher):",
            "",
            "| Coverage | Threshold | Kappa on answered | Agreement on answered | Cascade kappa |",
            "|---|---|---|---|---|",
        ]
        for p in r["curve"]:
            out.append(
                f"| {_pct(p['coverage'])} | {_f(p['threshold'])} | {_f(p['kappa'])}"
                f" | {_f(p['agreement'])} | {_f(p['system_kappa'])} |"
            )
        out += _ceiling_lines(r.get("ceiling"))
        if src.get("teacher_kind") == "gold":
            out += [
                "",
                "The teacher here is the dataset's gold labels (no API key in this run): this"
                " measures the student's confidence against clean labels, not against Claude.",
            ]
        out.append("")
    return "\n".join(out)


def _rate(count: int, total: int) -> str:
    return f"{100 * count / total:.1f}%" if total else "n/a"


def _met(cell: Mapping[str, Any]) -> str:
    return f"{cell['met']}/{cell['found']} ({_rate(cell['met'], cell['found'])})"


_COVERAGE_METHODS = {
    "percentile": "Percentile",
    "bca": "BCa",
    "exact": "Exact",
    "wilson": "Wilson",
}
_SEARCH_RULES = {
    "point": "Point",
    "lower": "Percentile",
    "bca": "BCa",
    "exact": "Exact",
    "wilson": "Wilson",
}


def render_coverage(report: Report, meta: Report) -> str:
    """The coverage simulation (`eval.coverage`): does the one-sided 95% lower bound cover the
    population kappa, and does the threshold search keep its target on the population?"""
    cfg, fixed, search = report["config"], report["fixed"], report["search"]
    fixed = sorted(fixed, key=lambda c: (c["labels"], c["n"], c["kappa"]))
    methods = [m for m in _COVERAGE_METHODS if any(m in c["lower_covered"] for c in fixed)]
    names = [_COVERAGE_METHODS[m] for m in methods]
    rules = [r for r in _SEARCH_RULES if all(r in c for c in search)]
    search = sorted(search, key=lambda c: (c["labels"], c["n"], c["target"]))
    pop = cfg["search_population"]
    at = ", ".join(f"{_pct(float(u))} -> {_f(k)}" for u, k in pop["kappa_at_coverage"].items())
    out = [
        "- Raw output: `eval/results/coverage-simulation/` (`report.json`, `meta.json`)",
        f"- Command: `{meta.get('command', '?')}` at {_commit(meta)}; {cfg['fixed_reps']}"
        f" replications per fixed-sample cell, {cfg['search_reps']} per threshold-search cell;"
        f" {cfg['n_resamples']} bootstrap resamples; {meta.get('workers', '?')} workers; wall"
        f" {_f(meta.get('wall_seconds'), 0)} s; spent $0.00",
        f"- Search population: agreement probability 1 - {pop['beta']} * (1 - c) ** {pop['gamma']}"
        f" for confidence c uniform on [0, 1]; population kappa at coverage ({at})",
        "",
        "#### Fixed sample: does the one-sided 95% lower bound cover the population kappa?",
        "",
        "| Labels | Items | Kappa | "
        + " | ".join(f"{n} bound covers" for n in names)
        + " | Mean bound: "
        + " / ".join(names)
        + " | BCa level for the 5% bound (mean) | Two-sided 95%: percentile / BCa"
        " | Bias of the estimate (± s.e.) | SD of the estimate / bootstrap SD |",
        "|---|---|---|" + "---|" * len(names) + "---|---|---|---|---|",
    ]
    for c in fixed:
        reps, lo, iv = c["reps"], c["lower_covered"], c["interval_covered"]
        means = " / ".join(_f(c["mean_lower"].get(m)) for m in methods)
        out.append(
            f"| {c['labels']} | {c['n']} | {c['kappa']}"
            + "".join(f" | {_rate(lo[m], reps)}" if m in lo else " | n/a" for m in methods)
            + f" | {means} | {_pct(c['mean_bca_level'])}"
            f" | {_rate(iv['percentile'], reps)} / {_rate(iv['bca'], reps)}"
            f" | {c['bias']:+.4f} ± {c['bias_se']:.4f}"
            f" | {_f(c['sd_estimate'], 4)} / {_f(c['mean_bootstrap_sd'], 4)} |"
        )
    out += ["", *_pooled_coverage(fixed, methods, names), ""]
    out += [
        "#### Threshold search: how often does the chosen threshold truly keep the target?",
        "",
        "| Labels | Items | Target | Oracle coverage | "
        + " | ".join(f"{_SEARCH_RULES[r]}: met" for r in rules)
        + " | "
        + " | ".join(f"{_SEARCH_RULES[r]}: true coverage" for r in rules)
        + " |",
        "|---|---|---|---|" + "---|" * (2 * len(rules)),
    ]
    for c in search:
        out.append(
            f"| {c['labels']} | {c['n']} | {c['target']} | {_pct(c['oracle_coverage'])}"
            + "".join(f" | {_met(c[r])}" for r in rules)
            + "".join(f" | {_pct(c[r].get('mean_true_coverage'))}" for r in rules)
            + " |"
        )
    pooled_rules = []
    for rule in rules:
        found = sum(c[rule]["found"] for c in search)
        met = sum(c[rule]["met"] for c in search)
        given = [
            (c[rule]["found"], c[rule]["mean_coverage_given_up"])
            for c in search
            if c[rule].get("mean_coverage_given_up") is not None
        ]
        weight = sum(w for w, _ in given)
        up = _pct(sum(w * g for w, g in given) / weight) if weight else "n/a"
        pooled_rules.append(
            f"{_RULE_NAMES[rule]} met {met}/{found} ({_rate(met, found)}), coverage given up"
            f" versus the oracle {up}"
        )
    out += [
        "",
        f"Pooled over all {len(search)} cells: " + "; ".join(pooled_rules) + ".",
        "",
    ]
    for labels in sorted({c["labels"] for c in search}):
        cells = [c for c in search if c["labels"] == labels]
        parts = []
        for rule in rules:
            found = sum(c[rule]["found"] for c in cells)
            met = sum(c[rule]["met"] for c in cells)
            reps = sum(c["reps"] for c in cells)
            served = sum(c[rule]["found"] * (c[rule]["mean_true_coverage"] or 0) for c in cells)
            parts.append(
                f"{_SEARCH_RULES[rule]} found a threshold in {found}/{reps}, met"
                f" {_rate(met, found)}, true coverage {_pct(_weighted(cells, rule))}"
                f" ({_pct(served / reps)} counting a search with no threshold as 0)"
            )
        out.append(f"- {labels} labels: " + "; ".join(parts) + ".")
    out.append("")
    out += _pointwise_lines(report.get("pointwise"), cfg)
    return "\n".join(out)


def _pointwise_lines(rows: list[Mapping[str, Any]] | None, cfg: Mapping[str, Any]) -> list[str]:
    """The exact pointwise coverage of p_o by the count bounds (no sampling)."""
    if not rows:
        return []
    grid = cfg.get("pointwise_rates") or {}
    out = [
        "#### Pointwise: how often does each count bound cover the agreement rate itself?",
        "",
        f"Exact binomial sums, no sampling: for m answered inputs and each of"
        f" {grid.get('points', '?')} agreement rates p from {grid.get('low', '?')} to"
        f" {grid.get('high', '?')}, the probability that the 95% bound on p_o is at or below p.",
        "",
        "| Answered inputs | Bound | Worst coverage | At agreement rate | Mean coverage"
        " | Share of rates under 95% |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        out.append(
            f"| {r['n']} | {_COVERAGE_METHODS.get(r['method'], r['method'])}"
            f" | {100 * r['min']:.2f}% | {_f(r['rate_at_min'])} | {100 * r['mean']:.2f}%"
            f" | {_pct(r['share_below_nominal'])} |"
        )
    out.append("")
    return out


def _weighted(cells: list[Mapping[str, Any]], rule: str) -> float | None:
    """Mean true coverage of the found thresholds over `cells`, weighted by how many were
    found in each."""
    rows = [(c[rule]["found"], c[rule]["mean_true_coverage"]) for c in cells]
    weight = sum(w for w, v in rows if v is not None)
    return sum(w * v for w, v in rows if v is not None) / weight if weight else None


def _pooled_coverage(
    fixed: list[Mapping[str, Any]], methods: list[str], names: list[str]
) -> list[str]:
    """Pooled one-sided coverage of each method over all cells and by label count, and each
    method's worst cell."""
    total = sum(c["reps"] for c in fixed)

    def pooled(cells: list[Mapping[str, Any]], m: str) -> str:
        return _rate(sum(c["lower_covered"][m] for c in cells), sum(c["reps"] for c in cells))

    lines = [
        f"Pooled over all {len(fixed)} cells ({total} replications), the one-sided bound"
        " covers: "
        + ", ".join(f"{n} {pooled(fixed, m)}" for m, n in zip(methods, names, strict=True))
        + "."
    ]
    for labels in sorted({c["labels"] for c in fixed}):
        cells = [c for c in fixed if c["labels"] == labels]
        lines.append(
            f"- {labels} labels: "
            + ", ".join(f"{n} {pooled(cells, m)}" for m, n in zip(methods, names, strict=True))
        )
    worst = []
    for m, n in zip(methods, names, strict=True):
        cell = min(fixed, key=lambda c: c["lower_covered"][m] / c["reps"])
        worst.append(
            f"{n} {_rate(cell['lower_covered'][m], cell['reps'])} ({cell['labels']} labels,"
            f" {cell['n']} items, kappa {cell['kappa']})"
        )
    lines.append("- Worst cell: " + "; ".join(worst))
    return lines


def _kci(stats: Mapping[str, Any] | None) -> str:
    """`kappa [low, high]` of a `{kappa, low, high}` summary."""
    if not stats:
        return "n/a"
    return f"{_f(stats['kappa'])} [{_f(stats['low'])}, {_f(stats['high'])}]"


def _ceiling_lines(ceiling: Mapping[str, Any] | None) -> list[str]:
    """The teacher's self-agreement on the inputs a threshold accepts (`eval.calibration`)."""
    if not ceiling:
        return []
    out = [
        "",
        f"The ceiling on the inputs the cascade answers: of the {ceiling['asked']} held-out"
        " inputs the teacher labelled under both prompts, those whose calibrated confidence"
        " clears each threshold, with the teacher's agreement with itself and the student's"
        " agreement with the teacher on the same inputs:",
        "",
        "| Threshold | Coverage (all held-out) | Ceiling items | Teacher self-agreement kappa"
        " [95% CI] | Student vs teacher kappa [95% CI] |",
        "|---|---|---|---|---|",
    ]
    for row in ceiling["by_level"]:
        out.append(
            f"| top {_pct(row['level'])} | {_pct(row['coverage'])} | {row['n']}"
            f" | {_kci(row.get('ceiling'))} | {_kci(row.get('student_vs_teacher'))} |"
        )
    for target, row in ceiling["served"].items():
        if row is None:
            out.append(f"| served at kappa {target} | no threshold found | | | |")
        else:
            out.append(
                f"| served at kappa {target} | {_pct(row['coverage'])} | {row['n']}"
                f" | {_kci(row.get('ceiling'))} | {_kci(row.get('student_vs_teacher'))} |"
            )
    return out


def _sci(x: object) -> str:
    return f"{x:.1e}" if isinstance(x, int | float) else "n/a"


def _answered(p: Mapping[str, Any]) -> str:
    if p.get("agreement_answered") is None:
        return "n/a"
    return f"{_f(p['agreement_answered'])} (n={p.get('answered', '?')})"


def render_export(runs: list[tuple[str, Report, Report]]) -> str:
    """fp32 and int8 parity of the exported students, and the int8 layer-group probe."""
    if not runs:
        return "No export run yet: `make eval-export`.\n"
    out = [
        "| Dataset | Student | Variant | Compared with | Probes | Max abs logit diff"
        " | Mean abs logit diff | Argmax agreement | Agreement on answered inputs"
        " | Kappa vs teacher (drop) | Size | Gate |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r, _meta in runs:
        src = r["source"]
        for p in (r["fp32"], r.get("int8")):
            if not p:
                continue
            drop = p.get("kappa_drop")
            kappa = f"{_f(p['kappa'])} ({drop:+.3f})" if isinstance(drop, int | float) else "n/a"
            out.append(
                f"| {src['dataset']} | {_student_name(src['student_model'])} | {p['variant']}"
                f" | {p['reference']} | {p['n']} | {_sci(p['max_abs_diff'])}"
                f" | {_sci(p['mean_abs_diff'])} | {_f(p['agreement'])} | {_answered(p)} | {kappa}"
                f" | {_f(p['size_mb'], 0)} MB | {'passed' if p['passed'] else 'failed'} |"
            )
    out.append("")
    for name, r, meta in runs:
        src = r["source"]
        out += [
            f"#### {src['dataset']}: export",
            "",
            f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`)",
            f"- Command: `{meta.get('command', '?')}` at {_commit(meta)}; student run"
            f" `{src['student_run']}`; export and both parity checks {_f(r['export_seconds'], 0)}"
            f" s; wall {_f(meta.get('wall_seconds'), 0)} s; spent $0.00",
            f"- Gates: fp32 {r['fp32']['gate']}; int8 {(r.get('int8') or {}).get('gate', 'n/a')}"
            f" ({r.get('int8_fp32_nodes', 0)} MLP down-projections kept in fp32)",
        ]
        probe = r.get("quantization_probe")
        if probe:
            groups = ", ".join(f"{g} {n}" for g, n in probe["groups"].items())
            out += [
                f"- int8 layer probe (nodes per group: {groups}); each row quantizes only the"
                " named group unless it says otherwise, plus the head and classifier Gemms,"
                " which the quantizer converts before it reads its exclusion list, and is"
                f" compared with the fp32 graph on the same {r['probes']} held-out inputs (fp32"
                f" kappa vs teacher {_f(probe['fp32_kappa'])}); answered inputs are those the fp32"
                " or the int8 graph scores at or above the bundle's threshold:",
                "",
                "| int8 variant | Quantized nodes | Size | Mean abs logit diff"
                " | Max abs logit diff | Argmax agreement | Agreement on answered inputs"
                " | Kappa drop |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for v in probe["variants"]:
                out.append(
                    f"| {v['variant']} | {v['quantized_nodes']} | {_f(v['size_mb'], 0)} MB"
                    f" | {_f(v['mean_abs_diff'])} | {_f(v['max_abs_diff'])}"
                    f" | {_f(v['agreement'])} | {_answered({'answered': probe['answered'], **v})}"
                    f" | {v['kappa_drop']:+.3f} |"
                )
        out.append("")
    return "\n".join(out)


_ENGINE_NAMES = {
    "pytorch": "PyTorch (in process)",
    "onnx-fp32": "ONNX Runtime fp32 (in process)",
    "onnx-int8": "ONNX Runtime int8 (in process)",
    "http-fp32": "`anvil serve` fp32 (HTTP)",
    "http-int8": "`anvil serve` int8 (HTTP)",
}


def _ms(stats: Mapping[str, Any] | None) -> str:
    if not stats:
        return "n/a"
    return f"{stats['p50_ms']:.1f} / {stats['p95_ms']:.1f}"


def _cpu(per_batch: object) -> str:
    if not isinstance(per_batch, Mapping):
        return _f(per_batch, 1)
    return f"{_f(per_batch.get('batch1'), 1)} / {_f(per_batch.get('batch32'), 1)}"


def render_bench(runs: list[tuple[str, Report, Report]]) -> str:
    """Latency and throughput per engine, the student's cost per 1,000 inputs, the teacher's."""
    if not runs:
        return "No bench run yet: `make bench`.\n"
    out: list[str] = []
    for name, r, meta in runs:
        cost = r["cost"]
        price = cost["vcpu_hour_usd"]
        out += [
            f"#### {r['dataset']}: student `{_student_name(r['student_model'])}`,"
            f" {r['threads']} threads",
            "",
            f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`)",
            f"- Command: `{meta.get('command', '?')}` at {_commit(meta)}; {meta.get('cpu', '?')},"
            f" {meta.get('cpus', '?')} CPUs; inputs: the {r['inputs']} held-out texts (mean"
            f" {_f(r['mean_words'], 1)} words), cycled; wall {_f(meta.get('wall_seconds'), 0)} s",
            "",
            "| Engine | Batch 1: p50 / p95 ms | Batch 1: inputs/s | Batch 32: p50 / p95 ms"
            " | Batch 32: inputs/s | CPU seconds per 1k inputs (batch 1 / 32)"
            f" | $ per 1k at ${price}/vCPU-hour, cheaper batch |",
            "|---|---|---|---|---|---|---|",
        ]
        for engine, e in r["engines"].items():
            c = cost.get(engine, {})
            out.append(
                f"| {_ENGINE_NAMES.get(engine, engine)} | {_ms(e.get('batch1'))}"
                f" | {_f(e['batch1']['inputs_per_s'], 0)} | {_ms(e.get('batch32'))}"
                f" | {_f(e['batch32']['inputs_per_s'], 0)} | {_cpu(c.get('core_seconds_per_1k'))}"
                f" | {_f(c.get('usd_per_1k'), 5)} |"
            )
        concurrent = [
            (engine, e["concurrent4_batch1"])
            for engine, e in r["engines"].items()
            if "concurrent4_batch1" in e
        ]
        if concurrent:
            out += ["", "Four concurrent HTTP clients at batch 1:", ""]
            out += [
                f"- {_ENGINE_NAMES.get(engine, engine)}: {_f(c['inputs_per_s'], 0)} inputs/s,"
                f" p50 / p95 {_ms(c)} ms"
                for engine, c in concurrent
            ]
        t = r["teacher"]
        recorded = t.get("recorded_cost")
        live = t.get("live")
        out += [
            "",
            f"The CPU price is an assumption (`--vcpu-hour-usd`), not a measurement; the CPU"
            f" seconds are measured. Teacher `{t['model']}`: "
            + (
                f"${recorded['usd_per_1k']:.3f} per 1k inputs from {recorded['labels']} recorded"
                " labels"
                if recorded
                else "no recorded Claude labels for this dataset yet, so its cost per 1k is under"
                " To measure"
            )
            + (
                f"; live latency p50 / p95 {_ms(live)} ms over {live['calls']} calls."
                if live
                else "; its latency needs `--teacher-live N` with a key (To measure)."
            ),
        ]
        cal = r.get("calibration") or {}
        if isinstance(cal.get("coverage"), int | float):
            out.append(
                f"At the bundle's threshold ({_f(cal.get('threshold'), 4)}, target kappa"
                f" {cal.get('target_kappa')}, rule `{cal.get('rule')}`) the student answers"
                f" {_pct(cal['coverage'])} of held-out inputs, so the cascade calls the teacher"
                f" for {_pct(1 - cal['coverage'])} of traffic."
            )
        out.append("")
    return "\n".join(out)


LEDGER_EVALS = (
    ("classify-", "eval-classify"),
    ("cascade-extract-", "eval-extract-cascade"),
    ("extract-", "eval-extract"),
    ("generate-", "eval-generate"),
    ("frontier-", "eval-frontier"),
    ("select-", "eval-select"),
    ("calibration-", "eval-calibration"),
    ("export-", "eval-export"),
    ("bench-", "bench"),
    ("noise-", "eval-noise"),
    ("flag-", "eval-flag"),
    ("coverage-", "eval-coverage"),
    ("shift-", "eval-shift"),
    ("spike-onnx", "spike-onnx"),
)
"""Directory-name prefixes and the `make` target that writes them."""


_SCORE_NAMES = {
    "max_softmax": "max softmax (served)",
    "max_logit": "max logit",
    "energy": "energy",
    "knn": "k-NN cosine to training embeddings",
    "mahalanobis": "Mahalanobis to class means",
}


def _pci(stats: Mapping[str, Any] | None) -> str:
    """`41.2% [38.1%, 44.3%]` of an `{estimate, low, high}` rate."""
    if not stats:
        return "n/a"
    return f"{_pct(stats['estimate'])} [{_pct(stats['low'])}, {_pct(stats['high'])}]"


def _eci(stats: Mapping[str, Any] | None) -> str:
    """`0.912 [0.897, 0.925]` of an `{estimate, low, high}` summary."""
    if not stats:
        return "n/a"
    return f"{_f(stats['estimate'])} [{_f(stats['low'])}, {_f(stats['high'])}]"


def _signed(stats: Mapping[str, Any] | None) -> str:
    if not stats:
        return "n/a"
    return f"{stats['estimate']:+.3f} [{stats['low']:+.3f}, {stats['high']:+.3f}]"


def _verdict(held: object) -> str:
    return "**held**" if held is True else "**refuted**"


def render_shift(runs: list[tuple[str, Report, Report]]) -> str:
    """The served bundle off-distribution (`eval.shift`): coverage and kappa on the dataset's
    own test split, how much of other datasets the student answers, and five scores for the
    answer-or-escalate decision."""
    if not runs:
        return "No shift run yet: `make eval-shift`.\n"
    out: list[str] = []
    for name, r, meta in runs:
        src, bundle, ref = r["source"], r["bundle"], r["reference"]
        sets: Mapping[str, Any] = r["sets"]
        ood = [n for n, s in sets.items() if s["role"] == "out_of_distribution"]
        ref_set = sets[ref["set"]]
        secs = float(meta.get("wall_seconds", 0))
        out += [
            f"#### {src['dataset']}, teacher `{src['teacher_model']}`, student"
            f" `{_student_name(src['student_model'])}`: the served bundle off-distribution",
            "",
            f"- Raw output: `eval/results/{name}/` (`report.json`, `meta.json`, `scores.npz`"
            " with every input's logits and distance scores)",
            f"- Command: `{meta.get('command', '?')}` at {_commit(meta)}; student run"
            f" `{src['student_run']}` ({src['train_inputs']} training inputs); bundle"
            f" `{bundle['variant']}` through `anvil/runtime.py` at temperature"
            f" {_f(bundle['temperature'], 3)} and threshold {_f(bundle['threshold'], 4)};"
            f" {meta.get('cpus')} CPUs; wall {_duration(secs)};"
            f" spent ${float(meta.get('spent_usd', 0.0)):.2f}",
            f"- Re-analysis from the committed scores, no model: `python -m eval.shift --scores"
            f" eval/results/{name}/scores.npz`",
            "",
            "| Set | Role | Source (pinned) | Licence | Inputs | Answered by the student [95% CI]"
            " | Mean confidence | Most frequent answers |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for set_name, s in sets.items():
            source = s.get("source") or {}
            where = (
                f"`{source.get('repo')}`@`{str(source.get('revision'))[:7]}`"
                f" {source.get('config') or ''}{'/' if source.get('config') else ''}"
                f"{source.get('split')}"
                + (f", label `{source['keep']}`" if source.get("keep") else "")
            )
            answers = ", ".join(f"`{a['label']}` {a['n']}" for a in s.get("top_answers", []))
            out.append(
                f"| {set_name} | {s['role'].replace('_', ' ')} | {where}"
                f" | {source.get('license', 'n/a')} | {s['n']} | {_pci(s['answered'])}"
                f" | {_f(s['mean_confidence'])} | {answers or '-'} |"
            )
        overlap = (r.get("overlap") or {}).get(ref["set"]) or {}
        out += [
            "",
            f"In distribution: the threshold chosen on {ref['heldout_n']} held-out inputs, read"
            f" on {ref_set['n']} {ref['set']} inputs (kappa and ECE against the gold labels).",
            "",
            "| | Held-out (threshold chosen here) | Test split |",
            "|---|---|---|",
            f"| Coverage | {_pct(ref['heldout_coverage'])} | {_pci(ref['coverage'])} |",
            f"| Kappa on answered inputs [95% CI] | {_f(ref['heldout_kappa_answered'])}"
            f" | {_eci(ref['kappa_answered'])} |",
            f"| Accuracy on answered inputs | | {_pct(ref['accuracy_answered'])} |",
            f"| Kappa on all inputs | | {_f(ref['kappa_all'])} |",
            f"| ECE ({ref['ece_bins']} bins), T = 1 -> bundle temperature | |"
            f" {_f(ref['ece_raw'])} -> {_f(ref['ece_calibrated'])} |",
        ]
        fresh = ref.get("fresh")
        if fresh:
            out += [
                f"| Test inputs that are not near-copies of a training-sample text | |"
                f" {fresh['n']}: coverage {_pct(fresh['coverage'])}, kappa on answered"
                f" {_eci(fresh['kappa_answered'])} |",
                f"| Test inputs that are near-copies | | {fresh['near_duplicates']}: coverage"
                f" {_pct(fresh['near_duplicate_coverage'])} |",
            ]
        out.append("")
        if overlap:
            out += [
                f"Test inputs identical to one of the {overlap['sample_size']} sampled training"
                f" rows: {overlap['identical_to_sample']}; near-duplicates (word Jaccard >= 0.8):"
                f" {overlap['near_duplicates_of_sample']}.",
                "",
            ]
        bins = int(r["n_bins"])
        out += [
            "Served confidence (max softmax at the bundle temperature), share of each set per"
            f" bin; the threshold is {_f(bundle['threshold'], 3)}:",
            "",
            "| Confidence | " + " | ".join(sets) + " |",
            "|---" * (len(sets) + 1) + "|",
        ]
        for b in range(bins):
            cells = [_pct(s["histogram"][b] / s["n"]) for s in sets.values()]
            out.append(f"| {b / bins:.1f}-{(b + 1) / bins:.1f} | " + " | ".join(cells) + " |")
        sep: Mapping[str, Any] = r["separation"]
        out += [
            "",
            f"Separating {ref['set']} from each other set: AUROC (in-distribution inputs"
            " scoring higher) with 95% bootstrap intervals, its paired difference from max"
            " softmax, and the share of the other set at or above the score that keeps 95% of"
            " the test split (FPR at 95% TPR):",
            "",
            "| Score | "
            + " | ".join(f"{o}: AUROC | {o}: minus max softmax | {o}: FPR95" for o in ood)
            + " |",
            "|---" * (1 + 3 * len(ood)) + "|",
        ]
        for score in _SCORE_NAMES:
            cells = []
            for o in ood:
                c = sep[o][score]
                cells += [
                    _eci(c["auroc"]),
                    "-" if score == "max_softmax" else _signed(c["minus_max_softmax"]),
                    _pct(c["fpr_at_95_tpr"]),
                ]
            out.append(f"| {_SCORE_NAMES[score]} | " + " | ".join(cells) + " |")
        gates: Mapping[str, Any] = r["gates"]
        cost = float(r["coverage_cost"])
        out += [
            "",
            f"Each score as a second gate on top of the served threshold, its cut set to give"
            f" up {cost:.0%} of the held-out inputs, then read on the test split and the other"
            " sets:",
            "",
            "| Gate | Held-out coverage | Test coverage (cost) | Test kappa on answered | "
            + " | ".join(f"{o} answered" for o in ood)
            + " |",
            "|---" * (4 + len(ood)) + "|",
            f"| served threshold only | {_pct(ref['heldout_coverage'])}"
            f" | {_pct(ref['coverage']['estimate'])} | {_f(ref['kappa_answered']['estimate'])} | "
            + " | ".join(_pct(sets[o]["answered"]["estimate"]) for o in ood)
            + " |",
        ]
        for score in _SCORE_NAMES:
            g = gates[score]
            out.append(
                f"| + {_SCORE_NAMES[score]} | {_pct(g['heldout_coverage'])}"
                f" | {_pct(g['coverage'])} ({100 * g['coverage_cost']:.1f} points)"
                f" | {_f(g['kappa_answered'])} | "
                + " | ".join(_pct(g["answered"][o]) for o in ood)
                + " |"
            )
        for o in ood:
            out += [
                "",
                f"The {o} inputs the student is surest of (each answered with a"
                f" {src['dataset']} label):",
                "",
            ]
            for ex in sets[o].get("most_confident", [])[:5]:
                text = str(ex["text"]).replace("|", "/")
                if len(text) > 110:
                    text = text[:107] + "..."
                out.append(f'- "{text}" -> `{ex["label"]}` ({_f(ex["confidence"])})')
        h, crit = r["hypotheses"], r["criteria"]
        out += [
            "",
            "Hypotheses (PLAN.md, M9), decided by the numbers above:",
            "",
            f"- H1 {_verdict(h['h1']['held'])}: on the test split coverage stays within"
            f" {100 * crit['h1_coverage_points']:.0f} points of the held-out"
            f" {_pct(ref['heldout_coverage'])} (moved {100 * h['h1']['coverage_change']:.1f}"
            f" points) and kappa on the answered inputs stays at or above {crit['h1_kappa']}"
            f" ({_f(h['h1']['kappa_answered'])}).",
        ]
        for o, v in h["h2"].items():
            out.append(
                f"- H2 {_verdict(v['held'])} for {o}: the student answers at least"
                f" {crit['h2_answered']:.0%} above the served threshold ({_pct(v['answered'])})."
            )
        winners = ", ".join(_SCORE_NAMES[s] for s in h["h3"]["scores"]) or "none"
        out += [
            f"- H3 {_verdict(h['h3']['held'])}: a free score beats max softmax by at least"
            f" {crit['h3_auroc_gain']} AUROC on every other set while its gate costs under"
            f" {100 * cost:.0f} points of test coverage (scores that do: {winners}).",
            "",
        ]
        diffs = (r.get("onnx_vs_torch") or {}).get("max_abs_prob_diff") or {}
        if diffs:
            worst = max(float(v) for v in diffs.values())
            out += [
                "The distance scores use the PyTorch student's embeddings; its probabilities"
                f" differ from the ONNX bundle's by at most {_sci(worst)} on any input scored.",
                "",
            ]
    return "\n".join(out)


def _first(*values: object) -> object:
    return next((v for v in values if v is not None), None)


def _duration(seconds: float) -> str:
    hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s" if hours else f"{minutes}m {secs:02d}s"


def render_ledger(results_dir: Path) -> str:
    """One row per run directory under `results_dir` with its command, commit, wall time and
    teacher spend, read from `meta.json` (the M1 spike names its fields `wall_time_s` and
    `dollars_spent`). A field a run did not record is `n/a`, never a guess."""
    lines = [
        "| Run | Eval | Teacher | Command | Commit | Wall time | Teacher $ |",
        "|---|---|---|---|---|---|---|",
    ]
    wall_total = usd_total = 0.0
    rows = 0
    for meta_path in sorted(results_dir.glob("*/meta.json")):
        name = meta_path.parent.name
        if name.endswith(SMOKE_SUFFIX):
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        report_path = meta_path.parent / "report.json"
        report = (
            json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        )
        teacher = _first(
            meta.get("teacher_model"),
            (report.get("teacher") or {}).get("model"),
            (report.get("source") or {}).get("teacher_model"),
        )
        wall = _first(meta.get("wall_seconds"), meta.get("wall_time_s"))
        usd = _first(meta.get("spent_usd"), meta.get("dollars_spent"), report.get("spent_usd"))
        eval_name = next((e for prefix, e in LEDGER_EVALS if name.startswith(prefix)), "n/a")
        command = meta.get("command")
        lines.append(
            f"| `{name}` | {eval_name} | {teacher if teacher else 'n/a'}"
            f" | {f'`{command}`' if command else 'n/a'} | {_commit(meta)}"
            f" | {_duration(wall) if isinstance(wall, int | float) else 'n/a'}"
            f" | {f'${usd:.2f}' if isinstance(usd, int | float) else 'n/a'} |"
        )
        rows += 1
        wall_total += wall if isinstance(wall, int | float) else 0.0
        usd_total += usd if isinstance(usd, int | float) else 0.0
    if not rows:
        return "No runs yet: nothing is committed under `eval/results/`.\n"
    lines.append(
        f"| **Total ({rows} runs)** | | | | | {_duration(wall_total)} | ${usd_total:.2f} |"
    )
    lines += [
        "",
        "Dollars are the sum of the `usage` of each teacher call, priced in"
        " `anvil/teacher/pricing.py`; a run whose teacher is `gold` makes no calls and costs"
        " $0. Wall time is the invocation's `wall_seconds`; for the frontier runs that is the"
        " figure the frontier section below treats as the run's time, and the training +"
        " selection time it also lists is not added again. A run that resumed an earlier"
        " invocation counts only its own. Totals skip runs that did not record a field.",
        "",
    ]
    return "\n".join(lines)


def load_coverage(results_dir: Path) -> tuple[Report, Report] | None:
    folder = results_dir / "coverage-simulation"
    if not (folder / "report.json").exists():
        return None
    return (
        json.loads((folder / "report.json").read_text(encoding="utf-8")),
        json.loads((folder / "meta.json").read_text(encoding="utf-8")),
    )


def blocks_for(results_dir: Path) -> dict[str, str]:
    coverage = load_coverage(results_dir)
    return {
        "ledger": render_ledger(results_dir),
        "classify": render_classify(load_runs(results_dir)),
        "classify-seeds": render_seed_spread(load_runs(results_dir, seeds=True)),
        "classify-teacher": render_fallible(load_runs(results_dir)),
        "noise": render_noise(load_runs(results_dir, prefix="noise-")),
        "flag": render_flag(load_runs(results_dir, prefix="flag-")),
        "select": render_select(load_select(results_dir)),
        "frontier": render_frontier(load_frontiers(results_dir)),
        "extract": render_extract(load_runs(results_dir, prefix="extract-")),
        "extract-cascade": render_extract_cascade(
            load_runs(results_dir, prefix="cascade-extract-")
        ),
        "calibration": render_calibration(load_runs(results_dir, prefix="calibration-")),
        "coverage": (
            render_coverage(*coverage)
            if coverage
            else "No coverage simulation yet: `make eval-coverage`.\n"
        ),
        "export": render_export(load_runs(results_dir, prefix="export-")),
        "bench": render_bench(load_runs(results_dir, prefix="bench-")),
        "generate": render_generate(load_runs(results_dir, prefix="generate-")),
        "shift": render_shift(load_runs(results_dir, prefix="shift-")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.render", description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if RESULTS.md is stale")
    args = parser.parse_args(argv)
    current = RESULTS_MD.read_text(encoding="utf-8")
    updated = render(current, blocks_for(RESULTS_DIR))
    if args.check:
        if updated != current:
            return 1
        return 0
    RESULTS_MD.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
