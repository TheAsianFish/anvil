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
    gold = r.get("gold") or {}
    if "teacher_vs_gold" in gold:
        out.append(
            f"- Against gold: teacher {_ci(gold['teacher_vs_gold'])},"
            f" student {_ci(gold['student_vs_gold'])}"
        )
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


_METHOD_NAMES = {
    "point": "point estimate, calibrated",
    "lower": "95% lower bound, calibrated",
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
            f"#### {src['dataset']}: does the threshold hold on inputs it was not chosen on?",
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
        if src.get("teacher_kind") == "gold":
            out += [
                "",
                "The teacher here is the dataset's gold labels (no API key in this run): this"
                " measures the student's confidence against clean labels, not against Claude.",
            ]
        out.append("")
    return "\n".join(out)


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


def blocks_for(results_dir: Path) -> dict[str, str]:
    return {
        "classify": render_classify(load_runs(results_dir)),
        "classify-seeds": render_seed_spread(load_runs(results_dir, seeds=True)),
        "select": render_select(load_select(results_dir)),
        "frontier": render_frontier(load_frontiers(results_dir)),
        "extract": render_extract(load_runs(results_dir, prefix="extract-")),
        "calibration": render_calibration(load_runs(results_dir, prefix="calibration-")),
        "export": render_export(load_runs(results_dir, prefix="export-")),
        "bench": render_bench(load_runs(results_dir, prefix="bench-")),
        "generate": render_generate(load_runs(results_dir, prefix="generate-")),
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
