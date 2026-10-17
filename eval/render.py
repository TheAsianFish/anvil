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


def _student_name(model: str) -> str:
    return model.rsplit("/", 1)[-1]


SMOKE_SUFFIX = "-smoke"
"""Results directories ending in this hold smoke runs (`eval.classify` with overrides)."""


def load_runs(results_dir: Path, prefix: str = "classify-") -> list[tuple[str, Report, Report]]:
    """Every run under `results_dir`, except smoke runs (`...-smoke/`): those used overridden
    settings (another student, fewer epochs, a smaller sample) and are not results."""
    runs = []
    for path in sorted(results_dir.glob(f"{prefix}*/report.json")):
        if path.parent.name.endswith(SMOKE_SUFFIX):
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
        "| Dataset | Teacher | Student | Train labels | Held-out | Student kappa [95% CI]"
        " | Accuracy | Macro F1 | Majority acc. | TF-IDF kappa [95% CI] | Teacher ceiling kappa"
        " | Label cost | Wall time |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
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
            f" | {_ci(r['baselines']['tfidf_logreg'])} | {ceiling_cell}"
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
        f"- Training: {train.get('epochs')} epochs, {train.get('steps')} steps,"
        f" best epoch {train.get('best_epoch')}, {_f(train.get('seconds'), 0)} s",
    ]
    if t["kind"] == "gold":
        out.append(
            "- The teacher here is the dataset's gold labels (no API key in this run), so this"
            " row measures the harness and how well the student learns from 2,000 clean labels;"
            " it is not a parity-with-Claude number."
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


def blocks_for(results_dir: Path) -> dict[str, str]:
    return {"classify": render_classify(load_runs(results_dir))}


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
