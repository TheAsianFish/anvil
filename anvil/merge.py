"""Merged export of a LoRA generator: fold the adapter into the base weights, check that the
merged model answers exactly as the adapter did, and write it as a plain transformers model.

A LoRA student is a base model plus `B @ A` per adapted layer; serving it that way needs PEFT
and an extra matrix product per layer per token. Merging computes `W + (alpha / r) B A` once,
so the result is an ordinary checkpoint any transformers-compatible runtime loads (vLLM,
llama.cpp after conversion, plain `from_pretrained`). Merging is exact arithmetic in principle,
but it changes the order of floating-point operations, so it is checked, not assumed: the
merged model's logits on the probe prompts must match the adapter model's to
`MERGE_MAX_ABS_DIFF`, and its greedy completions must be identical on every probe. Nothing is
written unless both hold.

ONNX is not the target here: a decoder's useful ONNX graph carries a KV cache and a decoding
loop (onnxruntime-genai territory), which is a serving project of its own; the merged weights
are what such a pipeline starts from.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from anvil.errors import ExportError
from anvil.log import get_logger
from anvil.students.generator import META_FILE, LoraGenerator

MERGE_MAX_ABS_DIFF = 1e-3
"""fp32 logits of the merged model against the adapter model, on the probe prompts."""
LOGIT_PROBES = 8
"""Probes whose full-prompt logits are compared (each is one forward pass per model)."""
MERGED_META = "anvil_merged.json"
MERGED_FORMAT = 1

LOG = get_logger(__name__)


@dataclass(frozen=True)
class MergeParity:
    probes: int
    """Probe inputs whose greedy completions were compared."""
    logit_probes: int
    max_abs_logit_diff: float
    same_completions: int
    passed: bool
    gate: str


@dataclass(frozen=True)
class MergeReport:
    out: Path
    parity: MergeParity
    size_mb: float
    seconds: float
    adapter_parameters: int
    """Parameters of the adapter that was folded into the base weights."""


@torch.no_grad()
def _max_logit_diff(reference: LoraGenerator, merged: Any, prompts: Sequence[list[int]]) -> float:
    worst = 0.0
    for ids in prompts:
        x = torch.tensor([ids], dtype=torch.long, device=reference.device)
        a = reference.model(input_ids=x).logits.float()
        b = merged(input_ids=x).logits.float()
        worst = max(worst, float((a - b).abs().max()))
    return worst


def _size_mb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6


def export_generator(
    student_dir: Path,
    out: Path,
    probe_texts: Sequence[str],
    *,
    spec: dict[str, object] | None = None,
    batch_size: int = 16,
) -> MergeReport:
    """Merge the student in `student_dir` and write it to `out` if the merge keeps parity.

    `probe_texts` are inputs to compare on (held-out inputs: the merge must keep the answers
    the eval measured). Raises `ExportError` with the measured parity when the gate fails;
    `out` is then left as it was."""
    if not probe_texts:
        raise ExportError("no probe inputs to check the merged model on; label held-out inputs")
    started = time.perf_counter()
    reference = LoraGenerator.load(student_dir)
    donor = LoraGenerator.load(student_dir)  # merging consumes the adapter, so a second copy
    adapter = reference.adapter_parameters
    merged = donor.merged()
    merged.eval()
    prompts = reference.prompts(probe_texts[:LOGIT_PROBES])
    diff = _max_logit_diff(reference, merged, prompts)
    want = reference.generate(probe_texts, batch_size)
    got = reference.generate(probe_texts, batch_size, model=merged)
    same = sum(a == b for a, b in zip(want, got, strict=True))
    passed = diff <= MERGE_MAX_ABS_DIFF and same == len(probe_texts)
    parity = MergeParity(
        probes=len(probe_texts),
        logit_probes=len(prompts),
        max_abs_logit_diff=diff,
        same_completions=same,
        passed=passed,
        gate=f"max abs logit diff <= {MERGE_MAX_ABS_DIFF} and every completion identical",
    )
    LOG.info("merge parity", extra=asdict(parity))
    if not passed:
        raise ExportError(
            f"the merged model does not answer as the adapter does: max logit diff {diff:.2e},"
            f" {same}/{len(probe_texts)} identical completions (gate: {parity.gate});"
            " nothing was written"
        )
    tmp = out.with_name(f".{out.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        merged.save_pretrained(tmp)
        reference.tokenizer.save_pretrained(tmp)
        source = json.loads((student_dir / META_FILE).read_text(encoding="utf-8"))
        meta = {
            "format": MERGED_FORMAT,
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "student": student_dir.name,
            "outputs": reference.outputs,
            "system": reference.system,
            "max_length": reference.max_length,
            "max_new_tokens": reference.max_new_tokens,
            "stop_ids": reference.stop_ids,
            "parity": asdict(parity),
            "source": {k: v for k, v in source.items() if k not in ("outputs", "system")},
            "spec": spec,
        }
        (tmp / MERGED_META).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return MergeReport(out, parity, _size_mb(out), time.perf_counter() - started, adapter)


def load_merged(path: Path) -> LoraGenerator:
    """A merged export as a generator that can answer (`generate`, `predict`), with no PEFT
    involved. It has no adapter, so it cannot be trained or saved as a student."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    meta_path = path / MERGED_META
    if not meta_path.exists():
        raise ExportError(f"{path} is not a merged Anvil export (no {MERGED_META})")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != MERGED_FORMAT:
        raise ExportError(f"{path} has export format {meta.get('format')}; re-export it")
    model: Any = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(path)
    return LoraGenerator(
        model,
        tokenizer,
        [str(x) for x in meta["outputs"]],
        str(meta["system"]),
        int(meta["max_length"]),
        int(meta["max_new_tokens"]),
        extra_meta={"merged_from": meta.get("student")},
        extra_stop_ids=[int(x) for x in meta.get("stop_ids", [])],
    )
