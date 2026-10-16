"""Spike: export ModernBERT-base (sequence classification) to ONNX on CPU and check parity.

Runs every exporter configuration in ``CONFIGS``, compares ONNX Runtime logits with PyTorch
eval-mode logits on a fixed set of sentences, measures batch-1 CPU latency, and tries dynamic
int8 quantization of each successful fp32 export. Writes ``results.json`` and ``meta.json``.

Usage:
    uv run python eval/spikes/onnx_export_spike.py --out eval/results/spike-onnx-export
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch
import transformers
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

LOG = logging.getLogger("onnx_export_spike")

MODEL_ID = "answerdotai/ModernBERT-base"
NUM_LABELS = 4
SEED = 0
NUM_THREADS = 4
WARMUP_RUNS = 5
TIMED_RUNS = 20

FloatArray = npt.NDArray[np.float32]


@dataclass(frozen=True)
class ExportConfig:
    name: str
    dynamo: bool
    attn_implementation: str
    # How reference_compile is handed to transformers: not at all, as a from_pretrained kwarg,
    # or set on the config object before loading.
    reference_compile: Literal["omit", "kwarg_false", "kwarg_true", "config_true"] = "omit"


CONFIGS: tuple[ExportConfig, ...] = (
    ExportConfig("dynamo-sdpa", dynamo=True, attn_implementation="sdpa"),
    ExportConfig("dynamo-eager", dynamo=True, attn_implementation="eager"),
    ExportConfig("torchscript-sdpa", dynamo=False, attn_implementation="sdpa"),
    ExportConfig("torchscript-eager", dynamo=False, attn_implementation="eager"),
    ExportConfig(
        "dynamo-sdpa-refcompile-kwarg-false",
        dynamo=True,
        attn_implementation="sdpa",
        reference_compile="kwarg_false",
    ),
    ExportConfig(
        "dynamo-sdpa-refcompile-config-true",
        dynamo=True,
        attn_implementation="sdpa",
        reference_compile="config_true",
    ),
)

_SUBJECTS = [
    "The package",
    "My refund request",
    "The new firmware update",
    "Customer support",
    "This invoice",
    "The delivery driver",
    "Our quarterly report",
    "The login page",
]
_PREDICATES = [
    "arrived late.",
    "was approved after three weeks of back and forth with the billing team.",
    "broke the bluetooth pairing on every device I own, and rolling back did not help.",
    "is great.",
    "has a typo in the total and charges VAT twice, which I only noticed after paying it.",
    "left the box in the rain at the wrong address, then marked it as delivered to me "
    "personally, and nobody has answered the three emails I sent since Monday morning.",
    "needs a second review before Friday.",
    "keeps timing out.",
]


def build_sentences() -> list[str]:
    """32 deterministic sentences of varied length (2 to ~60 words)."""
    sentences = [f"{s} {p}" for s, p in zip(_SUBJECTS, _PREDICATES, strict=True)]
    rotated = _PREDICATES[3:] + _PREDICATES[:3]
    sentences += [f"{s} {p}" for s, p in zip(_SUBJECTS, rotated, strict=True)]
    sentences += ["Thanks!", "Cancel it.", "Where is my order?", "ok"]
    long_tail = " ".join(_PREDICATES)
    sentences += [f"{s} {long_tail}" for s in _SUBJECTS[:4]]
    sentences += [f"{s} {p} {q}" for s, p, q in zip(_SUBJECTS, _PREDICATES, rotated, strict=True)]
    assert len(sentences) == 32, len(sentences)
    return sentences


class LogitsOnly(torch.nn.Module):
    """Wraps the HF model so the exported graph has two inputs and one output."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits: torch.Tensor = out.logits
        return logits


@dataclass
class Inputs:
    """Numpy inputs: one padded batch of all sentences plus each sentence alone."""

    batch_ids: npt.NDArray[np.int64]
    batch_mask: npt.NDArray[np.int64]
    single: list[tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]]
    latency_index: int


@dataclass
class Latency:
    """Batch-1 wall time in milliseconds over TIMED_RUNS runs after WARMUP_RUNS warmups."""

    median: float
    min: float
    max: float


@dataclass
class Parity:
    ref_abs_max: float  # largest |logit| in the PyTorch reference, for scale
    mean_abs_diff: float
    max_abs_diff_batch: float
    max_abs_diff_single: float
    max_abs_diff: float
    argmax_agreement: float
    n_compared: int


@dataclass
class Variant:
    """One ONNX artefact (fp32 export or its int8 quantization)."""

    ok: bool
    error_type: str | None = None
    error_message: str | None = None
    seconds: float | None = None
    size_mb: float | None = None
    parity: Parity | None = None
    ort_latency_ms: Latency | None = None
    opset: int | None = None
    n_nodes: int | None = None


@dataclass
class ConfigResult:
    config: ExportConfig
    load_ok: bool
    torch_latency_ms: Latency | None = None
    fp32: Variant | None = None
    int8: Variant | None = None
    int8_strip_value_info: Variant | None = None
    int8_per_channel: Variant | None = None
    int8_excl_mlp_wo: Variant | None = None
    notes: list[str] = field(default_factory=list)


def tokenize(tokenizer: Any, sentences: list[str]) -> Inputs:  # Any: HF tokenizer is untyped
    enc = tokenizer(sentences, padding=True, return_tensors="np")
    single = []
    for s in sentences:
        e = tokenizer([s], return_tensors="np")
        single.append((e["input_ids"].astype(np.int64), e["attention_mask"].astype(np.int64)))
    lengths = [ids.shape[1] for ids, _ in single]
    median_len = sorted(lengths)[len(lengths) // 2]
    return Inputs(
        batch_ids=enc["input_ids"].astype(np.int64),
        batch_mask=enc["attention_mask"].astype(np.int64),
        single=single,
        latency_index=lengths.index(median_len),
    )


def load_model(cfg: ExportConfig) -> LogitsOnly:
    torch.manual_seed(SEED)  # makes the random-init classification head identical per config
    kwargs: dict[str, Any] = {}  # Any: from_pretrained kwargs are heterogeneous
    if cfg.reference_compile == "kwarg_false":
        kwargs["reference_compile"] = False
    elif cfg.reference_compile == "kwarg_true":
        kwargs["reference_compile"] = True
    elif cfg.reference_compile == "config_true":
        config = AutoConfig.from_pretrained(MODEL_ID, num_labels=NUM_LABELS)
        config.reference_compile = True
        kwargs["config"] = config
    if "config" not in kwargs:
        kwargs["num_labels"] = NUM_LABELS
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, attn_implementation=cfg.attn_implementation, **kwargs
    )
    model.eval()
    return LogitsOnly(model).eval()


def torch_logits(model: LogitsOnly, inputs: Inputs) -> tuple[FloatArray, list[FloatArray]]:
    with torch.inference_mode():
        batch = model(torch.from_numpy(inputs.batch_ids), torch.from_numpy(inputs.batch_mask))
        single = [
            model(torch.from_numpy(i), torch.from_numpy(m)).numpy().astype(np.float32)
            for i, m in inputs.single
        ]
    return batch.numpy().astype(np.float32), single


def time_ms(fn: Any) -> Latency:  # Any: zero-arg callable returning anything
    for _ in range(WARMUP_RUNS):
        fn()
    times = []
    for _ in range(TIMED_RUNS):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return Latency(
        median=round(statistics.median(times), 2),
        min=round(min(times), 2),
        max=round(max(times), 2),
    )


def torch_latency(model: LogitsOnly, inputs: Inputs) -> Latency:
    ids, mask = inputs.single[inputs.latency_index]
    t_ids, t_mask = torch.from_numpy(ids), torch.from_numpy(mask)

    def run() -> None:
        with torch.inference_mode():
            model(t_ids, t_mask)

    return time_ms(run)


def export(model: LogitsOnly, cfg: ExportConfig, inputs: Inputs, path: Path) -> None:
    # Trace with batch 2 and a non-trivial length so neither axis gets specialised to 1.
    ex_ids = torch.from_numpy(inputs.batch_ids[:2, :24].copy())
    ex_mask = torch.from_numpy(inputs.batch_mask[:2, :24].copy())
    names = {"input_names": ["input_ids", "attention_mask"], "output_names": ["logits"]}
    if cfg.dynamo:
        batch = torch.export.Dim("batch", min=1, max=1024)
        seq = torch.export.Dim("sequence", min=2, max=8192)
        torch.onnx.export(
            model,
            (ex_ids, ex_mask),
            str(path),
            dynamo=True,
            dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
            **names,  # type: ignore[arg-type]
        )
    else:
        axes = {0: "batch", 1: "sequence"}
        torch.onnx.export(
            model,
            (ex_ids, ex_mask),
            str(path),
            dynamo=False,
            dynamic_axes={"input_ids": axes, "attention_mask": axes, "logits": {0: "batch"}},
            opset_version=17,
            **names,  # type: ignore[arg-type]
        )


def dir_size_mb(directory: Path) -> float:
    total = sum(p.stat().st_size for p in directory.iterdir() if p.is_file())
    return round(total / 1e6, 1)


def session(path: Path) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = NUM_THREADS
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def evaluate_onnx(
    path: Path, inputs: Inputs, ref_batch: FloatArray, ref_single: list[FloatArray]
) -> tuple[Parity, Latency]:
    sess = session(path)

    def run(ids: npt.NDArray[np.int64], mask: npt.NDArray[np.int64]) -> FloatArray:
        out = sess.run(["logits"], {"input_ids": ids, "attention_mask": mask})
        return np.asarray(out[0], dtype=np.float32)

    got_batch = run(inputs.batch_ids, inputs.batch_mask)
    got_single = np.concatenate([run(i, m) for i, m in inputs.single])
    ref_single_cat = np.concatenate(ref_single)
    diff_batch = float(np.max(np.abs(got_batch - ref_batch)))
    diff_single = float(np.max(np.abs(got_single - ref_single_cat)))
    agree = np.concatenate(
        [
            got_batch.argmax(-1) == ref_batch.argmax(-1),
            got_single.argmax(-1) == ref_single_cat.argmax(-1),
        ]
    )
    ref_all = np.concatenate([ref_batch, ref_single_cat])
    got_all = np.concatenate([got_batch, got_single])
    parity = Parity(
        ref_abs_max=round(float(np.max(np.abs(ref_all))), 6),
        mean_abs_diff=round(float(np.mean(np.abs(got_all - ref_all))), 6),
        max_abs_diff_batch=round(diff_batch, 6),
        max_abs_diff_single=round(diff_single, 6),
        max_abs_diff=round(max(diff_batch, diff_single), 6),
        argmax_agreement=round(float(agree.mean()), 4),
        n_compared=int(agree.size),
    )
    ids, mask = inputs.single[inputs.latency_index]
    latency = time_ms(lambda: run(ids, mask))
    return parity, latency


def graph_stats(path: Path) -> tuple[int, int]:
    proto = onnx.load(str(path), load_external_data=False)
    opset = next((o.version for o in proto.opset_import if o.domain in ("", "ai.onnx")), -1)
    return opset, len(proto.graph.node)


def failure(exc: BaseException, seconds: float | None = None) -> Variant:
    first_line = (str(exc).strip().splitlines() or [""])[0][:500]
    LOG.error("failed: %s", "".join(traceback.format_exception(exc))[-4000:])
    return Variant(
        ok=False, error_type=type(exc).__name__, error_message=first_line, seconds=seconds
    )


def run_config(cfg: ExportConfig, tokenizer: Any, sentences: list[str], tmp: Path) -> ConfigResult:
    LOG.info("config %s", cfg.name)
    try:
        model = load_model(cfg)
    except Exception as exc:
        res = ConfigResult(config=cfg, load_ok=False)
        res.fp32 = failure(exc)
        return res
    res = ConfigResult(config=cfg, load_ok=True)
    stored = getattr(model.model.config, "reference_compile", "<absent>")
    res.notes.append(f"config.reference_compile after load = {stored!r}")
    inputs = tokenize(tokenizer, sentences)
    ref_batch, ref_single = torch_logits(model, inputs)
    refs = (ref_batch, ref_single)
    pad_vs_single = float(np.max(np.abs(ref_batch - np.concatenate(ref_single))))
    res.notes.append(f"torch padded-batch vs single max abs diff = {pad_vs_single:.6f}")
    res.torch_latency_ms = torch_latency(model, inputs)
    res.notes.append(f"latency input length = {inputs.single[inputs.latency_index][0].shape[1]}")

    out_dir = tmp / cfg.name
    out_dir.mkdir()
    fp32_path = out_dir / "model.onnx"
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            export(model, cfg, inputs, fp32_path)
    except Exception as exc:
        res.fp32 = failure(exc, round(time.perf_counter() - t0, 1))
        return res
    export_s = round(time.perf_counter() - t0, 1)
    del model
    try:
        opset, n_nodes = graph_stats(fp32_path)
        parity, latency = evaluate_onnx(fp32_path, inputs, ref_batch, ref_single)
    except Exception as exc:
        res.fp32 = failure(exc, export_s)
        res.notes.append("export succeeded but onnxruntime load/run failed")
        return res
    res.fp32 = Variant(
        ok=True,
        seconds=export_s,
        size_mb=dir_size_mb(out_dir),
        parity=parity,
        ort_latency_ms=latency,
        opset=opset,
        n_nodes=n_nodes,
    )
    LOG.info("fp32 %s", res.fp32)

    res.int8 = quantize_and_evaluate(fp32_path, tmp / f"{cfg.name}-int8", inputs, refs)
    LOG.info("int8 %s", res.int8)
    if not res.int8.ok:
        # Workaround: drop value_info entries that duplicate initializers. The dynamo exporter
        # writes them; the quantizer's Gemm->MatMul rewrite transposes the classifier weight
        # but leaves its value_info at the old shape, so the next shape inference fails.
        clean_path = strip_initializer_value_info(fp32_path, tmp / f"{cfg.name}-clean")
        res.int8_strip_value_info = quantize_and_evaluate(
            clean_path, tmp / f"{cfg.name}-clean-int8", inputs, refs
        )
        LOG.info("int8 (stripped value_info) %s", res.int8_strip_value_info)
    else:
        clean_path = fp32_path
    res.int8_per_channel = quantize_and_evaluate(
        clean_path, tmp / f"{cfg.name}-int8-pc", inputs, refs, per_channel=True
    )
    LOG.info("int8 per-channel %s", res.int8_per_channel)
    excluded = mlp_down_projection_nodes(clean_path)
    res.notes.append(f"int8_excl_mlp_wo keeps {len(excluded)} MLP Wo MatMuls in fp32")
    res.int8_excl_mlp_wo = quantize_and_evaluate(
        clean_path, tmp / f"{cfg.name}-int8-excl", inputs, refs, nodes_to_exclude=excluded
    )
    LOG.info("int8 excluding MLP Wo %s", res.int8_excl_mlp_wo)
    return res


def quantize_and_evaluate(
    fp32_path: Path,
    q_dir: Path,
    inputs: Inputs,
    refs: tuple[FloatArray, list[FloatArray]],
    per_channel: bool = False,
    nodes_to_exclude: list[str] | None = None,
) -> Variant:
    q_dir.mkdir()
    q_path = q_dir / "model.onnx"
    t0 = time.perf_counter()
    try:
        quantize_dynamic(
            fp32_path,
            q_path,
            weight_type=QuantType.QInt8,
            per_channel=per_channel,
            nodes_to_exclude=nodes_to_exclude,
        )
        q_s = round(time.perf_counter() - t0, 1)
        parity, latency = evaluate_onnx(q_path, inputs, *refs)
        opset, n_nodes = graph_stats(q_path)
    except Exception as exc:
        return failure(exc, round(time.perf_counter() - t0, 1))
    return Variant(
        ok=True,
        seconds=q_s,
        size_mb=dir_size_mb(q_dir),
        parity=parity,
        ort_latency_ms=latency,
        opset=opset,
        n_nodes=n_nodes,
    )


def mlp_down_projection_nodes(path: Path) -> list[str]:
    """Names of the MatMuls whose weight is an MLP down-projection (intermediate x hidden).

    Identified by weight shape so it works for both exporters' naming schemes. ModernBERT-base:
    intermediate_size 1152, hidden_size 768. These consume the GeGLU output.
    """
    proto = onnx.load(str(path), load_external_data=False)
    down = {i.name for i in proto.graph.initializer if list(i.dims) == [1152, 768]}
    return [n.name for n in proto.graph.node if n.op_type == "MatMul" and down & set(n.input)]


def strip_initializer_value_info(src: Path, out_dir: Path) -> Path:
    proto = onnx.load(str(src))
    inits = {i.name for i in proto.graph.initializer}
    keep = [vi for vi in proto.graph.value_info if vi.name not in inits]
    del proto.graph.value_info[:]
    proto.graph.value_info.extend(keep)
    out_dir.mkdir()
    dst = out_dir / "model.onnx"
    onnx.save(proto, str(dst), save_as_external_data=True, location="model.onnx.data")
    return dst


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--configs", nargs="*", default=None, help="subset of config names")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ort.set_default_logger_severity(3)

    start = time.perf_counter()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(NUM_THREADS)
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    sentences = build_sentences()
    configs = [c for c in CONFIGS if args.configs is None or c.name in args.configs]
    results: list[ConfigResult] = []
    with tempfile.TemporaryDirectory(prefix="anvil-onnx-spike-") as tmp:
        for cfg in configs:
            cfg_tmp = Path(tmp) / cfg.name
            cfg_tmp.mkdir()
            results.append(run_config(cfg, tokenizer, sentences, cfg_tmp))
            shutil.rmtree(cfg_tmp)  # each config writes ~1.4 GB of artefacts
            (out / "results.json").write_text(
                json.dumps([asdict(r) for r in results], indent=2) + "\n"
            )

    wall = round(time.perf_counter() - start, 1)
    lengths = [len(tokenizer(s)["input_ids"]) for s in sentences]
    meta = {
        "command": " ".join(["uv run python", *sys.argv]),
        "git_sha": git_sha(),
        "model": MODEL_ID,
        "num_labels": NUM_LABELS,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
        "machine": {
            "cpu_count": os.cpu_count(),
            "processor": platform.processor() or platform.machine(),
            "platform": platform.platform(),
        },
        "threads": {"torch": NUM_THREADS, "ort_intra_op": NUM_THREADS},
        "latency": {"batch_size": 1, "warmup_runs": WARMUP_RUNS, "timed_runs": TIMED_RUNS},
        "n_sentences": len(sentences),
        "token_lengths": {"min": min(lengths), "max": max(lengths)},
        "seed": SEED,
        "wall_time_s": wall,
        "dollars_spent": 0.0,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    def summary(v: Variant | None) -> str:
        if v is None:
            return "-"
        if not v.ok or v.parity is None:
            return f"FAIL {v.error_type}"
        return f"diff={v.parity.max_abs_diff:g} agree={v.parity.argmax_agreement:g}"

    for r in results:
        variants = {
            "fp32": r.fp32,
            "int8": r.int8,
            "int8-stripvi": r.int8_strip_value_info,
            "int8-pc": r.int8_per_channel,
            "int8-excl-mlp-wo": r.int8_excl_mlp_wo,
        }
        line = "  ".join(f"{k}: {summary(v)}" for k, v in variants.items() if v is not None)
        print(f"{r.config.name}  {line}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
