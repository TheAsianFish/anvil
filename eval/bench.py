"""`make bench`: latency, throughput and cost of the exported student against the teacher.

The student is the Banking77 `eval.classify` student as `make eval-export` bundled it. Inputs
are its held-out texts (real Banking77 messages, about 12 words), cycled. Every timing is wall
clock on this machine after warm-up calls, reported as p50 / p95 per call and inputs per second:

- `pytorch`: the trained model in PyTorch (`EncoderClassifier.predict_proba`), what serving
  without an export would run;
- `onnx-fp32`, `onnx-int8`: the bundle in ONNX Runtime in process (`OnnxStudent`), tokenizer
  included;
- `http-fp32`, `http-int8`: `anvil serve` in a subprocess (uvicorn, teacher off) called over
  HTTP on localhost, one client in sequence, and four concurrent clients at batch 1.

Every engine runs with the same intra-op thread count (`--threads`, default 4). Cost per 1,000
inputs for the student is its measured CPU time (user + system, all threads, of the process
doing the work: this one in process, the server over HTTP) at the cheaper of the two batch
sizes, since a padded batch can cost more per input on a CPU, priced at
`--vcpu-hour-usd`, which is an input, not a measurement; the teacher's comes from the
recorded cost of its labels in `eval/labels/` when a Claude run exists, and its latency from
`--teacher-live N` timed calls (needs a key). The cascade row combines them with the coverage
`make eval-calibration` measured.

Raw output: `eval/results/bench-<dataset>-<teacher>/` (`report.json`, `meta.json`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np

from anvil import __version__, log
from anvil.errors import AnvilError, ConfigError
from anvil.settings import Settings
from eval.classify import LABELS_DIR, TeacherKind, git_provenance
from eval.render import SMOKE_SUFFIX

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "eval" / "results"
DEFAULT_DATASET = "banking77"
DEFAULT_THREADS = 4
DEFAULT_VCPU_HOUR_USD = 0.05
BATCHES = (1, 32)

logger = log.get_logger("anvil.eval.bench")


def _timings(seconds: Sequence[float], inputs_per_call: int) -> dict[str, object]:
    ms = np.asarray(seconds, dtype=np.float64) * 1000
    total = float(np.sum(seconds))
    return {
        "calls": len(seconds),
        "batch": inputs_per_call,
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
        "inputs_per_s": len(seconds) * inputs_per_call / total if total else 0.0,
    }


def _batches(texts: Sequence[str], size: int) -> Iterator[list[str]]:
    i = 0
    while True:
        yield [texts[(i + j) % len(texts)] for j in range(size)]
        i += size


def time_calls(
    fn: Callable[[list[str]], object],
    texts: Sequence[str],
    batch: int,
    calls: int,
    warmup: int,
    cpu: Callable[[], float] = time.process_time,
) -> dict[str, object]:
    """Time `calls` calls of `fn` on consecutive batches of `texts`, after `warmup` calls.

    `cpu` reads the CPU time (user + system, every thread) of whatever does the work: this
    process for an in-process engine, the server process for HTTP. Its difference over the
    timed calls is the measured CPU cost, which is less than threads x wall time whenever a
    part of the work (tokenizing, Python, HTTP) runs on one core."""
    gen = _batches(texts, batch)
    for _ in range(warmup):
        fn(next(gen))
    seconds: list[float] = []
    cpu_start = cpu()
    for _ in range(calls):
        b = next(gen)
        started = time.perf_counter()
        fn(b)
        seconds.append(time.perf_counter() - started)
    out = _timings(seconds, batch)
    out["cpu_seconds"] = cpu() - cpu_start
    return out


def process_cpu_seconds(pid: int) -> float:
    """User + system CPU seconds of process `pid`, all threads (Linux `/proc/<pid>/stat`)."""
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    # After the command name: fields[11] is utime and fields[12] stime, in clock ticks.
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@contextlib.contextmanager
def served(bundle: Path, variant: str, threads: int) -> Iterator[tuple[str, int]]:
    """`anvil serve` in a subprocess; yields its base URL and pid once `/healthz` answers."""
    import httpx

    port = _free_port()
    # A file, not a pipe: a pipe nobody reads blocks the server once it fills.
    log_file = tempfile.TemporaryFile()  # noqa: SIM115 - closed in the finally below
    env = {**os.environ, "ANVIL_SERVE_THREADS": str(threads), "ANVIL_LOG_LEVEL": "WARNING"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "anvil.cli",
            "serve",
            str(bundle),
            "--variant",
            variant,
            "--teacher",
            "none",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=log_file,
    )
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 120
        while True:
            if proc.poll() is not None:
                log_file.seek(0)
                err = log_file.read().decode(errors="replace")
                raise ConfigError(f"anvil serve exited with {proc.returncode}: {err[-500:]}")
            try:
                if httpx.get(f"{url}/healthz", timeout=1).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            if time.monotonic() > deadline:
                raise ConfigError("anvil serve did not answer /healthz within 120 s")
            time.sleep(0.2)
        yield url, proc.pid
    finally:
        log_file.close()
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def bench_http(
    bundle: Path, variant: str, texts: Sequence[str], threads: int, calls: int, warmup: int
) -> dict[str, object]:
    import httpx

    out: dict[str, object] = {}
    with (
        served(bundle, variant, threads) as (url, pid),
        httpx.Client(base_url=url, timeout=60) as c,
    ):

        def post(batch: list[str]) -> object:
            r = c.post("/predict", json={"texts": batch})
            r.raise_for_status()
            return r.json()

        for b in BATCHES:
            out[f"batch{b}"] = time_calls(
                post,
                texts,
                b,
                calls if b == 1 else calls // 4,
                warmup,
                cpu=lambda: process_cpu_seconds(pid),
            )
        # Four clients at batch 1, each with its own connection: what one replica sustains.
        clients, per_client = 4, calls // 2
        seconds: list[float] = []
        lock = threading.Lock()

        def worker(offset: int) -> None:
            with httpx.Client(base_url=url, timeout=60) as own:
                local: list[float] = []
                for i in range(per_client):
                    text = texts[(offset + i) % len(texts)]
                    started = time.perf_counter()
                    own.post("/predict", json={"texts": [text]}).raise_for_status()
                    local.append(time.perf_counter() - started)
                with lock:
                    seconds.extend(local)

        pool = [threading.Thread(target=worker, args=(k * 97,)) for k in range(clients)]
        started = time.perf_counter()
        for t in pool:
            t.start()
        for t in pool:
            t.join()
        wall = time.perf_counter() - started
        stats = _timings(seconds, 1)
        stats["inputs_per_s"] = len(seconds) / wall
        stats["clients"] = clients
        out["concurrent4_batch1"] = stats
    return out


def teacher_cost(dataset: str) -> dict[str, object] | None:
    """Mean recorded cost and tokens per input of every Claude label file for `dataset`."""
    rows: list[dict[str, object]] = []
    failures = 0
    for path in sorted((LABELS_DIR / dataset).glob("claude-*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("label") is None:
                failures += 1  # a refused or invalid answer: billed, but not a label
            rows.append(row)
    if not rows:
        return None

    def mean(key: str) -> float:
        return statistics.fmean(float(str(r.get(key) or 0)) for r in rows)

    # Per input sent, failures included: a refused input is billed like any other.
    return {
        "labels": len(rows) - failures,
        "failures": failures,
        "usd_per_1k": 1000 * mean("cost_usd"),
        "input_tokens_mean": mean("input_tokens"),
        "output_tokens_mean": mean("output_tokens"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.bench", description=__doc__)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--teacher", choices=["auto", "claude", "replay", "gold"], default="auto")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--calls", type=int, default=400, help="timed calls at batch 1")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--vcpu-hour-usd", type=float, default=DEFAULT_VCPU_HOUR_USD)
    parser.add_argument(
        "--teacher-live", type=int, default=0, help="time N live teacher calls (needs a key)"
    )
    args = parser.parse_args(argv)
    settings = Settings.from_env()
    log.configure(settings.log_level)
    command = "python -m eval.bench " + " ".join(argv if argv is not None else sys.argv[1:])
    provenance = git_provenance()
    smoke = args.calls != 400 or args.threads != DEFAULT_THREADS
    started = time.perf_counter()
    try:
        report = run(
            args.dataset,
            args.teacher,
            settings,
            threads=args.threads,
            calls=args.calls,
            warmup=args.warmup,
            vcpu_hour_usd=args.vcpu_hour_usd,
            teacher_live=args.teacher_live,
        )
    except AnvilError as exc:
        print(f"error: {exc}", file=sys.stderr)  # noqa: T201 - the CLI's error line
        return exc.exit_code
    teacher_model = str(report["teacher_model"])
    out = RESULTS_DIR / (f"bench-{args.dataset}-{teacher_model}" + (SMOKE_SUFFIX if smoke else ""))
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "command": command,
        **provenance,
        "anvil": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": _cpu_model(),
        "cpus": os.cpu_count(),
        "threads": args.threads,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "spent_usd": report.get("spent_usd", 0.0),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n", "utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", "utf-8")
    print(f"bench {args.dataset}: raw output {out}")  # noqa: T201 - the final summary
    return 0


def _cpu_model() -> str:
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def run(
    dataset: str,
    teacher: TeacherKind,
    settings: Settings,
    *,
    threads: int,
    calls: int,
    warmup: int,
    vcpu_hour_usd: float,
    teacher_live: int,
) -> dict[str, object]:
    import torch

    from anvil.runtime import OnnxStudent
    from anvil.students.encoder import EncoderClassifier
    from eval.export import bundle_dir
    from eval.student import trained_student

    ts = trained_student(dataset, teacher, settings)
    bundle = bundle_dir(ts)
    if not (bundle / "anvil_export.json").exists():
        raise ConfigError(f"no export bundle at {bundle}; run `make eval-export` first")
    texts = [it.text for it, _ in ts.heldout]
    engines: dict[str, dict[str, object]] = {}

    torch.set_num_threads(threads)
    pt = EncoderClassifier.load(ts.path)
    engines["pytorch"] = {
        f"batch{b}": time_calls(
            pt.predict_proba, texts, b, calls if b == 1 else calls // 4, warmup
        )
        for b in BATCHES
    }
    del pt
    variants = json.loads((bundle / "anvil_export.json").read_text(encoding="utf-8"))["variants"]
    servable = [v for v in ("fp32", "int8") if variants.get(v, {}).get("passed")]
    for variant in servable:
        model = OnnxStudent.load(bundle, "int8" if variant == "int8" else "fp32", threads=threads)
        engines[f"onnx-{variant}"] = {
            f"batch{b}": time_calls(
                model.predict_proba, texts, b, calls if b == 1 else calls // 4, warmup
            )
            for b in BATCHES
        }
        engines[f"http-{variant}"] = bench_http(bundle, variant, texts, threads, calls, warmup)

    cost: dict[str, object] = {"vcpu_hour_usd": vcpu_hour_usd, "assumed_price": True}
    for name, e in engines.items():
        # Per batch size: on a CPU a padded batch can cost more per input than single inputs,
        # so the price is quoted at whichever batch size is cheaper.
        per_batch: dict[str, float] = {}
        for b in BATCHES:
            stats = e[f"batch{b}"]
            assert isinstance(stats, dict)
            inputs = int(str(stats["calls"])) * int(str(stats["batch"]))
            per_batch[f"batch{b}"] = float(str(stats["cpu_seconds"])) * 1000 / inputs
        best = min(per_batch, key=per_batch.__getitem__)
        cost[name] = {
            "core_seconds_per_1k": per_batch,
            "cheapest": best,
            "usd_per_1k": per_batch[best] / 3600 * vcpu_hour_usd,
        }

    teacher_stats: dict[str, object] = {"model": ts.teacher_model}
    recorded = teacher_cost(dataset) if ts.kind != "gold" else None
    teacher_stats["recorded_cost"] = recorded
    spent = 0.0
    if teacher_live:
        live, spent = _teacher_latency(ts, settings, teacher_live)
        teacher_stats["live"] = live
    calibration = variants and json.loads(
        (bundle / "anvil_export.json").read_text(encoding="utf-8")
    ).get("calibration")
    return {
        "dataset": dataset,
        "teacher_kind": ts.kind,
        "teacher_model": ts.teacher_model,
        "student_model": ts.spec.student.model,
        "student_run": ts.path.name,
        "bundle": str(bundle.relative_to(settings.home))
        if bundle.is_relative_to(settings.home)
        else str(bundle),
        "inputs": len(texts),
        "mean_words": statistics.fmean(len(t.split()) for t in texts),
        "threads": threads,
        "engines": engines,
        "cost": cost,
        "teacher": teacher_stats,
        "calibration": calibration,
        "spent_usd": spent + ts.spent_usd,
    }


def _teacher_latency(ts: object, settings: Settings, n: int) -> tuple[dict[str, object], float]:
    """Time `n` uncached live teacher calls, one input each (needs a key)."""
    from anvil.teacher import AnthropicTransport, ClaudeTeacher
    from eval.student import TrainedStudent

    assert isinstance(ts, TrainedStudent)
    # No cache: a timing run must reach the API every time.
    teacher = ClaudeTeacher(ts.spec, AnthropicTransport(), cache=None, workers=1)
    seconds: list[float] = []
    spent = 0.0
    for item, _ in ts.heldout[:n]:
        started = time.perf_counter()
        batch = teacher.label([item])
        seconds.append(time.perf_counter() - started)
        spent += batch.spent_usd
    return _timings(seconds, 1), spent


if __name__ == "__main__":
    raise SystemExit(main())
