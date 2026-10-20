"""The HTTP server: the student answers, and inputs it is unsure of escalate to the teacher.

`create_app` builds the FastAPI app around any `ServedModel` (in production an
`anvil.runtime.OnnxStudent`); `load_app` loads an export bundle and picks the teacher. This
module runs in the torch-free serving image, so it must never import torch or transformers.

Escalation: an input whose top student probability is below the threshold is sent to the
teacher; every such input of a request goes in one `Teacher.label` call. Whatever goes wrong
with the teacher (a failure for one input, an exception, a label outside the model's label
set), the request still succeeds: those inputs get the student's answer and an `error`. A
calibration that found no threshold means every input goes to the teacher, so such a bundle
cannot be served without one (an explicit `threshold` overrides it). Input texts are never
logged, and a teacher exception reaches the caller as its type name only.
"""

# No `from __future__ import annotations`: FastAPI resolves endpoint annotations at runtime,
# and the request model is built per app (its limits are create_app arguments).

import re
import threading
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol

import numpy as np
import numpy.typing as npt
from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from anvil.calibrate import THRESHOLD_TOLERANCE
from anvil.errors import ConfigError
from anvil.log import get_logger
from anvil.settings import Settings
from anvil.spec import TaskSpec
from anvil.teacher.base import BatchAbortedError, Item, LabelBatch, Teacher

log = get_logger(__name__)

TeacherMode = Literal["auto", "claude", "fake", "none"]
Variant = Literal["fp32", "int8"]
LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,128}")
_MAX_ERROR_CHARS = 200


class ServedModel(Protocol):
    """What the server needs from a student (read-only, so plain attributes satisfy it)."""

    @property
    def labels(self) -> list[str]: ...
    @property
    def temperature(self) -> float: ...
    @property
    def threshold(self) -> float | None: ...
    @property
    def variant(self) -> str: ...
    @property
    def meta(self) -> Mapping[str, object]: ...

    def predict_proba(self, texts: Sequence[str]) -> npt.NDArray[np.float32]:
        """Calibrated class probabilities, shape (len(texts), len(labels))."""
        ...


class Prediction(BaseModel):
    label: str
    confidence: float
    """The student's top probability (also when the teacher answered)."""
    source: Literal["student", "teacher"]
    student_label: str
    below_threshold: bool
    """Confidence is under the threshold, or the calibration found no threshold (then every
    input counts as below it); false everywhere when the bundle has no threshold and asked for
    no target."""
    escalated: bool
    """The input was sent to the teacher (it may still have fallen back to the student)."""
    error: str | None
    """Why an escalated input fell back to the student's answer."""


class ModelInfo(BaseModel):
    variant: str
    threshold: float | None


class PredictResponse(BaseModel):
    predictions: list[Prediction]
    model: ModelInfo
    request_id: str


class Health(BaseModel):
    status: Literal["ok"]
    variant: str
    threshold: float | None
    labels: int
    teacher: str | None
    escalate_all: bool
    """The bundle's calibration found no threshold reaching its target kappa."""


class Metrics:
    """Counters and a latency histogram, rendered in the Prometheus text format."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = 0
        self.inputs = {"student": 0, "teacher": 0}
        self.escalations = 0
        self.escalation_failures = 0
        self.spent_usd = 0.0
        self._buckets = [0] * len(LATENCY_BUCKETS)
        self._latency_sum = 0.0

    def observe(
        self, *, student: int, teacher: int, escalated: int, failed: int, spent: float, s: float
    ) -> None:
        with self._lock:
            self.requests += 1
            self.inputs["student"] += student
            self.inputs["teacher"] += teacher
            self.escalations += escalated
            self.escalation_failures += failed
            self.spent_usd += spent
            self._latency_sum += s
            for i, bound in enumerate(LATENCY_BUCKETS):
                if s <= bound:
                    self._buckets[i] += 1

    def render(self) -> str:
        with self._lock:
            lines = [
                "# HELP anvil_requests_total Predict requests answered.",
                "# TYPE anvil_requests_total counter",
                f"anvil_requests_total {self.requests}",
                "# HELP anvil_inputs_total Inputs answered, by who answered them.",
                "# TYPE anvil_inputs_total counter",
                *(f'anvil_inputs_total{{source="{k}"}} {v}' for k, v in self.inputs.items()),
                "# HELP anvil_escalations_total Inputs sent to the teacher.",
                "# TYPE anvil_escalations_total counter",
                f"anvil_escalations_total {self.escalations}",
                "# HELP anvil_escalation_failures_total Escalated inputs that fell back to the"
                " student.",
                "# TYPE anvil_escalation_failures_total counter",
                f"anvil_escalation_failures_total {self.escalation_failures}",
                "# HELP anvil_teacher_spent_usd_total Dollars spent on teacher calls.",
                "# TYPE anvil_teacher_spent_usd_total counter",
                f"anvil_teacher_spent_usd_total {self.spent_usd!r}",
                "# HELP anvil_request_seconds Predict request latency.",
                "# TYPE anvil_request_seconds histogram",
                *(
                    f'anvil_request_seconds_bucket{{le="{b}"}} {n}'
                    for b, n in zip(LATENCY_BUCKETS, self._buckets, strict=True)
                ),
                f'anvil_request_seconds_bucket{{le="+Inf"}} {self.requests}',
                f"anvil_request_seconds_sum {self._latency_sum!r}",
                f"anvil_request_seconds_count {self.requests}",
            ]
        return "\n".join(lines) + "\n"


@dataclass
class _Escalation:
    """The teacher's answers for one request: a label or an error per request position."""

    labels: dict[int, str]
    errors: dict[int, str]
    spent_usd: float


def _short(reason: str) -> str:
    return reason if len(reason) <= _MAX_ERROR_CHARS else reason[: _MAX_ERROR_CHARS - 3] + "..."


def _escalate(
    teacher: Teacher, request_id: str, texts: Sequence[str], positions: list[int], known: set[str]
) -> _Escalation:
    ids = {f"{request_id}:{i}": i for i in positions}
    error: str | None = None
    try:
        batch = teacher.label([Item(id=k, text=texts[i]) for k, i in ids.items()])
    except BatchAbortedError as exc:
        # Keep the answers that completed (they are paid for); the rest fall back.
        batch, error = exc.batch, f"teacher batch aborted: {type(exc.cause).__name__}"
        log.warning("teacher batch aborted", extra={"request_id": request_id, "error": error})
    except Exception as exc:  # any teacher failure degrades to the student, never a 500
        # Only the exception type leaves this function: its message may quote an API error
        # body or the input text, which belong neither in a response nor in the log.
        batch, error = LabelBatch(), f"teacher error: {type(exc).__name__}"
        log.warning(
            "teacher call failed",
            extra={"request_id": request_id, "error_type": type(exc).__name__},
        )
    out = _Escalation(labels={}, errors={}, spent_usd=batch.spent_usd)
    for f in batch.failures:
        if f.input_id in ids:
            out.errors[ids[f.input_id]] = f"teacher failed: {f.reason}"
    for lb in batch.labels:
        if lb.input_id not in ids:
            continue
        pos = ids[lb.input_id]
        if lb.label in known:
            out.labels[pos] = lb.label
            out.errors.pop(pos, None)
        else:
            out.errors[pos] = f"teacher label {lb.label!r} is not in the model's label set"
    for pos in positions:
        if pos not in out.labels and pos not in out.errors:
            out.errors[pos] = error or "teacher returned no answer"
        if pos in out.errors:
            out.errors[pos] = _short(out.errors[pos])
    return out


def _request_id(header: str | None) -> str:
    if header is not None and _REQUEST_ID_RE.fullmatch(header):
        return header
    return uuid.uuid4().hex


def create_app(
    model: ServedModel,
    teacher: Teacher | None,
    *,
    threshold: float | None = None,
    max_batch: int = 256,
    max_chars: int = 20_000,
    max_body_bytes: int | None = None,
) -> FastAPI:
    """The serving app. `threshold` overrides the bundle's calibrated `model.threshold`.

    `max_body_bytes` (default: room for `max_batch` texts of `max_chars` four-byte characters)
    is checked against Content-Length before the body is read, since the validation limits
    only apply once the whole body is in memory.
    """
    thr = threshold if threshold is not None else model.threshold
    # A calibration that asked for a target kappa and found no threshold reaching it means the
    # student cannot be trusted at any coverage: every input goes to the teacher.
    calibration = model.meta.get("calibration")
    unreachable = (
        thr is None
        and isinstance(calibration, Mapping)
        and calibration.get("target_kappa") is not None
    )
    if thr is not None and not (0.0 <= thr <= 1.0):  # also rejects NaN
        raise ConfigError(f"threshold {thr} must be between 0 and 1")
    if max_batch < 1 or max_chars < 1:
        raise ConfigError("max_batch and max_chars must be at least 1")
    max_body = max_body_bytes if max_body_bytes is not None else 4 * max_batch * max_chars + 4096
    if max_body < 1:
        raise ConfigError("max_body_bytes must be at least 1")
    if unreachable and teacher is None:
        raise ConfigError(
            "the bundle's calibration found no threshold that keeps its target kappa, so every"
            " input must go to the teacher, and no teacher is configured: pass --teacher claude"
            " (with credentials) or --teacher fake, or an explicit --threshold to answer with"
            " the student anyway"
        )
    labels = list(model.labels)
    known = set(labels)
    metrics = Metrics()
    teacher_id = teacher.model_id if teacher is not None else None
    info = ModelInfo(variant=model.variant, threshold=thr)
    if unreachable:
        log.warning("the calibration found no threshold that keeps its target: escalating all")
    elif thr is None:
        log.warning("no threshold: every input is answered by the student, none escalated")
    if model.variant == "int8" and (
        teacher is None
        or (model.threshold is not None and thr is not None and thr < model.threshold)
    ):
        # The int8 parity gate scored argmax agreement on the inputs the cascade answers at the
        # bundle's threshold; below it, or with no teacher, int8 also answers inputs the gate
        # only covered through the kappa-drop limit.
        log.warning(
            "the int8 variant answers inputs its parity gate scored only by kappa drop",
            extra={"bundle_threshold": model.threshold, "threshold": thr, "teacher": teacher_id},
        )

    class PredictRequest(BaseModel):
        texts: Annotated[
            list[Annotated[str, Field(max_length=max_chars)]],
            Field(min_length=1, max_length=max_batch),
        ]

    app = FastAPI(title="anvil", summary="A distilled student with teacher escalation.")
    app.state.metrics = metrics

    @app.middleware("http")
    async def limit_body(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.method == "POST":
            length = request.headers.get("content-length")
            if length is None:
                return JSONResponse({"detail": "Content-Length is required"}, status_code=411)
            if not length.isdigit() or int(length) > max_body:
                return JSONResponse(
                    {"detail": f"request body must be at most {max_body} bytes"},
                    status_code=413,
                )
        return await call_next(request)

    # Plain `def` endpoints: FastAPI runs them in its threadpool, since ORT and the teacher
    # block.
    @app.post("/predict", response_model=PredictResponse)
    def predict(
        body: PredictRequest,
        response: Response,
        x_request_id: Annotated[str | None, Header()] = None,
    ) -> PredictResponse:
        start = time.perf_counter()
        rid = _request_id(x_request_id)
        response.headers["X-Request-ID"] = rid
        texts = body.texts
        probs = np.asarray(model.predict_proba(texts), dtype=np.float64)
        top = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        below = [
            unreachable or (thr is not None and float(c) < thr - THRESHOLD_TOLERANCE) for c in conf
        ]
        positions = [i for i, b in enumerate(below) if b] if teacher is not None else []
        esc = (
            _escalate(teacher, rid, texts, positions, known)
            if teacher is not None and positions
            else _Escalation(labels={}, errors={}, spent_usd=0.0)
        )
        escalated = set(positions)
        predictions = []
        for i in range(len(texts)):
            student_label = labels[int(top[i])]
            teacher_label = esc.labels.get(i)
            predictions.append(
                Prediction(
                    label=teacher_label if teacher_label is not None else student_label,
                    confidence=float(conf[i]),
                    source="teacher" if teacher_label is not None else "student",
                    student_label=student_label,
                    below_threshold=below[i],
                    escalated=i in escalated,
                    error=esc.errors.get(i),
                )
            )
        seconds = time.perf_counter() - start
        metrics.observe(
            student=len(texts) - len(esc.labels),
            teacher=len(esc.labels),
            escalated=len(positions),
            failed=len(esc.errors),
            spent=esc.spent_usd,
            s=seconds,
        )
        log.info(
            "predict",
            extra={
                "request_id": rid,
                "n": len(texts),
                "chars": sum(len(t) for t in texts),
                "escalated": len(positions),
                "teacher_answered": len(esc.labels),
                "teacher_failed": len(esc.errors),
                "spent_usd": esc.spent_usd,
                "latency_ms": round(seconds * 1000, 2),
                "threshold": thr,
            },
        )
        return PredictResponse(predictions=predictions, model=info, request_id=rid)

    @app.get("/healthz", response_model=Health)
    def healthz() -> Health:
        return Health(
            status="ok",
            variant=model.variant,
            threshold=thr,
            labels=len(labels),
            teacher=teacher_id,
            escalate_all=unreachable,
        )

    @app.get("/metrics")
    def metrics_endpoint() -> Response:
        return Response(metrics.render(), media_type="text/plain; version=0.0.4")

    return app


Loader = Callable[[Path, Variant, int | None], ServedModel]


def _load_onnx(bundle: Path, variant: Variant, threads: int | None) -> ServedModel:
    from anvil.runtime import OnnxStudent  # lazy: needs onnxruntime and tokenizers

    return OnnxStudent.load(bundle, variant, threads=threads)


def _spec_of(model: ServedModel, bundle: Path) -> TaskSpec | None:
    raw = model.meta.get("spec")
    if raw is None:
        return None
    try:
        return TaskSpec.model_validate(raw)
    except ValidationError as exc:
        raise ConfigError(
            f"bundle {bundle} has an invalid task spec; re-export it: {exc}"
        ) from exc


def _teacher_for(
    mode: TeacherMode, spec: TaskSpec | None, settings: Settings, bundle: Path
) -> Teacher | None:
    from anvil.teacher import build_teacher
    from anvil.teacher.fake import FakeTeacher

    if mode == "none":
        return None
    if spec is None:
        if mode == "auto":
            log.warning(
                "escalation disabled: the bundle has no task spec", extra={"bundle": str(bundle)}
            )
            return None
        raise ConfigError(
            f"bundle {bundle} has no task spec, so it cannot build a {mode} teacher;"
            " re-export it or use --teacher none"
        )
    if mode == "fake":
        return FakeTeacher(spec)  # ConfigError without a `fake:` section
    if mode == "claude":
        return build_teacher(spec, settings, provider="claude")
    try:
        return build_teacher(spec, settings, provider="claude")
    except ConfigError:
        log.warning("escalation disabled: no teacher credentials")
        return None


def load_app(
    bundle: Path,
    *,
    variant: Variant = "fp32",
    teacher: TeacherMode = "auto",
    threshold: float | None = None,
    settings: Settings | None = None,
    threads: int | None = None,
    loader: Loader = _load_onnx,
) -> FastAPI:
    """Load an export bundle and serve it. `teacher="auto"` escalates to Claude when
    credentials are available and disables escalation otherwise. `loader` exists for tests."""
    if not bundle.is_dir():
        raise ConfigError(f"bundle {bundle} is not a directory; create one with `anvil export`")
    settings = settings or Settings.from_env()
    model = loader(bundle, variant, threads if threads is not None else settings.serve_threads)
    spec = _spec_of(model, bundle)
    t = _teacher_for(teacher, spec, settings, bundle)
    app = create_app(model, t, threshold=threshold)
    log.info(
        "serve start",
        extra={
            "bundle": str(bundle),
            "variant": model.variant,
            "threshold": threshold if threshold is not None else model.threshold,
            "teacher": t.model_id if t is not None else None,
        },
    )
    return app
