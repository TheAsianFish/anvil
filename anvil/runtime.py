"""Serve an exported student with ONNX Runtime and the Rust tokenizer, without torch.

This module runs inside the slim serving image, which has neither torch nor transformers, so
it must not import them, directly or through another `anvil` module. `anvil export` writes
the bundle this reads (see `anvil.export`):

    model.onnx (+ model.onnx.data)   fp32 graph, logits only
    model.int8.onnx                  optional dynamic-int8 graph
    tokenizer.json                   the student's fast tokenizer
    anvil_export.json                BUNDLE_META: labels, max_length, calibration, parity, ...
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
from tokenizers import Tokenizer

from anvil.errors import ExportError
from anvil.log import get_logger

BUNDLE_META = "anvil_export.json"
TOKENIZER_FILE = "tokenizer.json"
BUNDLE_FORMAT = 1
Variant = Literal["fp32", "int8"]
VARIANT_FILES: dict[str, str] = {"fp32": "model.onnx", "int8": "model.int8.onnx"}

LOG = get_logger(__name__)
_EXPORT_HINT = "run `anvil export` to (re)build it"


def softmax(logits: npt.NDArray[np.float32], temperature: float = 1.0) -> npt.NDArray[np.float32]:
    """Row-wise softmax of `logits / temperature`, shifted by the row max for stability."""
    z = logits.astype(np.float64) / temperature
    z -= z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    out: npt.NDArray[np.float32] = (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)
    return out


def read_bundle_meta(bundle: Path) -> dict[str, object]:
    path = bundle / BUNDLE_META
    if not path.is_file():
        raise ExportError(
            f"{bundle} is not an exported Anvil bundle (no {BUNDLE_META}); {_EXPORT_HINT}"
        )
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ExportError(f"{path} is unreadable ({exc}); {_EXPORT_HINT}") from exc
    if not isinstance(meta, dict) or meta.get("format") != BUNDLE_FORMAT:
        raise ExportError(
            f"{path} is not a format-{BUNDLE_FORMAT} Anvil bundle; {_EXPORT_HINT} with this"
            " version of Anvil"
        )
    return meta


def _number(meta: dict[str, object], key: str, where: Path, default: float | None = None) -> float:
    value = meta.get(key)
    if value is None and default is not None:
        return default
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ExportError(f"{where}: {key} is {value!r}, expected a number; {_EXPORT_HINT}")
    return float(value)


class OnnxStudent:
    """An exported classifier: tokenizes with `tokenizer.json`, runs one ONNX graph."""

    def __init__(
        self,
        path: Path,
        model_file: Path,
        meta: dict[str, object],
        variant: str,
        threads: int | None = None,
    ) -> None:
        self.path = path
        self.meta = meta
        self.variant = variant
        where = path / BUNDLE_META
        labels = meta.get("labels")
        if not isinstance(labels, list) or not labels:
            raise ExportError(f"{where} has no labels; {_EXPORT_HINT}")
        self.labels = [str(x) for x in labels]
        self.max_length = int(_number(meta, "max_length", where))
        pad_id = int(_number(meta, "pad_token_id", where))
        self.temperature = _number(meta, "temperature", where, default=1.0)
        self.threshold = (
            _number(meta, "threshold", where) if meta.get("threshold") is not None else None
        )
        if self.temperature <= 0:
            raise ExportError(f"{where}: temperature must be > 0; {_EXPORT_HINT}")

        tok_path = path / TOKENIZER_FILE
        if not tok_path.is_file():
            raise ExportError(f"{path} has no {TOKENIZER_FILE}; {_EXPORT_HINT}")
        tok = Tokenizer.from_file(str(tok_path))
        # Reset whatever padding/truncation the file was saved with, so behaviour depends only
        # on the bundle meta and matches the HF call `padding=True, truncation=True`.
        tok.no_padding()
        tok.no_truncation()
        tok.enable_truncation(
            self.max_length, direction=str(meta.get("truncation_side") or "right")
        )
        pad_token = tok.id_to_token(pad_id)
        if pad_token is None:
            raise ExportError(f"pad_token_id {pad_id} is not in {tok_path}; {_EXPORT_HINT}")
        tok.enable_padding(
            direction=str(meta.get("padding_side") or "right"), pad_id=pad_id, pad_token=pad_token
        )
        self._tokenizer = tok

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        if threads is not None:
            opts.intra_op_num_threads = threads
        try:
            self._session = ort.InferenceSession(
                str(model_file), opts, providers=["CPUExecutionProvider"]
            )
        except Exception as exc:  # ORT raises its own untyped exception classes
            raise ExportError(f"could not load {model_file} ({exc}); {_EXPORT_HINT}") from exc
        self._inputs = [i.name for i in self._session.get_inputs()]
        unknown = set(self._inputs) - {"input_ids", "attention_mask"}
        if unknown:
            raise ExportError(
                f"{model_file} expects inputs {sorted(unknown)} that Anvil does not produce;"
                f" {_EXPORT_HINT}"
            )
        self._output = self._session.get_outputs()[0].name

    @classmethod
    def load(
        cls,
        bundle: Path,
        variant: Variant = "fp32",
        *,
        threads: int | None = None,
        require_parity: bool = True,
    ) -> OnnxStudent:
        """Open `variant` of the bundle. With `require_parity` (the default), a variant whose
        export failed its parity gate is refused: it is never served."""
        meta = read_bundle_meta(bundle)
        if variant not in VARIANT_FILES:
            raise ExportError(f"unknown variant {variant!r}; use one of {sorted(VARIANT_FILES)}")
        variants = meta.get("variants")
        info = variants.get(variant) if isinstance(variants, dict) else None
        if not isinstance(info, dict):
            flag = " --int8" if variant == "int8" else ""
            raise ExportError(
                f"{bundle} has no {variant} variant; run `anvil export{flag}` to add it"
            )
        if require_parity and info.get("passed") is not True:
            raise ExportError(
                f"the {variant} variant in {bundle} failed its parity gate at export"
                f" ({info.get('parity')}), so it is not served; use the fp32 variant or"
                f" {_EXPORT_HINT}"
            )
        model_file = bundle / str(info.get("file") or VARIANT_FILES[variant])
        if not model_file.is_file():
            raise ExportError(f"{model_file} is missing from the bundle; {_EXPORT_HINT}")
        student = cls(bundle, model_file, meta, variant, threads)
        LOG.info("onnx student loaded", extra={"bundle": str(bundle), "variant": variant})
        return student

    def encode(self, texts: Sequence[str]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        """input_ids and attention_mask, padded to the longest text, cut at max_length."""
        if not texts:
            empty = np.zeros((0, 0), dtype=np.int64)
            return empty, empty.copy()
        encs = self._tokenizer.encode_batch(list(texts))
        ids = np.asarray([e.ids for e in encs], dtype=np.int64)
        mask = np.asarray([e.attention_mask for e in encs], dtype=np.int64)
        return ids, mask

    def logits(self, texts: Sequence[str], batch_size: int = 32) -> npt.NDArray[np.float32]:
        out: list[npt.NDArray[np.float32]] = []
        for start in range(0, len(texts), batch_size):
            ids, mask = self.encode(texts[start : start + batch_size])
            feed = {"input_ids": ids, "attention_mask": mask}
            result = self._session.run([self._output], {k: feed[k] for k in self._inputs})
            out.append(np.asarray(result[0], dtype=np.float32))
        if not out:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        return np.concatenate(out)

    def predict_proba(self, texts: Sequence[str], batch_size: int = 32) -> npt.NDArray[np.float32]:
        """Class probabilities, calibrated by the bundle's temperature (1.0 if uncalibrated)."""
        return softmax(self.logits(texts, batch_size), self.temperature)

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [self.labels[i] for i in self.logits(texts).argmax(axis=1)]
