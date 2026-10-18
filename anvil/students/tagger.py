"""Encoder tagger student: any Hugging Face encoder with a token-classification head.

The student tags each token `O`, `B-<type>` or `I-<type>` (`anvil.align.tag_names`) and its
spans are read back off the tags with the tokenizer's character offsets, so it needs a fast
(Rust) tokenizer: only those return an offset mapping. `tiny` builds a two-layer random-init
ModernBERT over the word-level vocabulary of `anvil.students.encoder`, so tests run offline
in seconds with the same code path as a pretrained `answerdotai/ModernBERT-base`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    ModernBertConfig,
    ModernBertForTokenClassification,
    PreTrainedTokenizerFast,
)

from anvil.align import AlignmentReport, Offset, snap_to_tokens, tag_names, tags_to_spans
from anvil.errors import StudentError
from anvil.spans import Span
from anvil.spec import StudentSpec
from anvil.students.encoder import TINY, build_tiny_tokenizer
from anvil.train.loop import TrainResult

META_FILE = "anvil_tagger.json"
IGNORE = -100
"""Target id of special and padding tokens: the cross-entropy's `ignore_index`."""


def build_tiny_tagger_model(
    tokenizer: PreTrainedTokenizerFast, tags: Sequence[str], max_length: int
) -> ModernBertForTokenClassification:
    """The same tiny ModernBERT as `encoder.build_tiny_model`, with a per-token head."""
    config = ModernBertConfig(
        id2label=dict(enumerate(tags)),
        label2id={name: i for i, name in enumerate(tags)},
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=max(128, max_length),
        layer_types=["full_attention", "full_attention"],
        pad_token_id=tokenizer.pad_token_id,
        cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )
    return ModernBertForTokenClassification(config)


class EncoderTagger:
    def __init__(
        self,
        model: Any,  # Any: a transformers PreTrainedModel with a token-classification head
        tokenizer: Any,  # Any: a transformers fast tokenizer
        types: list[str],
        max_length: int,
        extra_meta: dict[str, object] | None = None,
    ) -> None:
        if not getattr(tokenizer, "is_fast", False):
            raise StudentError(
                f"the tokenizer {type(tokenizer).__name__} is not a fast tokenizer, and only"
                " fast tokenizers report the character offsets a tagger needs; pick a student"
                " model that ships a tokenizer.json (most do) or use student.model: tiny"
            )
        self.model = model
        self.tokenizer = tokenizer
        self.types = types
        self.tags = tag_names(types)
        self.max_length = max_length
        self.meta: dict[str, object] = dict(extra_meta or {})
        self._tag_id = {tag: i for i, tag in enumerate(self.tags)}
        self.alignment: AlignmentReport | None = None
        """The snapping report of the last `fit` (what max_length and the vocabulary cost)."""

    @classmethod
    def build(cls, spec: StudentSpec, types: list[str], corpus: Sequence[str]) -> EncoderTagger:
        """A fresh, untrained tagger over `types` (spec order). `corpus` builds the tiny
        vocabulary (ignored otherwise)."""
        torch.manual_seed(spec.seed)
        tags = tag_names(types)
        id2label = dict(enumerate(tags))
        label2id = {v: k for k, v in id2label.items()}
        if spec.model == TINY:
            # Any: the tiny and pretrained branches return different transformers classes.
            tokenizer: Any = build_tiny_tokenizer(corpus)
            model: Any = build_tiny_tagger_model(tokenizer, tags, spec.max_length)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(spec.model)
                model = AutoModelForTokenClassification.from_pretrained(
                    spec.model,
                    num_labels=len(tags),
                    id2label=id2label,
                    label2id=label2id,
                )
            except OSError as exc:
                raise StudentError(
                    f"could not load student {spec.model!r}: {exc}; check the model id or use"
                    " student.model: tiny"
                ) from exc
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, tokenizer, types, spec.max_length)

    @property
    def device(self) -> torch.device:
        device: torch.device = next(self.model.parameters()).device
        return device

    def offsets(self, texts: Sequence[str]) -> list[list[Offset]]:
        """Each text's token offsets into the original text, truncated at `max_length`,
        special tokens included as `(0, 0)`. One entry per token `encode` produces for the
        text alone (padding excluded)."""
        if not texts:
            return []  # the tokenizer fails on an empty batch
        enc = self.tokenizer(
            list(texts),
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
        )
        return [[(int(s), int(e)) for s, e in row] for row in enc["offset_mapping"]]

    def targets(
        self,
        texts: Sequence[str],
        spans: Sequence[Sequence[Span]],
        offsets: Sequence[Sequence[Offset]] | None = None,
        input_ids: Sequence[str] | None = None,
    ) -> tuple[list[list[int]], AlignmentReport, list[list[Span]]]:
        """Per-token tag ids (`IGNORE` on special tokens), the snapping report merged over the
        texts, and the spans as the student can represent them (snapped to token boundaries,
        without those no token covers). `offsets` defaults to `self.offsets(texts)`.

        The report counts only what snapping adds (`no_token`, `token_conflict`, `snapped`);
        merge it into the character alignment's report before `check()`."""
        if len(spans) != len(texts):
            raise StudentError(f"{len(texts)} texts but {len(spans)} span lists")
        rows = self.offsets(texts) if offsets is None else offsets
        if len(rows) != len(texts):
            raise StudentError(f"{len(texts)} texts but {len(rows)} offset lists")
        report = AlignmentReport()
        ids: list[list[int]] = []
        represented: list[list[Span]] = []
        for i, (text, text_spans, row) in enumerate(zip(texts, spans, rows, strict=True)):
            name = input_ids[i] if input_ids is not None else str(i)
            aligned = snap_to_tokens(text, text_spans, row, input_id=name)
            report.merge(aligned.report)
            ids.append([IGNORE if tag is None else self._tag_id[tag] for tag in aligned.tags])
            represented.append(aligned.spans)
        return ids, report, represented

    def encode(self, texts: Sequence[str]) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = self.device
        return {k: v.to(device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def predict_tags(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> tuple[list[list[str | None]], list[list[float]]]:
        """Per text, one tag per token (`None` on special tokens) and the probability of that
        tag, over the tokens `offsets` lists (padding dropped)."""
        tags, probs, _ = self._predict(texts, batch_size)
        return tags, probs

    @torch.no_grad()
    def _predict(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> tuple[list[list[str | None]], list[list[float]], list[list[Offset]]]:
        self.model.eval()
        tags: list[list[str | None]] = []
        probs: list[list[float]] = []
        offsets = self.offsets(texts)
        for start in range(0, len(texts), batch_size):
            batch = self.encode(texts[start : start + batch_size])
            logits = self.model(**batch).logits
            best, best_id = torch.softmax(logits.float(), dim=-1).max(dim=-1)
            mask = batch["attention_mask"].bool()
            for k in range(best.shape[0]):
                row_offsets = offsets[start + k]
                # The mask drops padding on either side, whatever the tokenizer's padding_side.
                ids = best_id[k][mask[k]].tolist()
                if len(ids) != len(row_offsets):
                    raise StudentError(
                        f"{len(ids)} tokens encoded but {len(row_offsets)} offsets for one"
                        " text; the tokenizer pads or truncates inconsistently"
                    )
                tags.append(
                    [
                        None if e <= s else self.tags[t]
                        for t, (s, e) in zip(ids, row_offsets, strict=True)
                    ]
                )
                probs.append([float(x) for x in best[k][mask[k]].tolist()])
        return tags, probs, offsets

    def predict_spans(self, texts: Sequence[str]) -> list[list[Span]]:
        spans, _ = self.predict_with_confidence(texts)
        return spans

    def predict_with_confidence(
        self, texts: Sequence[str]
    ) -> tuple[list[list[Span]], npt.NDArray[np.float32]]:
        """Spans per text and one confidence per text: the minimum, over its real tokens, of
        the probability of the predicted tag.

        The minimum and not the mean, because one uncertain token is enough to get an entity
        wrong (a boundary one token off or a B read as I fails exact match), and an input's
        answer is only as good as its weakest token; a mean would let a long, easy input hide
        it. The cascade and active selection (M5) read this number. A text with no real token
        has nothing to get wrong and scores 1.0.
        """
        tags, probs, offsets = self._predict(texts)
        spans = [
            tags_to_spans(text, row_tags, row_offsets)
            for text, row_tags, row_offsets in zip(texts, tags, offsets, strict=True)
        ]
        confidence = np.array(
            [
                min((p for p, t in zip(row_p, row_t, strict=True) if t is not None), default=1.0)
                for row_p, row_t in zip(probs, tags, strict=True)
            ],
            dtype=np.float32,
        )
        return spans, confidence

    def fit(
        self,
        texts: Sequence[str],
        spans: Sequence[Sequence[Span]],
        spec: StudentSpec,
        metrics_path: Path | None = None,
        *,
        checkpoint_dir: Path | None = None,
        resume: bool = False,
        keep_checkpoint: bool = False,
    ) -> TrainResult:
        """Train on (text, verified spans) pairs; see `anvil.train.tagger.train_tagger`. The
        snapping report is kept in `self.alignment`."""
        # Imported here: the training module imports this one for its type hints.
        from anvil.train.tagger import train_tagger

        result, self.alignment = train_tagger(
            self,
            texts,
            spans,
            spec,
            metrics_path,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            keep_checkpoint=keep_checkpoint,
        )
        return result

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        meta = {
            "types": self.types,
            "tags": self.tags,
            "max_length": self.max_length,
            **self.meta,
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EncoderTagger:
        meta_path = path / META_FILE
        if not meta_path.exists():
            raise StudentError(f"{path} is not a saved Anvil tagger (no {META_FILE})")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            types = [str(t) for t in meta.pop("types")]
            tags = [str(t) for t in meta.pop("tags")]
            max_length = int(meta.pop("max_length"))
            model = AutoModelForTokenClassification.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
        except (KeyError, OSError, ValueError, TypeError) as exc:
            raise StudentError(
                f"{path} is not a complete Anvil tagger ({exc}); train one with `anvil train`"
            ) from exc
        if tags != tag_names(types):
            raise StudentError(
                f"{path} has tags {tags} that do not match its types {types}; the files were"
                " edited or come from another version, so retrain the tagger"
            )
        return cls(model, tokenizer, types, max_length, extra_meta=meta)
