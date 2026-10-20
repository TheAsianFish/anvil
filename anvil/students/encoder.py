"""Encoder classifier student: any Hugging Face encoder with a sequence-classification head.

`tiny` builds a two-layer random-init ModernBERT and a WordPiece vocabulary learned from the
training texts, so tests and the demo run offline in seconds with the same code path as a
pretrained `answerdotai/ModernBERT-base`.
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
    AutoModelForSequenceClassification,
    AutoTokenizer,
    ModernBertConfig,
    ModernBertForSequenceClassification,
    PreTrainedTokenizerFast,
)
from transformers.utils import logging as hf_logging

from anvil.errors import StudentError
from anvil.spec import StudentSpec
from anvil.train.loop import TrainResult

hf_logging.disable_progress_bar()  # type: ignore[no-untyped-call]
hf_logging.set_verbosity_error()  # type: ignore[no-untyped-call]

TINY = "tiny"
META_FILE = "anvil_student.json"
SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def build_tiny_tokenizer(texts: Sequence[str], vocab_size: int = 2000) -> PreTrainedTokenizerFast:
    """A word-level vocabulary of the corpus's most frequent words.

    Built by hand rather than with a `tokenizers` trainer because the trainers break frequency
    ties in hash-map order, so the same corpus gave different vocabularies across processes.
    Ties here are broken alphabetically.
    """
    from collections import Counter

    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

    normalizer = normalizers.BertNormalizer(lowercase=True)
    pre = pre_tokenizers.BertPreTokenizer()
    counts: Counter[str] = Counter()
    for text in texts:
        counts.update(word for word, _ in pre.pre_tokenize_str(normalizer.normalize_str(text)))
    words = sorted(counts, key=lambda w: (-counts[w], w))[: vocab_size - len(SPECIAL)]
    vocab = {token: i for i, token in enumerate([*SPECIAL, *words])}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    tok.normalizer = normalizer
    tok.pre_tokenizer = pre
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)]
    )
    return PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tok,
        pad_token="[PAD]",
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )


def build_tiny_model(
    tokenizer: PreTrainedTokenizerFast, labels: Sequence[str], max_length: int
) -> ModernBertForSequenceClassification:
    config = ModernBertConfig(
        id2label=dict(enumerate(labels)),
        label2id={name: i for i, name in enumerate(labels)},
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
    return ModernBertForSequenceClassification(config)


class EncoderClassifier:
    def __init__(
        self,
        model: Any,  # Any: a transformers PreTrainedModel with a classification head
        tokenizer: Any,  # Any: a transformers tokenizer
        labels: list[str],
        max_length: int,
        extra_meta: dict[str, object] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.labels = labels
        self.max_length = max_length
        self.meta: dict[str, object] = dict(extra_meta or {})

    @classmethod
    def build(
        cls, spec: StudentSpec, labels: list[str], corpus: Sequence[str]
    ) -> EncoderClassifier:
        """A fresh, untrained student. `corpus` builds the tiny vocabulary (ignored otherwise)."""
        torch.manual_seed(spec.seed)
        id2label = dict(enumerate(labels))
        label2id = {v: k for k, v in id2label.items()}
        if spec.model == TINY:
            # Any: the tiny and pretrained branches return different transformers classes.
            tokenizer: Any = build_tiny_tokenizer(corpus)
            model: Any = build_tiny_model(tokenizer, labels, spec.max_length)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(spec.model)
                model = AutoModelForSequenceClassification.from_pretrained(
                    spec.model,
                    num_labels=len(labels),
                    id2label=id2label,
                    label2id=label2id,
                )
            except OSError as exc:
                raise StudentError(
                    f"could not load student {spec.model!r}: {exc}; check the model id or use"
                    " student.model: tiny"
                ) from exc
        # Train on the GPU when there is one; CPU otherwise (the loop picks AMP to match).
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        return cls(model, tokenizer, labels, spec.max_length)

    @property
    def device(self) -> torch.device:
        """Where the model's parameters live; inputs are moved there by `encode`."""
        device: torch.device = next(self.model.parameters()).device
        return device

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

    def fit(
        self,
        texts: Sequence[str],
        targets: Sequence[int],
        spec: StudentSpec,
        metrics_path: Path | None = None,
        *,
        checkpoint_dir: Path | None = None,
        resume: bool = False,
        keep_checkpoint: bool = False,
    ) -> TrainResult:
        """Train on (text, label id) pairs; see `anvil.train.classifier.train_classifier`."""
        # Imported here: the training module imports this one for its type hints.
        from anvil.train.classifier import train_classifier

        return train_classifier(
            self,
            texts,
            targets,
            spec,
            metrics_path,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            keep_checkpoint=keep_checkpoint,
        )

    @torch.no_grad()
    def predict_logits(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> npt.NDArray[np.float32]:
        """Raw classification logits, shape (len(texts), len(labels)): what calibration fits a
        temperature to and what the ONNX export is checked against."""
        self.model.eval()
        out: list[npt.NDArray[np.float32]] = []
        for start in range(0, len(texts), batch_size):
            logits = self.model(**self.encode(texts[start : start + batch_size])).logits
            out.append(logits.float().cpu().numpy())
        if not out:
            return np.zeros((0, len(self.labels)), dtype=np.float32)
        return np.concatenate(out).astype(np.float32)

    def predict_proba(self, texts: Sequence[str], batch_size: int = 64) -> npt.NDArray[np.float32]:
        logits = torch.from_numpy(self.predict_logits(texts, batch_size))
        return torch.softmax(logits, dim=-1).numpy().astype(np.float32)

    @torch.no_grad()
    def predict_with_embeddings(
        self, texts: Sequence[str], batch_size: int = 64
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Class probabilities and one vector per text from the same forward pass: the last
        hidden layer mean-pooled over real (non-padding) tokens. Selection uses the vectors
        to tell near-duplicate inputs apart from genuinely different ones."""
        self.model.eval()
        probs: list[npt.NDArray[np.float32]] = []
        vectors: list[npt.NDArray[np.float32]] = []
        for start in range(0, len(texts), batch_size):
            batch = self.encode(texts[start : start + batch_size])
            out = self.model(**batch, output_hidden_states=True)
            probs.append(torch.softmax(out.logits.float(), dim=-1).cpu().numpy())
            mask = batch["attention_mask"].unsqueeze(-1).float()
            hidden = out.hidden_states[-1].float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            vectors.append(pooled.cpu().numpy())
        if not probs:
            width = int(self.model.config.hidden_size)
            return (
                np.zeros((0, len(self.labels)), dtype=np.float32),
                np.zeros((0, width), dtype=np.float32),
            )
        return (
            np.concatenate(probs).astype(np.float32),
            np.concatenate(vectors).astype(np.float32),
        )

    def predict(self, texts: Sequence[str]) -> list[str]:
        probs = self.predict_proba(texts)
        return [self.labels[i] for i in probs.argmax(axis=1)]

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        meta = {"labels": self.labels, "max_length": self.max_length, **self.meta}
        (path / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> EncoderClassifier:
        meta_path = path / META_FILE
        if not meta_path.exists():
            raise StudentError(f"{path} is not a saved Anvil student (no {META_FILE})")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        try:
            labels = list(meta.pop("labels"))
            max_length = int(meta.pop("max_length"))
            model = AutoModelForSequenceClassification.from_pretrained(path)
            tokenizer = AutoTokenizer.from_pretrained(path)
        except (KeyError, OSError, ValueError) as exc:
            raise StudentError(
                f"{path} is not a complete Anvil student ({exc}); train one with `anvil train`"
            ) from exc
        return cls(model, tokenizer, labels, max_length, extra_meta=meta)
