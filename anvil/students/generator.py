"""Generation student: a decoder with a LoRA adapter, trained on the teacher's answers.

The default is `Qwen/Qwen2.5-0.5B-Instruct`: an instruction-tuned decoder small enough to
fine-tune on a laptop CPU (about 20 s per step of 8 sequences of 160 tokens, measured) and to
serve without a GPU. Its weights stay frozen; a rank-r adapter on every linear layer of every
block is what trains (8.8M parameters, 1.7% of the model, at r = 16), so a checkpoint and a
saved student hold the adapter alone, and `merged` folds it back into a plain model for
export.

`tiny` builds a two-layer random-init Qwen2 over a byte-level vocabulary with the ChatML
template Qwen uses, so tests run offline in seconds through the same code path: the same
template checks, the same masking, the same PEFT wrapper and the same decoding.

Decoding is greedy with every sampling setting written out: Qwen's shipped
`generation_config.json` samples (temperature 0.7, top-p 0.8) with a repetition penalty of
1.1, and a penalty on repeated tokens is exactly wrong for SQL, which must repeat the column
names it was given. The student replaces the model's generation config with its own, so no
default can leak in.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)

from anvil.chat import ChatReport, Example, build_examples, end_of_turn_id, parse_completion
from anvil.errors import StudentError
from anvil.spec import LoraSpec, StudentSpec
from anvil.students.encoder import TINY
from anvil.train.loop import TrainResult

META_FILE = "anvil_generator.json"
ADAPTER_DIR = "adapter"
BASE_DIR = "base"
"""Where a `tiny` student keeps its random-init base model: it cannot be downloaded again."""

CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
)
"""Qwen2.5's ChatML layout without its default system prompt (Anvil always sends one)."""


def build_tiny_chat_tokenizer() -> PreTrainedTokenizerFast:
    """A byte-level tokenizer with no merges: every byte is one token, so any text encodes and
    decodes back exactly, and the vocabulary is fixed (no corpus, no trainer, nothing that
    can differ between processes). ChatML's three special tokens close the vocabulary."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["<|endoftext|>", "<|im_start|>", "<|im_end|>"])
    fast = PreTrainedTokenizerFast(  # type: ignore[no-untyped-call]
        tokenizer_object=tok,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        additional_special_tokens=["<|im_start|>"],
    )
    fast.chat_template = CHATML
    return fast


def build_tiny_decoder(tokenizer: PreTrainedTokenizerFast, max_positions: int) -> Qwen2ForCausalLM:
    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max(512, max_positions),
        tie_word_embeddings=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.pad_token_id,
    )
    return Qwen2ForCausalLM(config)  # type: ignore[no-untyped-call]


def _wrap(base: Any, lora: LoraSpec) -> Any:  # Any: transformers and peft models
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        task_type="CAUSAL_LM",
    )
    try:
        return get_peft_model(base, config)
    except ValueError as exc:
        raise StudentError(
            f"could not attach LoRA to {type(base).__name__} ({exc}); set"
            " student.lora.target_modules to linear layer names this model has"
        ) from exc


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class LoraGenerator:
    def __init__(
        self,
        model: Any,  # Any: a peft PeftModelForCausalLM
        tokenizer: Any,  # Any: a transformers tokenizer with a chat template
        outputs: list[str],
        system: str,
        max_length: int,
        max_new_tokens: int,
        extra_meta: dict[str, object] | None = None,
        extra_stop_ids: Sequence[int] = (),
    ) -> None:
        if not getattr(tokenizer, "chat_template", None):
            raise StudentError(
                "the student's tokenizer has no chat template; use an instruction-tuned model"
                " (for example Qwen/Qwen2.5-0.5B-Instruct) or student.model: tiny"
            )
        if tokenizer.pad_token_id is None:
            raise StudentError("the student's tokenizer has no pad token to batch with")
        self.model = model
        self.tokenizer = tokenizer
        self.outputs = outputs
        self.system = system
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.meta: dict[str, object] = dict(extra_meta or {})
        self.end_id = end_of_turn_id(tokenizer)
        # The base model's own end-of-text token stops decoding too: Qwen2.5-Instruct's
        # tokenizer names only <|im_end|> as eos, while its generation config also stops on
        # <|endoftext|>, which the untrained base model (the zero-shot baseline) can emit.
        self.stop_ids = sorted({self.end_id, *_ids_of(tokenizer.eos_token_id), *extra_stop_ids})
        self.meta["base_stop_ids"] = [int(x) for x in extra_stop_ids]
        self.chat: ChatReport | None = None
        """The alignment report of the last `fit` (what the template and max_length cost)."""
        self._set_greedy(model)

    @classmethod
    def build(cls, spec: StudentSpec, outputs: list[str], system: str) -> LoraGenerator:
        """A fresh student: the base model (or a tiny random one) with an untrained adapter.
        LoRA's B matrices start at zero, so before training it answers as the base model."""
        torch.manual_seed(spec.seed)
        if spec.model == TINY:
            # Any: the tiny and pretrained branches return different transformers classes.
            tokenizer: Any = build_tiny_chat_tokenizer()
            base: Any = build_tiny_decoder(tokenizer, spec.max_length + spec.max_new_tokens)
        else:
            try:
                tokenizer = AutoTokenizer.from_pretrained(spec.model)
                # fp32 explicitly: Qwen's config says bf16, which is slow and lossy on CPU;
                # on a GPU the loop's autocast picks the compute dtype.
                base = AutoModelForCausalLM.from_pretrained(spec.model, dtype=torch.float32)
            except OSError as exc:
                raise StudentError(
                    f"could not load student {spec.model!r}: {exc}; check the model id or use"
                    " student.model: tiny"
                ) from exc
        base.to(_device())
        base_stops = _ids_of(getattr(base.generation_config, "eos_token_id", None))
        model = _wrap(base, spec.lora)
        return cls(
            model,
            tokenizer,
            outputs,
            system,
            spec.max_length,
            spec.max_new_tokens,
            extra_meta={"student_model": spec.model, "lora": spec.lora.model_dump()},
            extra_stop_ids=base_stops,
        )

    @property
    def device(self) -> torch.device:
        device: torch.device = next(self.model.parameters()).device
        return device

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def adapter_parameters(self) -> int:
        """Parameters of the LoRA adapter, trainable or not (a loaded adapter is frozen)."""
        return sum(p.numel() for n, p in self.model.named_parameters() if "lora_" in n)

    def _set_greedy(self, model: Any) -> None:  # Any: a transformers or peft model
        config = GenerationConfig(  # type: ignore[no-untyped-call]
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.0,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.stop_ids,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        model.generation_config = config
        if hasattr(model, "get_base_model"):  # a PEFT wrapper generates through its base
            model.get_base_model().generation_config = config

    def examples(
        self,
        texts: Sequence[str],
        targets: Sequence[str],
        input_ids: Sequence[str] | None = None,
    ) -> tuple[list[Example | None], ChatReport]:
        """Training sequences for (input, rendered target) pairs; see `anvil.chat`."""
        return build_examples(
            self.tokenizer,
            self.system,
            texts,
            targets,
            self.max_length,
            self.end_id,
            input_ids,
            max_answer=self.max_new_tokens,
        )

    def prompts(self, texts: Sequence[str]) -> list[list[int]]:
        from anvil.chat import prompt_ids

        return [prompt_ids(self.tokenizer, self.system, text) for text in texts]

    @torch.no_grad()
    def generate(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
        model: Any = None,  # Any: another model sharing this tokenizer (the merged export)
    ) -> list[str]:
        """Greedy completions, one per text, decoded without the end-of-turn token.

        Prompts are batched in order of length, so a batch pads to similar lengths, and
        padded on the left, so every row's next token follows its own last prompt token.
        """
        net = self.model if model is None else model
        if model is not None:
            self._set_greedy(model)
        net.eval()
        prompts = self.prompts(texts)
        order = sorted(range(len(prompts)), key=lambda i: (len(prompts[i]), i))
        out: list[str] = [""] * len(prompts)
        pad = int(self.tokenizer.pad_token_id)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            width = max(len(prompts[i]) for i in chunk)
            ids = torch.full((len(chunk), width), pad, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            for row, i in enumerate(chunk):
                p = prompts[i]
                ids[row, width - len(p) :] = torch.tensor(p, dtype=torch.long)
                mask[row, width - len(p) :] = 1
            generated = net.generate(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                generation_config=net.generation_config,
            )
            for row, i in enumerate(chunk):
                tokens = generated[row, width:].tolist()
                cut = next((k for k, t in enumerate(tokens) if t in self.stop_ids), len(tokens))
                out[i] = self.tokenizer.decode(tokens[:cut], skip_special_tokens=False)
        return out

    def predict(self, texts: Sequence[str], batch_size: int = 16) -> list[dict[str, str] | None]:
        """Each text's answer as fields, None where the completion is not a valid answer."""
        return [parse_completion(c, self.outputs) for c in self.generate(texts, batch_size)]

    def fit(
        self,
        texts: Sequence[str],
        answers: Sequence[str],
        spec: StudentSpec,
        metrics_path: Path | None = None,
        *,
        input_ids: Sequence[str] | None = None,
        checkpoint_dir: Path | None = None,
        resume: bool = False,
        keep_checkpoint: bool = False,
    ) -> TrainResult:
        """Train the adapter on (input, stored teacher answer) pairs; see
        `anvil.train.generator.train_generator`. The alignment report is kept in `self.chat`."""
        # Imported here: the training module imports this one for its type hints.
        from anvil.train.generator import train_generator

        result, self.chat = train_generator(
            self,
            texts,
            answers,
            spec,
            metrics_path,
            input_ids=input_ids,
            checkpoint_dir=checkpoint_dir,
            resume=resume,
            keep_checkpoint=keep_checkpoint,
        )
        return result

    def merged(self) -> Any:  # Any: a transformers PreTrainedModel
        """The base model with the adapter folded into its weights (W + (alpha / r) B A), a
        plain transformers model that needs no PEFT to load. It consumes the adapter: this
        student must not be used afterwards, so load a second copy to compare against."""
        merged = self.model.merge_and_unload()
        self._set_greedy(merged)
        return merged

    def save(self, path: Path) -> None:
        if "merged_from" in self.meta:
            raise StudentError(
                "a merged export has no adapter to save as a student; train one with `anvil train`"
            )
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path / ADAPTER_DIR)
        self.tokenizer.save_pretrained(path)
        if self.meta.get("student_model", TINY) == TINY:
            # A random-init base cannot be fetched again, so it is saved with the adapter.
            # PEFT wraps the base model's linear layers in place, so the base is saved from a
            # copy with the adapter layers removed (not merged): saving `get_base_model()`
            # would write `base_layer` keys that a plain model loads as fresh random weights.
            copy.deepcopy(self.model).unload().save_pretrained(path / BASE_DIR)
        meta = {
            "outputs": self.outputs,
            "system": self.system,
            "max_length": self.max_length,
            "max_new_tokens": self.max_new_tokens,
            **self.meta,
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def load_meta(path: Path) -> dict[str, object]:
        """A saved generator's metadata, without loading any weights."""
        meta_path = path / META_FILE
        if not meta_path.exists():
            raise StudentError(f"{path} is not a saved Anvil generator (no {META_FILE})")
        meta: dict[str, object] = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta

    @classmethod
    def load(cls, path: Path) -> LoraGenerator:
        from peft import PeftModel

        meta: dict[str, Any] = dict(cls.load_meta(path))  # Any: JSON values, checked below
        try:
            outputs = [str(x) for x in meta.pop("outputs")]
            system = str(meta.pop("system"))
            max_length = int(meta.pop("max_length"))
            max_new_tokens = int(meta.pop("max_new_tokens"))
            base_stops = [int(x) for x in meta.pop("base_stop_ids", [])]
            base_id = str(meta.get("student_model", ""))
            source = str(path / BASE_DIR) if (path / BASE_DIR).exists() else base_id
            tokenizer = AutoTokenizer.from_pretrained(path)
            # Any: transformers types `to` through a decorator mypy cannot follow.
            base: Any = AutoModelForCausalLM.from_pretrained(source, dtype=torch.float32)
            base.to(_device())
            model = PeftModel.from_pretrained(base, path / ADAPTER_DIR)
        except (KeyError, OSError, ValueError, TypeError) as exc:
            raise StudentError(
                f"{path} is not a complete Anvil generator ({exc}); train one with `anvil train`"
            ) from exc
        return cls(model, tokenizer, outputs, system, max_length, max_new_tokens, meta, base_stops)


def _ids_of(value: object) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list | tuple):
        return [int(v) for v in value]
    raise StudentError(f"unexpected eos_token_id {value!r}")
