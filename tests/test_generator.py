"""The LoRA generator: adapter-only training, greedy decoding, save/load, merge and resume."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from anvil.chat import IGNORE
from anvil.errors import AlignmentError, StoreError, StudentError
from anvil.outputs import encode_output
from anvil.spec import StudentSpec
from anvil.students.generator import ADAPTER_DIR, BASE_DIR, META_FILE, LoraGenerator
from anvil.train import generator as train_gen
from anvil.train.generator import collate, mean_loss, validation_score
from anvil.train.loop import LAST

OUTPUTS = ["sql"]
SYSTEM = "SQL."
COLUMNS = ["name", "age", "city", "zip", "id", "email"]
TABLES = ["users", "orders", "items", "shops", "tags", "logs", "notes", "teams", "pets", "cars"]
SPEC = StudentSpec(
    model="tiny", max_length=96, max_new_tokens=32, epochs=4, batch_size=8,
    learning_rate=1e-2, validation_fraction=0.0, warmup_ratio=0.0, seed=0,
)  # fmt: skip


class SimulatedCrashError(Exception):
    pass


def _toy_set(n: int = 48) -> tuple[list[str], list[str]]:
    """`n` templated questions ("show c of t") and their stored answers ("SELECT c FROM t")."""
    texts: list[str] = []
    answers: list[str] = []
    for table in TABLES:
        for column in COLUMNS:
            texts.append(f"show {column} of {table}")
            answers.append(encode_output({"sql": f"SELECT {column} FROM {table}"}, OUTPUTS))
    return texts[:n], answers[:n]


def _student(spec: StudentSpec = SPEC) -> LoraGenerator:
    return LoraGenerator.build(spec, OUTPUTS, SYSTEM)


@pytest.fixture(scope="module")
def trained() -> LoraGenerator:
    student = _student()
    texts, answers = _toy_set()
    student.fit(texts, answers, SPEC.model_copy(update={"epochs": 25}))
    return student


@pytest.fixture(scope="module")
def saved(trained: LoraGenerator, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("generator") / "student"
    trained.save(path)
    return path


def _logits(model: Any, ids: list[int]) -> torch.Tensor:  # Any: a transformers/peft model
    with torch.no_grad():
        out: torch.Tensor = model(input_ids=torch.tensor([ids])).logits[0]
    return out


def _manual_greedy(student: LoraGenerator, model: Any, text: str) -> str:  # Any: as above
    """Argmax token by token on the unpadded prompt, until a stop id or max_new_tokens."""
    model.eval()
    ids = student.prompts([text])[0]
    new: list[int] = []
    for _ in range(student.max_new_tokens):
        token = int(_logits(model, ids + new)[-1].argmax())
        if token in student.stop_ids:
            break
        new.append(token)
    out: str = student.tokenizer.decode(new)
    return out


# Prompts of different lengths, so a batch needs left padding.
PROBES = ["show id of cars", "show email of notes", "show name of users please", "x"]


def _lora_state(student: LoraGenerator) -> dict[str, torch.Tensor]:
    return {n: p.detach().clone() for n, p in student.model.named_parameters() if p.requires_grad}


# --- building ---------------------------------------------------------------------------


def test_only_the_adapter_trains() -> None:
    student = _student()
    named = list(student.model.named_parameters())
    trainable = [n for n, p in named if p.requires_grad]
    assert trainable
    assert all("lora_" in n for n in trainable)
    assert all(not p.requires_grad for n, p in named if "lora_" not in n)
    assert any("lora_" in n for n, _ in named)
    assert student.trainable_parameters == sum(p.numel() for _, p in named if p.requires_grad)
    assert student.trainable_parameters < sum(p.numel() for _, p in named) / 2


def test_an_untrained_adapter_answers_as_the_base_model() -> None:
    student = _student()
    student.model.eval()
    lora_b = [p for n, p in student.model.named_parameters() if "lora_B" in n]
    assert lora_b
    assert all(torch.count_nonzero(p) == 0 for p in lora_b)
    ids = student.prompts(["show id of cars"])[0]
    with_adapter = _logits(student.model, ids)
    with student.model.disable_adapter():
        base = _logits(student.model, ids)
    assert torch.equal(with_adapter, base)


def test_generation_config_is_greedy_and_stops_at_the_end_of_turn() -> None:
    student = _student()
    assert student.end_id == student.tokenizer.convert_tokens_to_ids("<|im_end|>")
    for config in (
        student.model.generation_config,
        student.model.get_base_model().generation_config,
    ):
        assert config.do_sample is False
        assert config.num_beams == 1
        assert config.repetition_penalty == 1.0
        assert config.max_new_tokens == SPEC.max_new_tokens
        stop = config.eos_token_id
        assert student.end_id in (stop if isinstance(stop, list) else [stop])
        assert config.pad_token_id == student.tokenizer.pad_token_id
    assert student.end_id in student.stop_ids


def test_the_base_models_own_stop_tokens_stop_decoding_and_survive_a_save(
    trained: LoraGenerator, tmp_path: Path
) -> None:
    # Qwen2.5-Instruct's tokenizer names only <|im_end|> as eos while its generation config
    # also stops on <|endoftext|>; the base model's ids are kept in the student's meta.
    pad = int(trained.tokenizer.pad_token_id)
    assert pad not in trained.stop_ids
    student = LoraGenerator(
        trained.model, trained.tokenizer, OUTPUTS, SYSTEM, SPEC.max_length, SPEC.max_new_tokens,
        extra_meta={"student_model": "tiny"}, extra_stop_ids=[pad],
    )  # fmt: skip
    assert pad in student.stop_ids
    assert student.end_id in student.stop_ids
    config = student.model.generation_config
    assert pad in config.eos_token_id
    assert student.end_id in config.eos_token_id
    student.save(tmp_path / "s")
    assert LoraGenerator.load(tmp_path / "s").stop_ids == student.stop_ids
    student.model.generation_config = trained.model.generation_config  # restore the fixture
    LoraGenerator._set_greedy(trained, trained.model)


def test_build_reads_the_base_models_generation_config_stops() -> None:
    student = _student()
    assert student.meta["base_stop_ids"] == [student.end_id]  # the tiny config's eos


def test_a_tokenizer_without_a_template_or_pad_is_refused() -> None:
    student = _student()
    student.tokenizer.chat_template = None
    with pytest.raises(StudentError, match="no chat template"):
        LoraGenerator(student.model, student.tokenizer, OUTPUTS, SYSTEM, 64, 8)


# --- decoding ---------------------------------------------------------------------------


def test_batched_generate_equals_a_manual_greedy_loop(trained: LoraGenerator) -> None:
    lengths = {len(p) for p in trained.prompts(PROBES)}
    assert len(lengths) == len(PROBES)
    batched = trained.generate(PROBES, batch_size=8)
    assert len(set(batched)) > 1, batched  # a model that says one thing would prove nothing
    assert batched == [_manual_greedy(trained, trained.model, t) for t in PROBES]
    assert trained.generate(PROBES, batch_size=1) == batched
    assert trained.predict(PROBES) == [{"sql": c.strip()} for c in batched]


# --- training ---------------------------------------------------------------------------


def test_fit_lowers_the_loss_and_trains_only_the_adapter() -> None:
    student = _student()
    frozen = {
        n: p.detach().clone() for n, p in student.model.named_parameters() if not p.requires_grad
    }
    texts, answers = _toy_set()
    result = student.fit(texts, answers, SPEC)
    assert result.epoch_losses[-1] < result.epoch_losses[0]
    assert result.examples == len(texts)
    assert result.validation_examples == 0
    assert result.best_epoch is None
    assert student.chat is not None
    assert student.chat.unusable == 0
    for n, p in student.model.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    assert any(torch.count_nonzero(p) for n, p in _lora_state(student).items() if "lora_B" in n)


def test_fit_with_a_validation_slice_keeps_the_best_epoch(tmp_path: Path) -> None:
    texts, answers = _toy_set(60)
    spec = SPEC.model_copy(update={"validation_fraction": 0.3, "epochs": 3})
    student = _student(spec)
    result = student.fit(texts, answers, spec, metrics_path=tmp_path / "m.jsonl")
    assert result.validation_examples >= 10
    assert result.examples + result.validation_examples == len(texts)
    assert result.best_epoch is not None
    assert result.best_metric is not None
    assert (tmp_path / "m.jsonl").exists()


def test_collate_right_pads_and_masks() -> None:
    student = _student()
    built, _ = student.examples(["a", "a longer one"], ["X", "Y"])
    examples = [e for e in built if e is not None]
    batch = collate(examples, 999, torch.device("cpu"))
    short, long_ = (len(e.input_ids) for e in examples)
    assert batch["input_ids"].shape == (2, long_)
    assert batch["attention_mask"][0].tolist() == [1] * short + [0] * (long_ - short)
    assert batch["input_ids"][0, short:].tolist() == [999] * (long_ - short)
    assert batch["labels"][0, short:].tolist() == [IGNORE] * (long_ - short)
    loss = mean_loss(student.model, examples, int(student.tokenizer.pad_token_id))
    assert loss > 0


def test_a_label_that_is_not_an_answer_is_a_store_error() -> None:
    texts, answers = _toy_set(12)
    for bad in ("SELECT 1", '{"query": "SELECT 1"}', '{"sql": 1}', '["SELECT 1"]'):
        with pytest.raises(StoreError, match="not an answer to this generate task"):
            _student().fit(texts, [*answers[:-1], bad], SPEC)


def _with_long(n_long: int, n: int = 40) -> tuple[list[str], list[str]]:
    texts, answers = _toy_set(n)
    for i in range(n_long):
        texts[i] = f"show {COLUMNS[i % 6]} of {TABLES[0]} " + "and more words " * 6
    return texts, answers


def test_too_many_too_long_examples_refuse_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called = False

    def no_fit(*args: object, **kwargs: object) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(train_gen, "fit", no_fit)
    texts, answers = _with_long(3)  # 7.5%
    with pytest.raises(AlignmentError, match=r"3 of 40 .*too_long 3"):
        _student().fit(texts, answers, SPEC, metrics_path=tmp_path / "m.jsonl")
    assert not called
    assert not (tmp_path / "m.jsonl").exists()


def test_a_few_too_long_examples_are_dropped_and_reported() -> None:
    texts, answers = _with_long(2)  # 5%: at the limit, still trains
    student = _student()
    result = student.fit(texts, answers, SPEC.model_copy(update={"epochs": 1}))
    assert result.examples == 38
    assert student.chat is not None
    assert student.chat.dropped == {
        "too_long": 2, "answer_too_long": 0, "not_prefix": 0, "not_roundtrip": 0
    }  # fmt: skip
    assert student.chat.dropped_ids == ["0", "1"]
    named = _student()
    named.fit(texts, answers, SPEC.model_copy(update={"epochs": 1}),
              input_ids=[f"id{i}" for i in range(40)])  # fmt: skip
    assert named.chat is not None
    assert named.chat.dropped_ids == ["id0", "id1"]


def test_mismatched_or_empty_inputs_are_student_errors() -> None:
    texts, answers = _toy_set(10)
    with pytest.raises(StudentError, match="10 texts but 9 answers"):
        _student().fit(texts, answers[:9], SPEC)
    with pytest.raises(StudentError, match="no training examples"):
        _student().fit([], [], SPEC)


@given(n=st.integers(1, 500), hits=st.integers(0, 499), a=st.floats(0, 1e6), b=st.floats(0, 1e6))
def test_one_more_exact_match_beats_any_loss(n: int, hits: int, a: float, b: float) -> None:
    hits = min(hits, n - 1)
    assert validation_score((hits + 1) / n, a, n) > validation_score(hits / n, b, n)


@given(n=st.integers(1, 500), hits=st.integers(0, 500), a=st.floats(0, 1e3), b=st.floats(0, 1e3))
def test_equal_exact_match_prefers_the_lower_loss(n: int, hits: int, a: float, b: float) -> None:
    em = min(hits, n) / n
    if a < b:
        assert validation_score(em, a, n) >= validation_score(em, b, n)
    if a < b - 1e-3:
        assert validation_score(em, a, n) > validation_score(em, b, n)


@given(n=st.integers(1, 500), hits=st.integers(0, 500), loss=st.floats(0, 1e6))
def test_the_loss_term_is_under_half_an_answer(n: int, hits: int, loss: float) -> None:
    em = min(hits, n) / n
    assert em <= validation_score(em, loss, n) <= em + 0.5 / n


# --- save, load, merge ------------------------------------------------------------------


def test_save_and_load_give_the_same_answers(trained: LoraGenerator, saved: Path) -> None:
    assert (saved / ADAPTER_DIR / "adapter_model.safetensors").is_file()
    assert (saved / BASE_DIR).is_dir()
    meta = json.loads((saved / META_FILE).read_text(encoding="utf-8"))
    assert meta["outputs"] == OUTPUTS
    assert meta["system"] == SYSTEM
    assert (meta["max_length"], meta["max_new_tokens"]) == (SPEC.max_length, SPEC.max_new_tokens)
    assert meta["student_model"] == "tiny"
    loaded = LoraGenerator.load(saved)
    assert loaded.generate(PROBES) == trained.generate(PROBES)
    assert loaded.stop_ids == trained.stop_ids
    ids = trained.prompts(PROBES[:1])[0]
    trained.model.eval()
    assert torch.equal(_logits(loaded.model, ids), _logits(trained.model, ids))


def test_load_without_meta_is_a_student_error(tmp_path: Path, saved: Path) -> None:
    with pytest.raises(StudentError, match="not a saved Anvil generator"):
        LoraGenerator.load(tmp_path)
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / META_FILE).write_text((saved / META_FILE).read_text(encoding="utf-8"))
    with pytest.raises(StudentError, match="not a complete Anvil generator"):
        LoraGenerator.load(broken)


def test_merged_model_matches_the_adapter(saved: Path) -> None:
    reference = LoraGenerator.load(saved)
    student = LoraGenerator.load(saved)
    merged = student.merged()
    assert not any("lora_" in n for n, _ in merged.named_parameters())
    merged.eval()
    reference.model.eval()
    for ids in reference.prompts(PROBES):
        assert torch.allclose(_logits(merged, ids), _logits(reference.model, ids), atol=1e-4)
    assert reference.generate(PROBES, model=merged) == reference.generate(PROBES)
    assert merged.generation_config.do_sample is False
    assert merged.generation_config.repetition_penalty == 1.0


# --- checkpoints ------------------------------------------------------------------------


def _crash_at(monkeypatch: pytest.MonkeyPatch, call: int) -> None:
    original = train_gen.collate
    calls = 0

    def failing(
        examples: Sequence[Any], pad_id: int, device: torch.device
    ) -> dict[str, torch.Tensor]:  # Any: anvil.chat.Example
        nonlocal calls
        calls += 1
        if calls == call:
            raise SimulatedCrashError
        return original(examples, pad_id, device)

    monkeypatch.setattr(train_gen, "collate", failing)


def test_checkpoints_hold_the_adapter_and_resume_bit_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts, answers = _toy_set()
    spec = SPEC.model_copy(update={"epochs": 3, "grad_accum": 2, "batch_size": 4})
    ref = _student(spec)
    ref_result = ref.fit(texts, answers, spec)

    ckpt = tmp_path / "ckpt"
    _crash_at(monkeypatch, 16)  # epoch 2 (12 micro-batches an epoch), after the first checkpoint
    with pytest.raises(SimulatedCrashError):
        _student(spec).fit(texts, answers, spec, checkpoint_dir=ckpt)
    monkeypatch.undo()
    state = torch.load(ckpt / LAST, weights_only=True)
    student = _student(spec)
    assert set(state["model"]) == set(_lora_state(student))
    assert all("lora_" in k for k in state["model"])

    result = student.fit(texts, answers, spec, checkpoint_dir=ckpt, resume=True)
    assert result.resumed_at_step == 6
    assert result.epoch_losses == ref_result.epoch_losses
    ref_state, got = _lora_state(ref), _lora_state(student)
    assert ref_state.keys() == got.keys()
    for name, value in ref_state.items():
        assert torch.equal(value, got[name]), name
    assert not ckpt.exists()


def test_resume_with_another_adapter_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts, answers = _toy_set()
    ckpt = tmp_path / "ckpt"
    _crash_at(monkeypatch, 8)  # epoch 2 (6 micro-batches an epoch)
    with pytest.raises(SimulatedCrashError):
        _student().fit(texts, answers, SPEC, checkpoint_dir=ckpt)
    monkeypatch.undo()
    # The adapter settings are part of the data fingerprint.
    narrower = SPEC.model_copy(update={"lora": SPEC.lora.model_copy(update={"r": 4})})
    with pytest.raises(StudentError, match="different training config"):
        _student(narrower).fit(texts, answers, narrower, checkpoint_dir=ckpt, resume=True)
