"""The generation path of `label -> train -> eval`: align to the chat template, train a LoRA
generator, score its greedy answers against the teacher's.

Teacher answers are stored in their canonical form (`anvil.outputs`), so how the student is
taught to write them (`chat.render_target`) can change without paying for labels again.
The generic score here is exact match per field and over every field; a task with an
objective notion of "the same answer", such as running two SQL queries and comparing their
results, scores that in its eval (`eval/generate.py`), from the same predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from anvil.chat import ChatReport
from anvil.errors import StoreError
from anvil.evaluate.generation import GenerationAgreement, generation_agreement
from anvil.log import get_logger
from anvil.outputs import decode_output
from anvil.spec import TaskSpec
from anvil.students.generator import LoraGenerator
from anvil.teacher.base import Item
from anvil.train.loop import TrainResult

log = get_logger(__name__)


def student_system_prompt(spec: TaskSpec) -> str:
    """The system message the student is trained and served with: the task's instructions.

    Not the teacher's full prompt: that one asks for a JSON object of every field, while a
    one-field student writes the bare value (`chat.render_target`), and a prompt that
    contradicts the training targets is noise the adapter would have to learn around."""
    return spec.instructions.strip()


def answers(spec: TaskSpec, pairs: Sequence[tuple[Item, str]]) -> list[dict[str, str]]:
    """The fields of each stored teacher answer."""
    out: list[dict[str, str]] = []
    for item, label in pairs:
        try:
            out.append(decode_output(label, spec.output_names))
        except ValueError as exc:
            raise StoreError(
                f"label of {item.id} is not an answer to {spec.name} ({exc}); was it made for"
                " another version of this task?"
            ) from None
    return out


def fit_generator(
    spec: TaskSpec,
    teacher_model: str,
    pairs: Sequence[tuple[Item, str]],
    metrics_path: Path | None = None,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> tuple[LoraGenerator, TrainResult, ChatReport]:
    """Build a fresh generator and train its adapter on (input, teacher answer) pairs.

    The chat alignment gate runs before the first training step. With `checkpoint_dir`, the
    checkpoint is kept after training; the caller removes it once the student is saved."""
    student = LoraGenerator.build(spec.student, spec.output_names, student_system_prompt(spec))
    result = student.fit(
        [it.text for it, _ in pairs],
        [lab for _, lab in pairs],
        spec.student,
        metrics_path,
        input_ids=[it.id for it, _ in pairs],
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        keep_checkpoint=checkpoint_dir is not None,
    )
    assert student.chat is not None  # set by fit
    student.meta.update(
        {
            "task": spec.name,
            "kind": spec.kind,
            "teacher_model": teacher_model,
            "prompt_hash": spec.prompt_hash,
            "splits": spec.splits.model_dump(),
            "student_model": spec.student.model,
            "student_spec": spec.student.model_dump(),
            "train_examples": len(pairs),
            "trainable_parameters": student.trainable_parameters,
            "chat": student.chat.as_dict(),
            "train": asdict(result),
        }
    )
    return student, result, student.chat


@dataclass(frozen=True)
class GeneratorScore:
    """A generator scored against held-out teacher answers, with what an eval reports."""

    agreement: GenerationAgreement
    teacher: list[dict[str, str]]
    predicted: list[dict[str, str] | None]
    completions: list[str]


def score_generator(
    spec: TaskSpec,
    student: LoraGenerator,
    pairs: Sequence[tuple[Item, str]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    batch_size: int = 16,
) -> GeneratorScore:
    """Exact-match agreement of `student`'s greedy answers with the teacher's on `pairs`."""
    from anvil.chat import parse_completion

    texts = [it.text for it, _ in pairs]
    teacher = answers(spec, pairs)
    completions = student.generate(texts, batch_size)
    predicted = [parse_completion(c, spec.output_names) for c in completions]
    agreement = generation_agreement(
        teacher,
        predicted,
        spec.output_names,
        lengths=[len(t.split()) for t in texts],
        n_resamples=n_resamples,
        seed=seed,
    )
    return GeneratorScore(agreement, teacher, predicted, completions)
