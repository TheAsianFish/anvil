"""The extraction path of `label -> train -> eval`: align, gate, train a tagger, score it.

Teacher answers are stored raw (the mentions as the teacher gave them), so alignment runs at
train and eval time and can be improved without paying for the labels again. Training aligns
every pool answer to its input and to the student's tokens, refuses to start when more than
`align.MAX_UNALIGNED_RATE` of the mentions are lost on the way, and saves the alignment report
with the student. Scoring aligns the held-out answers the same way (to the text only: the
reference is what the teacher meant, not what the student's tokenizer can express, so a
student that cannot represent a boundary is charged for it).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from anvil.align import AlignmentReport, align_mentions
from anvil.errors import StoreError
from anvil.evaluate.extraction import ExtractionAgreement, extraction_agreement
from anvil.log import get_logger
from anvil.spans import Span, decode_answer
from anvil.spec import TaskSpec
from anvil.students.tagger import EncoderTagger
from anvil.teacher.base import Item
from anvil.train.loop import TrainResult

log = get_logger(__name__)


def answer_spans(
    spec: TaskSpec, pairs: Sequence[tuple[Item, str]]
) -> tuple[list[list[Span]], AlignmentReport]:
    """The verified spans of each stored teacher answer, and the merged alignment report."""
    report = AlignmentReport()
    out: list[list[Span]] = []
    for item, label in pairs:
        try:
            mentions = decode_answer(label)
        except ValueError as exc:
            raise StoreError(
                f"label of {item.id} is not an extraction answer ({exc}); was it made for a"
                " classify version of this task?"
            ) from None
        aligned = align_mentions(item.text, mentions, spec.label_names, item.id)
        report.merge(aligned.report)
        out.append(aligned.spans)
    return out, report


def fit_tagger(
    spec: TaskSpec,
    teacher_model: str,
    pairs: Sequence[tuple[Item, str]],
    corpus: Sequence[str],
    metrics_path: Path | None = None,
    *,
    checkpoint_dir: Path | None = None,
    resume: bool = False,
) -> tuple[EncoderTagger, TrainResult, AlignmentReport]:
    """Align the pool answers, check the unaligned rate, and train a fresh tagger on them.

    The gate runs before the first training step, on both alignment stages (text and
    tokens), so a broken prompt fails as soon as the student's tokenizer is loaded rather
    than after an epoch.
    """
    texts = [it.text for it, _ in pairs]
    spans, report = answer_spans(spec, pairs)
    student = EncoderTagger.build(spec.student, spec.label_names, corpus)
    _, token_report, _ = student.targets(texts, spans, input_ids=[it.id for it, _ in pairs])
    report.merge(token_report)
    log.info("aligned teacher answers", extra={"inputs": len(pairs), **report.as_dict()})
    report.check()
    result = student.fit(
        texts,
        spans,
        spec.student,
        metrics_path,
        checkpoint_dir=checkpoint_dir,
        resume=resume,
        keep_checkpoint=checkpoint_dir is not None,
    )
    student.meta.update(
        {
            "task": spec.name,
            "kind": spec.kind,
            "teacher_model": teacher_model,
            "prompt_hash": spec.prompt_hash,
            "splits": spec.splits.model_dump(),
            "student_model": spec.student.model,
            "train_examples": len(pairs),
            "alignment": report.as_dict(),
            "train": asdict(result),
        }
    )
    return student, result, report


@dataclass(frozen=True)
class TaggerScore:
    """A tagger scored against held-out teacher answers, with everything an eval reports."""

    agreement: ExtractionAgreement
    teacher: list[list[Span]]
    predicted: list[list[Span]]
    confidence: list[float]
    alignment: AlignmentReport


def score_tagger(
    spec: TaskSpec,
    student: EncoderTagger,
    pairs: Sequence[tuple[Item, str]],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> TaggerScore:
    """Agreement of `student` with the teacher's answers in `pairs` (held-out inputs)."""
    texts = [it.text for it, _ in pairs]
    teacher, report = answer_spans(spec, pairs)
    predicted, confidence = student.predict_with_confidence(texts)
    agreement = extraction_agreement(
        texts, teacher, predicted, spec.label_names, n_resamples=n_resamples, seed=seed
    )
    return TaggerScore(agreement, teacher, predicted, list(confidence), report)
