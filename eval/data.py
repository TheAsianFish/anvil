"""Dataset loaders for the evals: a pinned Hugging Face revision, a seeded sample, gold labels.

Each dataset is read at a fixed commit so a re-run sees the same rows, and the sample is a
seeded draw of distinct texts from the train split. Gold label names are normalised to the
spec's label alphabet (`Sci/Tech` -> `sci_tech`).
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from anvil.errors import ConfigError
from anvil.spans import Span, encode_answer, spans_to_mentions
from anvil.teacher.base import Item


def label_name(raw: str) -> str:
    """A dataset class name in the spec's label alphabet: lower case, runs of other
    characters folded to `_`."""
    return re.sub(r"[^a-z0-9_-]+", "_", raw.lower()).strip("_")


@dataclass(frozen=True)
class DatasetRef:
    repo: str
    revision: str
    split: str = "train"
    text_field: str = "text"
    label_field: str = "label"
    license: str = ""
    config: str | None = None
    """The dataset's configuration (`clinc/clinc_oos` has `small`, `plus` and `imbalanced`)."""


@dataclass(frozen=True)
class Sample:
    items: list[Item]
    gold: dict[str, str]
    """Input id -> gold label name."""
    source_rows: int

    @property
    def sha256(self) -> str:
        """Fingerprint of the sampled inputs and gold labels, recorded with every run."""
        blob = json.dumps(
            [[it.id, it.text, self.gold[it.id]] for it in self.items], ensure_ascii=False
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def load_rows(ref: DatasetRef) -> tuple[list[str], list[str]]:
    """All (text, gold label name) rows of the split, in dataset order."""
    from datasets import load_dataset

    ds = load_dataset(ref.repo, ref.config, revision=ref.revision, split=ref.split)
    names = ds.features[ref.label_field].names
    texts = [str(t) for t in ds[ref.text_field]]
    labels = [label_name(names[int(k)]) for k in ds[ref.label_field]]
    return texts, labels


def sample(name: str, texts: list[str], labels: list[str], n: int, seed: int) -> Sample:
    """`n` distinct, non-empty texts drawn without replacement in a seeded order.

    Ids are `<name>-<row index>`, so a sample is traceable to the source row.
    """
    order = np.random.default_rng(seed).permutation(len(texts))
    items: list[Item] = []
    gold: dict[str, str] = {}
    seen: set[str] = set()
    for i in order.tolist():
        text = texts[i].strip()
        if not text or text in seen:
            continue
        seen.add(text)
        item = Item(id=f"{name}-{i}", text=text)
        items.append(item)
        gold[item.id] = labels[i]
        if len(items) == n:
            break
    return Sample(items, gold, len(texts))


@dataclass(frozen=True)
class TaggedRef:
    """A word-tokenised BIO dataset stored as JSON lines (`{"tokens": [...], "tags": [...]}`
    with integer tags and a `label.json` tag map), read at a pinned revision. The T-NER
    copies on the Hugging Face Hub use this layout; reading the raw files directly avoids the
    loader scripts `datasets` 5 no longer runs."""

    repo: str
    revision: str
    filename: str
    label_map: str
    license: str = ""


def load_tagged_rows(ref: TaggedRef) -> list[tuple[list[str], list[str]]]:
    """Every (words, BIO tag names) row of the file, in file order."""
    from huggingface_hub import hf_hub_download

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(ref.repo, name, repo_type="dataset", revision=ref.revision))

    tag_ids = json.loads(fetch(ref.label_map).read_text(encoding="utf-8"))
    names = {int(v): str(k) for k, v in tag_ids.items()}
    rows: list[tuple[list[str], list[str]]] = []
    for line in fetch(ref.filename).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        rows.append(([str(w) for w in obj["tokens"]], [names[int(t)] for t in obj["tags"]]))
    return rows


def bio_to_spans(
    words: Sequence[str], tags: Sequence[str], row: object = None
) -> tuple[str, list[Span]]:
    """The words joined by single spaces and the gold spans as character offsets into it.

    Chunks follow the CoNLL convention (as `align.tags_to_spans` does for predictions): `B-x`
    opens a chunk, `I-x` continues a chunk of type `x` and otherwise opens one, `O` closes.
    Only BIO tags are accepted: an IO tag (`PER`) or a BIOES one (`E-PER`, `S-PER`) would be
    chunked wrongly without a word of warning, so it raises `ConfigError`, as does an empty
    word (the joined text would not split back into the same words). `row` names the source
    row in the message.
    """
    where = f"row {row}" if row is not None else "a row"
    if len(words) != len(tags):
        raise ConfigError(f"{where} has {len(words)} words but {len(tags)} tags")
    starts: list[int] = []
    pos = 0
    for i, w in enumerate(words):
        if not w.strip():
            raise ConfigError(f"{where} has an empty word at position {i}; fix the dataset")
        starts.append(pos)
        pos += len(w) + 1
    text = " ".join(words)
    spans: list[Span] = []
    open_type: str | None = None  # the type of the chunk the previous word belongs to
    for i, tag in enumerate(tags):
        prefix, dash, kind = tag.partition("-")
        if tag != "O" and not (dash and prefix in ("B", "I") and kind):
            raise ConfigError(
                f"{where} has tag {tag!r} at word {i}: only O, B-<type> and I-<type> are"
                " understood; convert the dataset to the BIO scheme"
            )
        end = starts[i] + len(words[i])
        if tag == "O":
            open_type = None
        elif prefix == "I" and open_type == kind:
            spans[-1] = Span(spans[-1].start, end, kind)
        else:
            spans.append(Span(starts[i], end, kind))
            open_type = kind
    return text, spans


def sample_tagged(
    name: str,
    rows: Sequence[tuple[list[str], list[str]]],
    n: int,
    seed: int,
    types: Sequence[str],
) -> Sample:
    """`n` distinct, non-empty sentences in a seeded order, like `sample`. The gold label of
    each is its spans as a stored extraction answer (`spans.encode_answer`), so the gold
    teacher replays them through the same path a Claude answer takes.

    Every gold type in the sample must be one of `types` (the spec's labels): alignment would
    otherwise drop the unknown ones as invalid, and the gold teacher, the student and the
    stress study would all quietly score a smaller task than the dataset's."""
    order = np.random.default_rng(seed).permutation(len(rows))
    items: list[Item] = []
    gold: dict[str, str] = {}
    seen: set[str] = set()
    unknown: set[str] = set()
    known = set(types)
    for i in order.tolist():
        words, tags = rows[i]
        if not words:
            continue
        text, spans = bio_to_spans(words, tags, row=i)
        if text in seen:
            continue
        seen.add(text)
        unknown.update(s.type for s in spans if s.type not in known)
        item = Item(id=f"{name}-{i}", text=text)
        items.append(item)
        gold[item.id] = encode_answer(spans_to_mentions(text, spans))
        if len(items) == n:
            break
    if unknown:
        raise ConfigError(
            f"the {name} gold answers use types the spec does not list:"
            f" {', '.join(sorted(unknown))}; add them to the spec's labels or fix the label map"
        )
    return Sample(items, gold, len(rows))


def write_inputs(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({"id": it.id, "text": it.text}, ensure_ascii=False) + "\n")


NEAR_DUPLICATE_JACCARD = 0.8


def _words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def near_duplicate_mask(
    texts: Sequence[str], pool: Sequence[str], threshold: float = NEAR_DUPLICATE_JACCARD
) -> list[bool]:
    """For each text, whether it shares at least `threshold` of its words (Jaccard over the
    sets of lower-case alphanumeric tokens) with some pool text."""
    pool_words = [w for w in (_words(t) for t in pool) if w]
    out: list[bool] = []
    for text in texts:
        words = _words(text)
        hit = False
        for other in pool_words if words else []:
            shared = len(words & other)
            if shared and shared / (len(words) + len(other) - shared) >= threshold:
                hit = True
                break
        out.append(hit)
    return out


def near_duplicates(
    heldout: Sequence[str], pool: Sequence[str], threshold: float = NEAR_DUPLICATE_JACCARD
) -> int:
    """How many held-out texts share at least `threshold` of their words with some pool text
    (`near_duplicate_mask`).

    The store's split hashes the exact text, so identical texts never straddle it, but a
    headline rewritten by a word or two can: this counts how many such near-copies the
    student may have seen during training, so a report can say so instead of claiming no
    leakage at all.
    """
    return sum(near_duplicate_mask(heldout, pool, threshold))
