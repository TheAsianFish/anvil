"""Dataset loaders for the evals: a pinned Hugging Face revision, a seeded sample, gold labels.

Each dataset is read at a fixed commit so a re-run sees the same rows, and the sample is a
seeded draw of distinct texts from the train split. Gold label names are normalised to the
spec's label alphabet (`Sci/Tech` -> `sci_tech`).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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

    ds = load_dataset(ref.repo, revision=ref.revision, split=ref.split)
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


def write_inputs(path: Path, items: list[Item]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(json.dumps({"id": it.id, "text": it.text}, ensure_ascii=False) + "\n")
