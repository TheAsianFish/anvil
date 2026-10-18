"""Generate the toy restaurant-search inputs (deterministic; the output is committed).

uv run python examples/toy-extract/make_inputs.py > examples/toy-extract/inputs.jsonl
"""

from __future__ import annotations

import json
import random
import sys

TEMPLATES = [
    "find me a {cuisine} place near {location}",
    "is {name} open {hours}",
    "where can I get {dish} in {location}",
    "book a table at {name} for {hours}",
    "any {price} {cuisine} restaurants around {location}",
    "I want {dish} from {name}",
    "show {cuisine} spots with {amenity}",
    "{price} {dish} near {location} please",
    "does {name} have {amenity}",
    "looking for {cuisine} food {hours}",
    "what's good to eat around here",
    "take me somewhere nice",
]
SLOTS = {
    "cuisine": ["thai", "mexican", "Italian", "sushi", "vegan", "Korean", "indian"],
    "location": ["downtown", "the airport", "Brooklyn", "my office", "union square", "Zürich"],
    "name": ["Pizza Hut", "Olive Garden", "Taco Bell", "Café Rouge", "Nobu", "Shake Shack"],
    "hours": ["tonight", "after 10pm", "on Sunday", "for lunch", "late"],
    "dish": ["ramen", "fish tacos", "pad thai", "a burger", "dim sum", "crème brûlée"],
    "price": ["cheap", "fancy", "affordable", "upscale"],
    "amenity": ["outdoor seating", "parking", "wifi", "a kids menu", "live music"],
}


def main() -> None:
    rng = random.Random(0)
    seen: set[str] = set()
    n = 0
    while n < 320:
        template = rng.choice(TEMPLATES)
        text = template.format(**{k: rng.choice(v) for k, v in SLOTS.items()})
        if rng.random() < 0.3:
            text = text[0].upper() + text[1:] + "?"
        if text in seen:
            continue
        seen.add(text)
        sys.stdout.write(json.dumps({"id": f"r{n:04d}", "text": text}, ensure_ascii=False) + "\n")
        n += 1


if __name__ == "__main__":
    main()
