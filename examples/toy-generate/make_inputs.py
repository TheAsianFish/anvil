"""Generate the toy text-to-SQL inputs (deterministic; the output is committed).

uv run python examples/toy-generate/make_inputs.py > examples/toy-generate/inputs.jsonl
"""

from __future__ import annotations

import json
import random
import sys

COLUMNS = '"name" (text), "team" (text), "city" (text), "age" (real)'
QUESTIONS = [
    "who works in the {team} team?",
    "who is in the {team} team?",
    "how many people live in {city}?",
    "how many people are based in {city}?",
    "how old is {name}?",
    "what age is {name}?",
    "which team is {name} on?",
    "what team does {name} work in?",
    "who is older than {age}?",
    "list everyone older than {age}",
    "what is the oldest age in {city}?",
    "the oldest age in {city}?",
]
SLOTS = {
    "team": ["sales", "support", "design", "finance", "legal", "data", "ops", "hr", "research"],
    "city": ["paris", "lima", "oslo", "osaka", "denver", "lagos", "hanoi", "quito", "porto"],
    "name": ["ana", "bo", "chen", "dara", "eli", "femi", "gus", "hana", "ivo", "jun", "kai"],
    "age": ["25", "30", "33", "41", "47", "52"],
}


def main() -> None:
    rng = random.Random(0)
    seen: set[str] = set()
    n = 0
    while n < 200:
        question = rng.choice(QUESTIONS).format(**{k: rng.choice(v) for k, v in SLOTS.items()})
        text = f"Table t columns: {COLUMNS}\nQuestion: {question}"
        if rng.random() < 0.3:  # the rewrites match either case, the student must too
            text = text.replace("Question: w", "Question: W").replace("Question: h", "Question: H")
        if text in seen:
            continue
        seen.add(text)
        sys.stdout.write(json.dumps({"id": f"q{n:04d}", "text": text}) + "\n")
        n += 1


if __name__ == "__main__":
    main()
