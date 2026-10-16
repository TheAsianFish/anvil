"""Generate the toy support-ticket inputs (deterministic; the output is committed).

uv run python examples/toy/make_inputs.py > examples/toy/inputs.jsonl
"""

from __future__ import annotations

import json
import random
import sys

SUBJECTS = {
    "billing": [
        "I was charged twice for my subscription",
        "my invoice shows the wrong amount",
        "please refund last month's payment",
        "why did my card get a charge from you",
        "the receipt for my order is missing the tax line",
        "can I switch my billing to annual",
    ],
    "bug": [
        "the app crashes when I open settings",
        "export to CSV throws an error",
        "the page freezes after I click save",
        "login fails with a blank screen on Android",
        "notifications stopped working after the update",
        "search returns a 500 error",
    ],
    "feature_request": [
        "it would be great to have dark mode",
        "please add an option to schedule reports",
        "could you support SSO with Okta",
        "I wish I could share dashboards with a link",
        "would love a keyboard shortcut for search",
        "can you add an API endpoint for exports",
    ],
    "account": [
        "I can't reset my password",
        "how do I change the email on my account",
        "please delete my account and data",
        "I need to add a teammate to my workspace",
        "my two-factor code is not accepted",
        "how do I transfer ownership of the account",
    ],
}
OPENERS = ["", "Hi team, ", "Hello, ", "Urgent: ", "Quick question - ", "Hey, "]
CLOSERS = ["", " Thanks.", " Thanks in advance!", " This is blocking us.", " Any update?", ""]
VAGUE = [
    "something is not right with my account page and the charge",
    "it doesn't work",
    "I have a question about your product",
    "nothing happens when I try",
    "can someone call me back",
]


def main(n: int = 400, seed: int = 7) -> None:
    rng = random.Random(seed)
    labels = list(SUBJECTS)
    for i in range(n):
        if rng.random() < 0.08:
            body = rng.choice(VAGUE)
        else:
            body = rng.choice(SUBJECTS[rng.choice(labels)])
        text = f"{rng.choice(OPENERS)}{body}.{rng.choice(CLOSERS)}".strip()
        sys.stdout.write(json.dumps({"id": f"t{i:04d}", "text": text}) + "\n")


if __name__ == "__main__":
    main()
