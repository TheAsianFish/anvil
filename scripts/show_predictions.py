"""Print an `anvil serve` /predict response (read from stdin) as one line per input."""

import json
import sys

for p in json.load(sys.stdin)["predictions"]:
    line = f"  {p['label']:<16} from the {p['source']:<8} (confidence {p['confidence']:.3f})"
    print(line)  # noqa: T201 - the demo's output
