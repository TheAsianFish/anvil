# A VHS (https://github.com/charmbracelet/vhs) script for the README demo recording:
# `anvil label`, `anvil train` and `anvil eval` on the toy task with Claude as the teacher.
# Run from the repository root with ANTHROPIC_API_KEY set and `teacher.model` chosen in
# examples/toy/anvil.yaml: `vhs docs/demo.tape` writes docs/screenshots/demo.gif.
# The labels cost about $0.04 with Haiku 5.5 (RESULTS.md, "To measure").

Output docs/screenshots/demo.gif

Set Shell bash
Set FontSize 15
Set Width 1200
Set Height 640
Set TypingSpeed 40ms

Type "cd examples/toy"
Enter
Sleep 500ms

Type "uv run anvil label --inputs inputs.jsonl --n 200"
Enter
Sleep 90s

Type "uv run anvil train"
Enter
Sleep 60s

Type "uv run anvil eval"
Enter
Sleep 20s
