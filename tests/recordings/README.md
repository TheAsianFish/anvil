# Recorded teacher responses

Each JSON file is one real Messages API exchange, `{"request": ..., "response": ...}`,
written by `RecordingTransport` when `ANVIL_RECORD_DIR` is set during a live run:

```bash
export ANTHROPIC_API_KEY=...
make record        # runs the live tests with recording on (toy task, claude-haiku-5-5)
```

`tests/test_replay.py` replays them offline through `ReplayTransport` and the same parsing
code as live calls, and fails if the current prompt no longer reproduces a recorded request
exactly. Recordings are never hand-written or edited.
