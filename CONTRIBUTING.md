# Contributing

This is the working guide for anyone touching the repo. Read it before the
first commit of a session.

## Setup

```bash
uv sync --all-extras          # Python 3.12, creates .venv and installs dev tools
uv run anvil --help
```

CPU-only torch is enough for everything except the GPU frontier run; CI
installs it from the CPU wheel index (see `pyproject.toml` `[tool.uv.sources]`
once M1 lands). Set `ANTHROPIC_API_KEY` only when you intend to spend money:
live tests and evals are skipped without it and nothing else needs it.

## Commands

| Command | What it does |
|---|---|
| `make check` | ruff format check, ruff lint, mypy strict, pytest (offline, fast). The quality gate. |
| `make fmt` | ruff format + autofix |
| `make test` | offline unit and integration tests (fake teacher, tiny models, replayed recordings) |
| `make test-live` | tests that call Claude; skipped without `ANTHROPIC_API_KEY` |
| `make demo` | the offline vertical slice on the toy task (under two minutes) |
| `make eval-classify` / `eval-extract` / `eval-generate` | parity evals; write `eval/results/` and refresh RESULTS.md |
| `make eval-frontier` / `eval-select` / `eval-calibration` / `eval-export` | the frontier, selection, calibration and export studies |
| `make bench` | latency, throughput and cost table |
| `make docker` | build the serving image |

Every eval and bench writes raw output under `eval/results/<name>/` with the
command, git SHA, models, seed, wall time and dollars spent. RESULTS.md is
regenerated from those files, never edited by hand for numbers.

## Layout

```
anvil/            package (spec, teacher/, store, students/, train/, align,
                  evaluate/, select, loop, calibrate, export, serve, hub, cli)
tests/            unit + integration; tests/recordings/ holds real Claude
                  responses replayed offline
eval/             datasets loaders, study scripts, results/
docs/             notes/ (spikes), screenshots, architecture
notebooks/        the Colab GPU run
```

## Conventions

- **Typed.** mypy strict on `anvil/` and `tests/`. No `Any` without a comment.
- **Formatted and linted** with ruff (line length 99). Imports sorted.
- **Tests are offline, fast and deterministic.** Network is mocked or
  replayed; models in tests are tiny random-init configs, never downloads;
  seeds are explicit. A test that needs a key is marked `live` and skipped
  without one.
- **Config via environment variables** with the `ANVIL_` prefix, read in one
  place (`anvil/settings.py`). No secrets in the repo, ever. The disk cache
  under `.anvil/` is gitignored.
- **Structured logging** (JSON lines) through `anvil/log.py`; no bare
  `print` outside the CLI's final summary.
- **Errors are specific.** Raise `AnvilError` subclasses with a message that
  says what to do; the CLI maps them to exit code 2 for configuration and
  1 for runtime failures.
- **Numbers are measured.** A number in README or RESULTS.md must come from
  a command in this file, with its raw output committed. If it was not run,
  it goes under "To measure" with the command and a cost estimate.
- **Teacher calls are cached and accounted.** Every call records tokens and
  dollars; re-running an eval with the same prompt, model and inputs costs
  nothing.

## Commits and PRs

- One logical step per commit, imperative subject under 72 characters, a
  short body only when the why is not obvious. Fold fix-ups into the commit
  they fix before pushing.
- Each milestone is one PR from `dev/<milestone>` into `main`. The PR body
  states what was built, what was measured (with the commands and output),
  and what was left out and why.
- Tick a milestone in PLAN.md only when its definition of done is met and
  `make check` is green in CI.

## Quality gates

A PR is mergeable when:

1. `make check` passes locally and in CI (lint, types, tests);
2. `make demo` runs offline;
3. every new number in RESULTS.md has its raw output under `eval/results/`;
4. no file under `.anvil/`, no credentials, no `CLAUDE.md`/`AGENTS.md`/`.claude/`/`.agent/` is tracked;
5. PLAN.md reflects the state of the milestone.
