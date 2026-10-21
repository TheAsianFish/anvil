.PHONY: help sync check fmt lint typecheck test test-live record demo wheel-check docker spike-onnx \
	eval-classify eval-extract eval-generate eval-select eval-frontier eval-calibration eval-export bench \
	results clean

UV ?= uv
RUN := $(UV) run
DEMO_HOME ?= $(CURDIR)/.anvil/demo

help:
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

sync: ## install the locked environment (CPU torch, all extras, dev tools)
	$(UV) sync --all-extras --locked

check: lint typecheck test ## the quality gate: format check, lint, mypy strict, offline tests

lint: ## ruff format check and lint
	$(RUN) ruff format --check .
	$(RUN) ruff check .

typecheck: ## mypy strict
	$(RUN) mypy

fmt: ## format and autofix
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

test: ## offline unit and integration tests
	$(RUN) pytest -q

test-live: ## tests that call Claude (skipped without ANTHROPIC_API_KEY)
	$(RUN) pytest -q -m live

record: ## re-record the Claude fixtures in tests/recordings (spends a few cents)
	ANVIL_RECORD_DIR=$(CURDIR)/tests/recordings/ticket-triage $(RUN) pytest -q -m live

demo: ## offline vertical slice on the toy task: label -> train -> eval
	./scripts/demo.sh "$(DEMO_HOME)"

wheel-check: ## build the wheel, install it in a fresh venv, run `anvil --help` and the demo on it
	./scripts/wheel_check.sh

docker: ## build the CLI image and the torch-free serving image
	docker build -t anvil:dev .
	docker build --target serve -t anvil-serve:dev .

TEACHER ?= auto
eval-classify: ## M2 parity on AG News and Banking77 (TEACHER=auto|claude|replay|gold), then RESULTS.md
	$(RUN) --group eval python -m eval.classify ag_news banking77 --teacher $(TEACHER)
	$(RUN) python -m eval.render

eval-extract: ## M4 span extraction parity on MIT Restaurant (TEACHER=auto|claude|replay|gold), then RESULTS.md
	$(RUN) python -m eval.extract mit_restaurant --teacher $(TEACHER)
	$(RUN) python -m eval.render

eval-generate: ## M6 text-to-SQL parity on WikiSQL by execution match (TEACHER=auto|claude|replay|gold), then RESULTS.md
	$(RUN) --group eval python -m eval.generate wikisql --teacher $(TEACHER)
	$(RUN) python -m eval.render

eval-select: ## M3 selection strategies on synthetic blobs (offline, free), then RESULTS.md
	$(RUN) python -m eval.select_study
	$(RUN) python -m eval.render

eval-frontier: ## M3 kappa vs teacher labels per strategy, 3 seeds x 5 budgets (TEACHER=auto|claude|replay|gold), then RESULTS.md
	$(RUN) --group eval python -m eval.frontier --teacher $(TEACHER)
	$(RUN) python -m eval.render

eval-calibration: ## M5 temperature, ECE and coverage at target kappa on the M2 students (TEACHER=...), then RESULTS.md
	$(RUN) --group eval python -m eval.calibration --teacher $(TEACHER)
	$(RUN) python -m eval.render

eval-export: ## M5 ONNX fp32/int8 parity and the int8 layer probe on the M2 students, then RESULTS.md
	$(RUN) --group eval python -m eval.export --teacher $(TEACHER)
	$(RUN) python -m eval.render

bench: ## M5 latency, throughput and $ per 1k: PyTorch vs ONNX vs the HTTP server (after eval-export)
	$(RUN) --group eval python -m eval.bench --teacher $(TEACHER)
	$(RUN) python -m eval.render

results: ## regenerate the RESULTS.md tables from eval/results/
	$(RUN) python -m eval.render

spike-onnx: ## M1 spike: export ModernBERT-base to ONNX and check parity (downloads ~600 MB)
	$(RUN) python eval/spikes/onnx_export_spike.py --out eval/results/spike-onnx-export

clean: ## remove the demo workspace and tool caches
	rm -rf .anvil/demo .pytest_cache .mypy_cache .ruff_cache dist
