.PHONY: help sync check fmt lint typecheck test test-live record demo docker spike-onnx clean

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

docker: ## build the CLI image
	docker build -t anvil:dev .

spike-onnx: ## M1 spike: export ModernBERT-base to ONNX and check parity (downloads ~600 MB)
	$(RUN) python eval/spikes/onnx_export_spike.py --out eval/results/spike-onnx-export

clean: ## remove the demo workspace and tool caches
	rm -rf .anvil/demo .pytest_cache .mypy_cache .ruff_cache dist
