# Two images from one file:
#   docker build -t anvil .                      the CLI (label, train, eval, export) on CPU torch
#   docker build --target serve -t anvil-serve .  `anvil serve` on ONNX Runtime, no torch
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project --extra export
COPY anvil ./anvil
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra export

# The serving dependencies, pinned by the same lock file. torch, transformers and peft are core
# dependencies of the package (training needs them), so they are filtered out of the exported
# requirements together with what only they pull in. The export is already the full pinned
# closure, so both installs skip dependency resolution: a training package left in the list
# must not pull torch back in from PyPI (the CUDA build, several GB). `anvil serve` never
# imports them; tests enforce that and check the filter against uv.lock.
FROM python:3.12-slim AS serve-build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv export --locked --no-dev --extra serve --no-emit-project --no-hashes --no-header \
        --no-annotate -o requirements-all.txt \
    && grep -viE '^(torch|transformers|peft|accelerate|psutil|sympy|mpmath|networkx|jinja2|markupsafe|safetensors|regex|setuptools)==' \
        requirements-all.txt | grep -v '^--' > requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && uv pip install --python /app/.venv --no-deps -r requirements.txt
COPY anvil ./anvil
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir dist && uv pip install --python /app/.venv --no-deps dist/*.whl

FROM python:3.12-slim AS serve
RUN useradd --create-home --uid 1000 anvil
COPY --from=serve-build /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" ANVIL_HOME=/work/.anvil PYTHONUNBUFFERED=1
WORKDIR /work
RUN chown anvil:anvil /work
USER anvil
EXPOSE 8000
# Mount an `anvil export` bundle at /model (read-only is fine); extra arguments are appended:
#   docker run -p 8000:8000 -v "$PWD/bundle:/model:ro" anvil-serve --teacher none
ENTRYPOINT ["anvil", "serve", "/model", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 anvil
COPY --from=build /app/.venv /app/.venv
COPY examples /app/examples
ENV PATH="/app/.venv/bin:$PATH" ANVIL_HOME=/work/.anvil PYTHONUNBUFFERED=1
WORKDIR /work
RUN chown anvil:anvil /work
USER anvil
ENTRYPOINT ["anvil"]
CMD ["--help"]
