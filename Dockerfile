# CLI image for anvil (label, train, eval) on CPU.
# M5 adds a slimmer serving image that ships ONNX Runtime without torch.
FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.13 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-install-project
COPY anvil ./anvil
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable

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
