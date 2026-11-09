FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY data ./data
RUN uv sync --frozen --no-dev

# Point at an Ollama server on the host, e.g. -e OLLAMA_HOST=http://host.docker.internal:11434
ENTRYPOINT ["uv", "run", "agent-eval"]
CMD ["--models", "reference", "--gate"]
