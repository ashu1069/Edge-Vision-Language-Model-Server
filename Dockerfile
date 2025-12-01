FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies with uv (no dev deps in container)
COPY pyproject.toml ./pyproject.toml
RUN uv sync --no-dev

# Copy application code
COPY app ./app
COPY README.md ./README.md

# Ensure the venv from uv is on PATH by default
ENV PATH="/app/.venv/bin:${PATH}"

# Default command (can be overridden by docker-compose)
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


