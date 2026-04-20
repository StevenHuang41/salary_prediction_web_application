FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /backend

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

COPY pyproject.toml uv.lock* ./

RUN uv sync --locked --no-dev --no-install-project


FROM python:3.12-slim

WORKDIR /backend

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
            libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /backend/.venv /backend/.venv

ENV PATH="/backend/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/backend/src

COPY . .

RUN mkdir -p /backend/artifacts

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
