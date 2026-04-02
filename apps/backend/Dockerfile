FROM python:3.12-slim AS builder

WORKDIR /backend

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --locked --no-dev


FROM python:3.12-slim

WORKDIR /backend

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /backend/.venv /backend/.venv

ENV PATH="/backend/.venv/bin:$PATH"
ENV PYTHONPATH=/backend/src

COPY . .

RUN mkdir -p /backend/artifacts

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
