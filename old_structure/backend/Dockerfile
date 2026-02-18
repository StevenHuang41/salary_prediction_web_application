FROM python:3.12.3-slim-bullseye

WORKDIR /backend

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y libgomp1 libsqlite3-dev curl&& \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --locked

ENV PATH="/backend/.venv/bin:$PATH"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", \
     "0.0.0.0", "--port", "8000"]
