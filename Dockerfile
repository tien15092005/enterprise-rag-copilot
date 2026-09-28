FROM python:3.12-slim AS builder

ARG TORCH_VERSION=2.14.0
ARG PYTORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Resolve PyTorch from the CPU-only index before sentence-transformers dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url "$PYTORCH_CPU_INDEX_URL" "torch==$TORCH_VERSION"

WORKDIR /build
COPY pyproject.toml README.md ./
COPY requirements/docker-constraints.txt ./requirements/docker-constraints.txt
COPY app ./app
RUN python -m pip install --constraint requirements/docker-constraints.txt .

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/app/data/model_cache \
    EMBEDDING_CACHE_DIRECTORY=/app/data/model_cache \
    RERANKER_CACHE_DIRECTORY=/app/data/model_cache \
    UPLOAD_DIRECTORY=/app/data/uploads

RUN groupadd --system app \
    && useradd --system --gid app --create-home app \
    && mkdir -p /app/data/uploads /app/data/model_cache \
    && chown -R app:app /app

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app app ./app
COPY --chown=app:app scripts/docker-entrypoint.sh ./scripts/docker-entrypoint.sh

USER app
EXPOSE 8000

CMD ["sh", "/app/scripts/docker-entrypoint.sh"]
