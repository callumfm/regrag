FROM python:3.14-slim-bookworm

ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.9.16 /uv /bin/uv

ENV PATH="/app/.venv/bin:$PATH"
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN useradd --create-home --uid 1000 regrag \
    && mkdir -p /data/raw \
    && chown -R regrag:regrag /data

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=.python-version,target=.python-version \
    uv sync --frozen --no-install-project --no-dev

ENV PYTHONPATH=/app

COPY ./pyproject.toml ./uv.lock ./.python-version ./alembic.ini /app/
COPY ./migrations /app/migrations
COPY ./app /app/app

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

USER regrag
EXPOSE 8000

CMD ["fastapi", "run", "--port", "8000", "app/main.py"]
