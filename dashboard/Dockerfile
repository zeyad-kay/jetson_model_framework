# The dashboard is stdlib-only Python serving static files; no GPU or PyTorch.
FROM ubuntu:24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
COPY common/ common/
COPY dashboard/ dashboard/

CMD ["python", "-m", "dashboard.main"]
