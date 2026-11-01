FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
COPY common/ common/
COPY llm_service/ llm_service/

CMD ["python", "-m", "llm_service.main"]
