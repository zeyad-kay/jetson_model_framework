FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN pip install transformers accelerate

# Build context is the repo root (see docker-compose.yml).
COPY common/ common/
COPY llm_aggregation_service/ llm_aggregation_service/

CMD ["python", "-m", "llm_aggregation_service.main"]
