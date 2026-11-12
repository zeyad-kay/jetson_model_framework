FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
COPY common/ common/
COPY llm_extraction_service/ llm_extraction_service/
COPY old_epcr/ old_epcr/
COPY artifacts/ artifacts/

RUN pip install transformers accelerate

CMD ["python", "-m", "llm_extraction_service.main"]
