FROM nvcr.io/nvidia/nemo-speech:26.07.00

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
COPY asr_service/ asr_service/

CMD ["python", "-m", "asr_service.main"]
