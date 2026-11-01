FROM nvcr.io/nvidia/nemo-speech:26.07.00

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# The base image's transformers predates AutoModelForRNNT, which Nemotron needs.
RUN pip install --no-cache-dir transformers==5.21.0

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
COPY common/ common/
COPY asr_service/ asr_service/

CMD ["python", "-m", "asr_service.main"]
