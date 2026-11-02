FROM nvcr.io/nvidia/pytorch:25.08-py3

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Build context is the repo root (see docker-compose.yml).
COPY common/ common/
COPY video_service/ video_service/

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && pip install transformers accelerate av

CMD ["python", "-m", "video_service.main"]
