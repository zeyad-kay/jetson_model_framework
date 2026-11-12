# Test clients (tests/integration, tests/e2e) with a full ffmpeg: RTSP output, the mp3/h264
# decoders and the aac/libx264 encoders, which the host's ffmpeg lacks.
# The test media (integration/*/inputs, ground_truth) is mounted at runtime, not baked in.
FROM ubuntu:24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tests
COPY integration/ integration/
COPY e2e/*.py e2e/

# Each compose service (client, asr_test, video_test, llm_extraction_test,
# llm_aggregation_test) sets its own entrypoint.
ENTRYPOINT ["python3"]
