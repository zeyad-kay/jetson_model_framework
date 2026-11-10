# The server is stdlib-only Python plus ffmpeg (CPU codecs) and MediaMTX; it
# needs no GPU or PyTorch, so a plain Ubuntu base is enough.
FROM ubuntu:24.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg python3 python-is-python3 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Download and extract the MediaMTX binary (linux arm64).
RUN mkdir -p mediamtx \
    && curl -fsSL https://github.com/bluenviron/mediamtx/releases/download/v1.21.2/mediamtx_v1.21.2_linux_arm64.tar.gz \
       | tar -xz -C mediamtx

COPY common/ common/
COPY server/ server/

CMD ["python", "-m", "server.main"]