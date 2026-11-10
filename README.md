# Jetson model framework

Real-time EMS scene capture on a Jetson. A phone streams audio and video to the Jetson; model
services transcribe the audio, build a structured **electronic Patient Care Report (ePCR)** from
the transcript, and segment the video, all live while the session is recorded.

## Setup

### Prerequisites

- **A Jetson with JetPack** (developed on L4T R39) and about **100 GB of free disk**: the three
  model images are built on NVIDIA's PyTorch and NeMo images, over 30 GB each.
- **Docker with Compose v2 and the NVIDIA Container Toolkit**, with `nvidia` as Docker's
  **default runtime**. `docker-compose.yml` doesn't request GPUs itself, so the model services
  only get one through the default runtime. Check with `docker info | grep -i runtime`; if the
  default isn't `nvidia`, set it in `/etc/docker/daemon.json` and restart Docker:

  ```json
  {
    "runtimes": { "nvidia": { "path": "nvidia-container-runtime", "args": [] } },
    "default-runtime": "nvidia"
  }
  ```

  ```bash
  sudo systemctl restart docker
  ```

- **A Hugging Face account and access token**, with access granted to the gated
  [`facebook/sam3`](https://huggingface.co/facebook/sam3) model (request it on the model page).

### 1. Configure

```bash
cp .env.example .env          # then set HF_TOKEN=hf_... in .env
mkdir -p sessions logs run    # bind-mounted into the containers; must exist before starting
```

`.env` is required by `docker-compose.yml` and read by every service. `HF_TOKEN` is the only
value you must set; the rest of `.env.example` works as it is. A setting left out of `.env`
uses its default from the service's `config.py`. Restart the affected services after a change
(`docker compose up -d` recreates them with the new values).

**Shared**

| Setting | `.env.example` | Meaning |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Hugging Face token; its account needs access to `facebook/sam3` |
| `SAMPLING_RATE` | `16000` | audio sample rate (Hz) of the recording; read by the server and the ASR |
| `LOG_LEVEL` | `INFO` | log level of every service |
| `SESSIONS_DIR` | `/sessions` | session folders, inside the containers (`./sessions` on the host) |
| `LOGS_DIR` | `/logs` | one `<service>.log` per service, inside the containers (`./logs` on the host) |
| `BRIDGE_DIR` | `/run/bridge` | the services' bridge sockets, inside the containers (`./run` on the host) |

**Server** (`SERVER_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `SERVER_COMMAND_HOST` | `0.0.0.0` | address the control port listens on |
| `SERVER_COMMAND_PORT` | `8000` | client control port |
| `SERVER_MONITOR` | `true` | ping the services in the background |
| `SERVER_PING_INTERVAL` | `20` | seconds between pings |
| `SERVER_PING_TIMEOUT` | `2` | seconds to wait for a ping reply |
| `SERVER_MEDIA_API_URL` | `http://127.0.0.1:9997` | MediaMTX Control API |
| `SERVER_MEDIA_RTSP_URL` | `rtsp://127.0.0.1:8554` | where the server reads the published streams from |
| `SERVER_MEDIA_BINARY` | `mediamtx/mediamtx` | MediaMTX binary, relative to the working dir |
| `SERVER_MEDIA_CONFIG` | `/app/server/mediamtx.yml` | MediaMTX config file |

**ASR** (`ASR_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `ASR_LANGUAGE` | `en-US` | transcription language |
| `ASR_DELAY` | `2` | seconds the model stays behind the recording, so it only reads flushed audio |
| `ASR_NUM_LOOKAHEAD_TOKENS` | `6` | lookahead of the streaming model (higher = more accurate, more latency) |
| `ASR_POLL_INTERVAL` | `0.05` | seconds between checks for new audio |
| `ASR_SEGMENT_PAUSE` | `20` | seconds of silence that close a transcript segment |

**LLM** (`LLM_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `LLM_DRAIN_IDLE` | `30` | seconds `segments.json` must stay unchanged after a session ends before extraction stops; must exceed the ASR's end-of-session drain, or the last segment is missed |
| `LLM_LINK` | `false` | attach ATC / ICD-10-PCS codes to extracted entities; needs a code index in `llm_service/terminology/index` that isn't in the repo |
| `LLM_EXTRACT_MODEL` | *(not set: `Qwen/Qwen3-4B-Instruct-2507`)* | per-segment extraction model |
| `LLM_AGGREGATE_MODEL` | *(not set: `Qwen/Qwen2.5-7B-Instruct`)* | end-of-session aggregation model |
| `LLM_POLL_INTERVAL` | *(not set: `0.5`)* | seconds between checks for new segments |

**Video** (`VIDEO_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `VIDEO_IDLE_TIMEOUT` | `10` | seconds without a new frame before the video counts as finished |
| `VIDEO_DRAIN_IDLE` | `1` | seconds the video file must stop growing after a session ends before segmentation finishes |
| `VIDEO_PROBE_TIMEOUT` | `60` | seconds to wait for the video stream to become readable |
| `VIDEO_ALPHA` | `0.5` | opacity of the mask overlay, 0–1 |
| `VIDEO_POLL_INTERVAL` | `0.05` | seconds between checks for new frames |

### 2. Build and start

```bash
docker compose build          # first build downloads the NVIDIA base images; takes a while
docker compose build client   # the mock test client (profile "test", so built by name)
docker compose up -d          # starts the four services, not the client
docker compose ps             # server, asr_service, llm_service, video_service: all "Up"
docker compose logs -f        # wait for "waiting for a session" from each model service
```

A model service that isn't ready yet when a session starts doesn't work on that session
(`start_session` reports it as `false`).

On first start the model services download their weights from Hugging Face. Nothing caches
them outside the containers, so they're downloaded again whenever a container is recreated
(after a rebuild or `docker compose down`).

The server uses host networking: clients (the phone, on the Jetson's network) reach it at the
Jetson's address on ports **8000** (control) and **8554** (RTSP). Code under `common/` and each
service folder is mounted into the containers, so `docker compose restart <service>` picks up
edits; Dockerfile changes need `docker compose build` first.

To check the whole pipeline end to end, see [Testing the live stack](#testing-the-live-stack).

### 3. Stop

```bash
docker compose down
```

## How it works

```
phone ──TCP :8000 (JSON)──▶ server ──unix sockets (bridge)──▶ asr_service / llm_service / video_service
  │                           │
  └──RTSP :8554──▶ MediaMTX ──┴──▶ ffmpeg ──▶ sessions/<id>/audio/audio.pcm, video/video.ts
```

1. The client sends `{"command": "start_session"}` to the server's control port. The server
   creates `sessions/<id>/`, tells every service to work on it, and replies with an RTSP
   `streaming_url`.
2. The client publishes its stream there. The server records the audio as 16 kHz PCM and the
   video as MPEG-TS, and the services follow those files as they grow.
3. `{"command": "stop_session"}` ends the recording; each service finishes the input already
   recorded, and the server exports `audio.mp3` and `video.mp4`.

## Services

| Service | Reads | Writes | Model |
|---|---|---|---|
| `server` | the client's commands and RTSP stream | `audio/audio.pcm` → `audio.mp3`, `video/video.ts` → `video.mp4` | none (CPU-only; runs MediaMTX) |
| `asr_service` | `audio/audio.pcm` | `transcript/transcript.txt`, `transcript.jsonl`, `segments.json` | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| `llm_service` | `transcript/segments.json` | `structured/extractions.json`, then `aggregate.json` | `Qwen/Qwen3-4B-Instruct-2507` (extract), `Qwen/Qwen2.5-7B-Instruct` (aggregate) |
| `video_service` | `video/video.ts` | `video/overlay.mp4` | `facebook/sam3` |

The server drives the services over unix sockets in `BRIDGE_DIR` with three commands:
`queue_session` (work on a session, ending the current one first), `end_session` (finish the
current session's remaining input, then wait) and `ping` (health check, every
`SERVER_PING_INTERVAL` seconds). The shared service logic is `WorkerService` in
`common/service.py`.

## Layout

```
server/          control port, session lifecycle, MediaMTX + ffmpeg recording
asr_service/     streaming speech recognition and pause-based segmentation
llm_service/     live ePCR extraction and aggregation (prompts/, terminology/)
video_service/   SAM 3 video segmentation
common/          shared WorkerService, bridge, config, logging, file helpers
tests/           mock streaming client (client.py) and its Dockerfile
sessions/        one folder per session (created at runtime)
logs/            one log file per service
```

## Testing the live stack

`tests/client.py` is a mock client that streams test media to the server like the phone app
would. It starts a session, publishes the `.mp3` and/or `.mp4` in `tests/inputs/` to the
server's RTSP URL in real time, then stops the session. It runs in its own container
(`tests/Dockerfile`), because the Jetson's own ffmpeg can't publish RTSP or decode MP3.

Start the stack and build the client first ([Setup](#setup), step 2).

### Run the client

```bash
docker compose run --rm client --mode both    # .mp4 video + .mp3 audio (default)
docker compose run --rm client --mode audio   # only the .mp3
docker compose run --rm client --mode video   # only the .mp4
```

Each run starts a session (printing its `session_id`; the client warns if a service didn't
accept it), streams in real time (each file
plays to its own end, so `both` lasts as long as the longer file), waits 2 s (`--linger`)
and stops the session. Ctrl-C stops early and still ends the session cleanly. The server
takes one client and one session at a time, so run one at a time.

`--dry-run` prints the ffmpeg command without contacting the server; `--help` lists every
option. `tests/client.py` and `tests/inputs/` are mounted into the container, so edits to
them need no rebuild.

### Check the results

Everything lands in `sessions/<session_id>/`:

| While streaming | After `stop_session` |
|---|---|
| `audio/audio.pcm` grows | `audio/audio.mp3` |
| `transcript/transcript.txt`, `transcript.jsonl`, `segments.json` | — |
| `video/video.ts` grows, plus the video service's output | `video/video.mp4` |
| `structured/extractions.json`, updated as each segment closes | `structured/aggregate.json`, about 30 s later (`LLM_DRAIN_IDLE`) |

```bash
ls -R sessions/<session_id>
tail -f sessions/<session_id>/transcript/transcript.txt   # live transcript
tail -f logs/*.log                                        # service logs
```

The ASR closes a segment only after 20 s of silence (`ASR_SEGMENT_PAUSE`), so
`segments.json` and `extractions.json` may stay empty until a short stream ends.

### Shorter runs with clips

The full inputs take minutes per run. To cut 20 s clips (using the client image's ffmpeg)
and stream those instead:

```bash
mkdir -p clips
docker compose run --rm -v "$PWD/clips:/out" --entrypoint sh client -c \
  'ffmpeg -v error -y -i inputs/audio.mp3 -t 20 -c copy /out/audio.mp3 &&
   ffmpeg -v error -y -i inputs/video.mp4 -t 20 -c copy /out/video.mp4'

docker compose run --rm -v "$PWD/clips:/tests/inputs:ro" client --mode both
```

### From another machine

On the Jetson's network, with Python 3 and a full ffmpeg installed:

```bash
python tests/client.py --host <jetson-ip> --mode both
```
