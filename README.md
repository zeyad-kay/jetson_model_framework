# Jetson model framework

Real-time EMS scene capture on a Jetson. A phone streams audio and video to the Jetson; model
services transcribe the audio, build a structured **electronic Patient Care Report (ePCR)** from
the transcript, and segment the video, all live while the session is recorded.

## Setup

### Prerequisites

- **A Jetson with JetPack** (developed on L4T R39) and about **100 GB of free disk**: the four
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
mkdir -p sessions logs/tests run   # bind-mounted into the containers; must exist before starting
```

`.env` is required by `docker-compose.yml` and read by every service. `HF_TOKEN` is the only
value you must set; the rest of `.env.example` works as it is. A setting left out of `.env`
uses its default from the service's `config.py`. Restart the affected services after a change
(`docker compose up -d` recreates them with the new values).

**Shared**

| Setting | `.env.example` | Meaning |
|---|---|---|
| `HF_TOKEN` | *(empty)* | Hugging Face token; its account needs access to `facebook/sam3`. No service reads it: Compose puts it in every container's environment, where the Hugging Face library picks it up when downloading weights |
| `SAMPLING_RATE` | `16000` | audio sample rate (Hz) of the recording; read by the server and the ASR |
| `LOG_LEVEL` | `INFO` | log level of every service |
| `SESSIONS_DIR` | `/sessions` | session folders, inside the containers (`./sessions` on the host) |
| `LOGS_DIR` | `/logs` | one `<service>.log` per service, inside the containers (`./logs` on the host) |
| `STATUS_DIR` | `/run/status` | the model services' heartbeats (`<service>.json`), which the server reads, inside the containers (`./run` on the host) |
| `SESSION_MAX_AGE` | `86400` | the model services pick up sessions created at most this many seconds ago |
| `SESSION_ABANDON_TIMEOUT` | `1800` | a model service gives up on a session whose input isn't marked complete once nothing in the session's folder has changed for this many seconds, and marks its output `abandoned` |

**Server** (`SERVER_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `SERVER_COMMAND_HOST` | `0.0.0.0` | address the control port listens on |
| `SERVER_COMMAND_PORT` | `8000` | client control port |
| `SERVER_MONITOR` | `true` | check the services' heartbeats in the background, logging when one goes up or down |
| `SERVER_MONITOR_INTERVAL` | `20` | seconds between heartbeat checks |
| `SERVER_STATUS_TIMEOUT` | `15` | a service whose heartbeat is older than this many seconds counts as down |
| `SERVER_MEDIA_API_URL` | `http://127.0.0.1:9997` | MediaMTX Control API |
| `SERVER_MEDIA_RTSP_URL` | `rtsp://127.0.0.1:8554` | where the server reads the published streams from |
| `SERVER_MEDIA_BINARY` | `mediamtx/mediamtx` | MediaMTX binary, relative to the working dir |
| `SERVER_MEDIA_CONFIG` | `/app/server/mediamtx.yml` | MediaMTX config file |

**ASR** (`ASR_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `ASR_LANGUAGE` | `en-US` | transcription language |
| `ASR_NUM_LOOKAHEAD_TOKENS` | `6` | lookahead of the streaming model (higher = more accurate, more latency) |
| `ASR_POLL_INTERVAL` | `0.05` | seconds between checks for new audio |
| `ASR_SEGMENT_PAUSE` | `20` | seconds of silence that close a transcript segment |
| `ASR_SEGMENT_MIN_LENGTH` | `20` | seconds of audio a segment needs before the end of a sentence closes it |

**LLM extraction** (`LLM_EXTRACTION_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `LLM_EXTRACTION_MODEL` | `Qwen/Qwen3-4B-Instruct-2507` | per-segment extraction model |
| `LLM_EXTRACTION_POLL_INTERVAL` | `0.5` | seconds between checks for new segments |
| `LLM_EXTRACTION_LINK` | `true` | attach ATC / ICD-10-PCS codes to extracted entities (`llm_extraction_service/terminology_linker.py`); needs a code index in `llm_extraction_service/terminology/index` that isn't in the repo, so set it to `false` until the index is in place |
| `LLM_EXTRACTION_DIR` | *(not set: the `llm_extraction_service/` folder)* | the service's folder; the code index is read from `<LLM_EXTRACTION_DIR>/terminology/index` |
| `LLM_EXTRACTION_PROMPTS_DIR` | *(not set: `<LLM_EXTRACTION_DIR>/prompts`)* | prompts: `separated/<category>.txt` (one prompt per category, named after it: one model call per prompt, given the category's earlier records and the new segment, and returning only what the new segment states; a file named after no category, or a category with no file, fails at startup; the demographics and mechanism of injury calls stop once those are complete), `_shared.yaml` for the fragments they use (a placeholder with no fragment fails at startup) |

**Video** (`VIDEO_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `VIDEO_PROMPT` | `wound` | text prompt naming what SAM 3 segments |
| `VIDEO_PROBE_TIMEOUT` | `60` | seconds to wait for the video stream to become readable |
| `VIDEO_MIN_SCORE` | `0.9` | minimum detection score, 0–1, of an object for its frame to be saved |
| `VIDEO_JPEG_QUALITY` | `90` | JPEG quality of the saved frames, 1–95 |
| `VIDEO_POLL_INTERVAL` | `0.05` | seconds between checks for new frames |

**Dashboard** (`DASHBOARD_`)

| Setting | `.env.example` | Meaning |
|---|---|---|
| `DASHBOARD_HOST` | `0.0.0.0` | address the dashboard's web page is served on |
| `DASHBOARD_PORT` | `8080` | its port |
| `DASHBOARD_LIVE_WINDOW` | `120` | a session still recording or processing shows as stalled once none of its files has changed for this many seconds |

### 2. Build and start

```bash
docker compose build          # first build downloads the NVIDIA base images; takes a while
docker compose build e2e_test # the test image (profile "test", so built by name)
docker compose up -d          # starts the six services, not the test clients
docker compose ps             # server, the four model services and the dashboard: all "Up"
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

To check the whole pipeline end to end, see [Testing the live stack](#testing-the-live-stack);
to test one model service on its own, see [Integration tests](#integration-tests).

### 3. Stop

```bash
docker compose down
```

## How it works

```
phone ──TCP :8000 (JSON)──▶ server ──▶ sessions/<id>/session.json ──▶ asr_service / llm_extraction_service /
  │                           │           (and the done markers)        video_service
  │                           │
  └──RTSP :8554──▶ MediaMTX ──┴──▶ ffmpeg ──▶ sessions/<id>/audio/audio.pcm, video/video.ts
```

1. The client sends `{"command": "start_session"}` to the server's control port. The server
   creates `sessions/<id>/` with `session.json` (`{"id", "created", "services"}`), which the
   services find on disk, and replies with an RTSP `streaming_url`.
2. The client publishes its stream there. The server records the audio as 16 kHz PCM and the
   video as MPEG-TS, and the services follow those files as they grow.
3. `{"command": "stop_session"}` ends the recording. The server marks both files complete
   (`audio.pcm.done`, `video.ts.done`) and replies; each service finishes the input already
   recorded. The server then exports `audio.mp3` and `video.mp4` in the background, one
   session at a time, each written atomically. When the server starts, it exports any
   session a previous run left unexported (a crash, a kill mid-export, or a session still
   recording when it was stopped).

### Done markers

A file that is written as the session goes on is marked complete by its writer with
`<file>.done` next to it, `{"status": ...}`, once it will write no more. Whoever follows
the file reads the rest and stops, without waiting for a timeout. Each writer marks its
output even when it fails, and writes the marker after its last write; readers check the
marker before reading (`done_marker`, `write_done` in `common/io.py`).

| Status | Means |
|---|---|
| `ok` | the file is complete |
| `failed` | its writer failed part way; the file holds what it wrote until then |
| `aborted` | a recording the server was cut off from by a crash or a kill; the server marks it when it starts again |
| `abandoned` | a service gave up waiting for its input, which nobody marked complete and nothing in the session changed for `SESSION_ABANDON_TIMEOUT` seconds; it processed all the input there was |

Each marker's status is about its own file: a service whose input ended `aborted` or
`failed` still marks its output `ok` once it has processed all of that input.

| Marker | Written by | Ends |
|---|---|---|
| `audio/audio.pcm.done` | server, when it stops recording (before the export) | `asr_service`'s session |
| `video/video.ts.done` | server, when it stops recording (before the export) | `video_service`'s session |
| `transcript/segments.json.done` | `asr_service`, after its last segment | `llm_extraction_service`'s session |
| `extractions/extractions.done` | `llm_extraction_service`, after its last segment | |
| `video/wounds.done` | `video_service`, after its last frame | |

The dashboard reads every marker to show a session's state (see [Dashboard](#dashboard)).

Each service also uses its own marker to tell which sessions it still has to do (see
below); deleting it makes the service redo the session.

## Services

| Service | Reads | Writes | Model |
|---|---|---|---|
| `server` | the client's commands and RTSP stream | `audio/audio.pcm` → `audio.mp3`, `video/video.ts` → `video.mp4` | none (CPU-only; runs MediaMTX) |
| `asr_service` | `audio/audio.pcm` | `transcript/transcript.jsonl` (the transcript's fragments, timestamped), `segments.json` | `nvidia/nemotron-3.5-asr-streaming-0.6b` |
| `llm_extraction_service` | `transcript/segments.json` | the ePCR as a log of per-segment records: `extractions/<category>.json` (one file per category: `medications`, `interventions`, `vital_signs`, `mechanism_of_injury`, `demographics`, `medical_history`, `patient_condition` (complaints, mental status, abnormal findings and suspected injuries, each with its clinical name and body site when stated); `[{"segment", "extractions": [...]}, ...]`, a record for each segment that stated something new for the category; each validated against `llm_extraction_service/schema.py`, with each quote a `{"text", "time"}` object; medications are actions on administrations, a bolus given or an infusion started, each with an id, then updated by id: details stated later, a new rate, stopped), `medication_administrations.json` (each administration's current state, merged from those actions) and `progress.json` (the last segment taken in), then `extractions.done` when the session's last segment is in | `Qwen/Qwen3-4B-Instruct-2507` |
| `video_service` | `video/video.ts` | `video/wounds/*.jpg` and `*.box` | `facebook/sam3` |
| `dashboard` | `extractions/<category>.json`, `medication_administrations.json` and `progress.json`, `video/wounds/`, and the done markers (read-only) | a web page on port 8080 | none (CPU-only) |

The services find their sessions on disk; nothing tells them a session started or
ended. The shared service logic is `WorkerService` in `common/service.py`. It works through
the pending sessions one at a time, oldest first. A session is pending if:
- it has `session.json`, at most `SESSION_MAX_AGE` seconds old;
- its `services` list (if any) names the service;
- the service's own output marker is missing (`segments.json.done`, `extractions.done`,
  `wounds.done`).

So a service that was down catches up on the sessions it missed. A session is tried once
per run of the service. On shutdown (`docker compose stop`), a service ends its current
session without marking it if its input wasn't complete, and redoes it on its next run.

A session whose input is never marked complete can't hold the services up for good:
- When the server starts, it marks the recordings a previous run left unfinished (a
  session with no `audio.pcm.done` or `video.ts.done`) `aborted` (`abort_unfinished` in
  `server/sessions.py`), so the services finish those sessions.
- A service gives up on its session once its reader has had nothing new and nothing in the
  session's folder has changed for `SESSION_ABANDON_TIMEOUT` seconds, and marks its output
  `abandoned`. For when the writer is gone and the server doesn't restart, e.g. an
  integration test that was killed. A live session's recordings keep growing, so it
  never looks idle.

Each service writes a heartbeat to `STATUS_DIR/<service>.json` every 5 seconds:
`{"service", "pid", "state", "session", "updated"}`, where `state` is `loading`, `idle`,
`working` or `stopped`. The server counts a service as up while its heartbeat is at most
`SERVER_STATUS_TIMEOUT` seconds old and it is idle or working. Files are the only way the
server and the services talk: the server never calls a service, and a service never calls
the server.

Every session runs in the same steps:

| Step | Does |
|---|---|
| `open_reader(session_dir)` | returns the reader of the session's input: `read()` gives the next input (a dict) or `None`, and raises `EOFError` once the input has finished for good; `close()` releases it |
| `setup(session_dir)` | opens the session's outputs and starts the model (`model.start()`) |
| `poll()` | the reader's next input, or `None`; raises `SessionDrained` once the reader finds the input finished (its upstream marked it complete), once the session is given up on, or at shutdown once the reader has nothing new |
| `predict(inputs)` | runs the model on that input (`model.predict(inputs)`) |
| `save(inputs, outputs)` | writes what it produced to the session folder |
| `cleanup()` | stops the model (`model.stop()`), calls `finalize_outputs()` (a no-op unless the service overrides it; the ASR closes its transcript files and writes the last segment) and marks `output_path(session_dir)` complete (its output marker, `session_status`: `"failed"` after a failure, `"abandoned"` if the session was given up on, else `"ok"`; none if shutdown interrupted the session), after the reader is closed; always runs |

`poll`, `predict` and `save` repeat until the session drains. `poll`, `predict` and `cleanup` are
implemented by `WorkerService`, so a service implements only `load_model`, `output_path`,
`open_reader`, `setup` and `save` (and `finalize_outputs` if it has outputs to finish). Models share the `StreamingModel` interface in `common/model.py`
(`start`, `predict`, `stop`): they never read files or wait for input; the service's reader does.
Each service's reader and model:

| Service | Reader: `read()` gives | Model: `predict()` returns |
|---|---|---|
| `asr_service` | `PcmReader` (`asr_service/reader.py`): `{"samples", "final"}`, the next chunk the model decodes (`Nemotron.chunk_span`, overlapping the one before), once recorded; the final read is the rest, shorter than a chunk | `Nemotron`: `{"fragments": [(text, seconds)], "seconds"}`; it runs the model's streaming `generate()` on a thread, fed one chunk per `predict()` |
| `llm_extraction_service` | `SegmentReader` (`llm_extraction_service/reader.py`): the next new segment | `QwenExtractor`: `{"segment", "start", "end", <category>: [...], "medication_administrations"}`, every category's records so far |
| `video_service` | `FrameReader` (`video_service/reader.py`): `{"frame", "seconds"}` | `Sam3`: `{"object_ids", "scores", "boxes"}` |

## Layout

```
server/                   control port, session lifecycle, MediaMTX + ffmpeg recording
asr_service/              streaming speech recognition, segmented at sentence ends and pauses
llm_extraction_service/   live ePCR, extracted per segment (prompts/, terminology_linker.py, terminology/index)
video_service/            SAM 3 video segmentation
dashboard/                web dashboard of a session's ePCR timeline (server.py, epcr.py, static/)
common/                   shared WorkerService, StreamingModel, config, logging, file helpers
tests/                    unit tests (unit/), mock streaming client (e2e/), per-service integration tests (integration/), their Dockerfile
sessions/                 one folder per session (created at runtime)
logs/                     one log file per service
```

## Dashboard

The `dashboard` service serves a web page of a session's ePCR timeline at
`http://<jetson-ip>:8080/` (`DASHBOARD_PORT`). It reads the extraction service's output in
`extractions/` (each `<category>.json`, `medication_administrations.json` and `progress.json`)
and re-reads it every 2 s, so the page follows a session as it is recorded.
`dashboard/epcr.py` builds the page's ePCR from them, without a model: each item is placed at
its earliest quote, demographics show their latest statement and the mechanism of injury its
latest description, and medications come from their administrations, so an infusion is a bar
from when it was started to when it was stopped (or to the latest segment while it runs).
It only reads the sessions folder, and the page loads nothing from the internet.

- **Header:** sex and age (`demographics`), the mechanism of injury with when it was reported
  and its first quote, and the session's state, from its done markers: **Recording** until
  the server marks both recordings complete, then **Processing** until every service the
  session is for (its `session.json`) marks its output complete, then **Done**. A session
  still recording or processing whose files haven't changed for `DASHBOARD_LIVE_WINDOW`
  seconds shows as **Stalled** (a service may be down). A marker whose status isn't `ok`
  (`failed`, `abandoned`, `aborted`) is added to the badge, e.g. *Done · extraction failed*,
  and turns it amber; its tooltip lists what the session still waits for and every problem.
  Only folders with a `session.json` are listed.
- **Download** (top right): once the session has stopped recording, downloads
  `session-<id>.zip` with the recordings the server exported (`audio.mp3`, `video.mp4`;
  whichever exist). Until then the button is disabled, and its tooltip says why. The server
  writes the exports atomically (`<name>.tmp.<ext>`, then renamed), so the zip never holds
  a half-written file.
- **Injuries:** the complaints, findings and suspected injuries of `patient_condition`,
  numbered. A front and back body map marks each where its name or body site names a body
  part (*left side of the neck*, *jaw*, *chest*…), with a numbered badge per spot; a chip per
  injury names its locations, and the list below gives every one. An item that names no body
  part has no spot, and a side that isn't named marks both. Clicking a badge, chip or list
  item finds it on the timeline. The bottom half of the panel shows the wound images a model
  predicts into the session's `video/wounds/*.jpg` (or `*.png`), one at a time, in video order (by the
  timestamp in the name, `000012.340.jpg`), with how far into the video each was taken. The
  **Boxes** switch draws each image's bounding boxes (its `.box` file) over it. The latest
  shows by default and new ones replace it as they arrive; the left and right arrows (or the
  arrow keys) step through the rest, and clicking an image opens it full size.
- **Medical history** (right column, top): the medical history grouped by kind (patient
  history, comorbidity, family history), each item's quotes with when it was first mentioned;
  hover or focus one for every time it was mentioned. It isn't on the timeline.
- **Latest vital signs** (right column, bottom): the latest reading of each vital sign (BP,
  MAP, HR, SpO₂, RR, EtCO₂, temperature, GCS, glucose, …) as a tile with when it was read;
  hover or focus one for its quotes.
- **Patient timeline:** four lanes on one time axis. *Patient condition* is the patient's
  condition (complaints, mental status, findings, suspected injuries),
  *Vital signs* the vital-sign readings, grouped into one event per set of readings taken
  together (a reading within 20 s of the last, until a vital repeats), *Medications* the medication administrations (a bolus is an event, an infusion a bar from
  start to stop, with a dot at each rate change), *Interventions* the interventions. Times are elapsed from the
  start of the recording. Hover or focus an event for its details, every time it was
  mentioned and the transcript quotes; a lane labels at most three rows of events, and the
  rest show as dots with the same details.
- **Controls:** the session picker (by default the newest live session, else the newest with
  an extraction; `?session=<id>` in the URL picks one), the visible span (10 min, 20 min, 1 h,
  All), and **Follow**, which keeps the latest events in view and turns off when you scroll
  back.

The server is `dashboard/server.py` (standard library only); the page is `dashboard/static/`.
Its JSON is also available directly: `GET /api/sessions` lists the sessions, and
`GET /api/sessions/<id>` returns one session with its ePCR (`epcr`) and its wound images
(`wounds`: `[{name, time, seconds, url, boxes}]`), each served at `GET /api/sessions/<id>/wounds/<name>`
(`.jpg`, `.jpeg` or `.png`, and the `.box` file of its bounding boxes at `boxes`).

## Testing the live stack

`tests/e2e/main.py` is a mock client that streams test media to the server like the phone app
would. It starts a session, publishes `tests/integration/asr/inputs/audio.mp3` and/or
`tests/integration/video/inputs/video.mp4` (the integration tests' media) to the server's RTSP
URL in real time, then stops the session. It runs in its own container (`tests/Dockerfile`),
because the Jetson's own ffmpeg can't publish RTSP or decode MP3.

Unlike the [integration tests](#integration-tests), this goes through the server to every
service at once, so it is the one test of the handoffs between them (ASR to extraction). Start the stack and build the test image first ([Setup](#setup), step 2).

### Run the client

```bash
docker compose run --rm e2e_test --mode both    # .mp4 video + .mp3 audio (default)
docker compose run --rm e2e_test --mode audio   # only the .mp3
docker compose run --rm e2e_test --mode video   # only the .mp4
```

Each run starts a session (logging its `session_id`; the client warns if a service didn't
accept it), streams in real time (each file plays to its own end, so `both` lasts as long as
the longer file), waits 2 s (`--linger`) and stops the session. Ctrl-C stops early and still ends the session cleanly. The server
takes one client and one session at a time, so run one at a time.

`--audio-file` and `--video-file` stream other media (paths inside the container); `--dry-run`
prints the ffmpeg command without contacting the server; `--help` lists every option.
`tests/e2e/` and `tests/integration/` are mounted into the container (at `/tests/e2e/` and
`/tests/integration/`), so edits to the client or the media need no rebuild.
The client's output, ffmpeg's included, goes to `logs/tests/e2e-<YYYYmmdd-HHMMSS>.log`, not the
console, which only shows the log's path; `--test-log-dir` (or `$TEST_LOG_DIR`) puts it elsewhere.

### Check the results

Everything lands in `sessions/<session_id>/`:

| While streaming | After `stop_session` |
|---|---|
| `audio/audio.pcm` grows | `audio/audio.pcm.done`, then `audio/audio.mp3` |
| `transcript/transcript.jsonl`, `segments.json` | the last segment, then `segments.json.done` |
| `video/video.ts` grows, plus the video service's output | `video/video.ts.done`, the last frames, `video/wounds.done`, then `video/video.mp4` |
| `extractions/<category>.json`, `medication_administrations.json` and `progress.json`, updated as each segment closes | the last segment's update, then `extractions.done` |

```bash
ls -R sessions/<session_id>
tail -f sessions/<session_id>/transcript/transcript.jsonl # live transcript, one fragment per line
tail -f logs/*.log                                        # service logs
```

The ASR closes a segment at the first sentence end after 20 s of audio
(`ASR_SEGMENT_MIN_LENGTH`), or after 20 s of silence (`ASR_SEGMENT_PAUSE`), so
`segments.json` and the `extractions/` files may stay empty until a short stream ends.

### Shorter runs with clips

The full inputs take minutes per run. To cut 20 s clips (using the test image's ffmpeg)
and stream those instead:

```bash
mkdir -p clips
docker compose run --rm -v "$PWD/clips:/out" --entrypoint sh e2e_test -c \
  'ffmpeg -v error -y -i integration/asr/inputs/audio.mp3 -t 20 -c copy /out/audio.mp3 &&
   ffmpeg -v error -y -i integration/video/inputs/video.mp4 -t 20 -c copy /out/video.mp4'

docker compose run --rm -v "$PWD/clips:/clips:ro" e2e_test --mode both \
  --audio-file /clips/audio.mp3 --video-file /clips/video.mp4
```

### From another machine

On the Jetson's network, with Python 3 and a full ffmpeg installed:

```bash
python tests/e2e/main.py --host <jetson-ip> --mode both
```

It uses only the standard library and finds the media under `tests/integration/` itself, so
run it from a checkout of this repository (or pass `--audio-file` / `--video-file`).

## Unit tests

`tests/unit/` tests the services' logic without models or GPUs: the `WorkerService` session
loop, the extraction and video services (with fake models), the extraction model's and Sam3's
per-session steps (with the model stubbed), and the video service's `FrameReader`. They need
Python, pytest, numpy and Pillow, so they run on the host:

```bash
python3 -m pytest tests/unit
```

The `FrameReader` tests decode real H.264 with ffmpeg, which the host's minimal ffmpeg can't, so
they skip there; run them in the video service's image (no GPU needed):

```bash
docker run --rm -v "$PWD":/repo -w /repo --entrypoint python3 \
    jetson_model_framework-video_service -m pytest -p no:cacheprovider tests/unit
```

## Integration tests

Each model service has an integration test in `tests/integration/<test>/`: `main.py` (the test),
`inputs/` (what it feeds the service) and `ground_truth/` (what the output is compared
against). A test drives one service on its own, without the server. It creates a session
folder the way the server does, with a `session.json` naming only that service, so the
others leave it alone. It then writes the service's input into it as it would arrive live,
marks the input complete, waits for the service to finish and checks its output. Shared code is in `tests/integration/harness.py`.

| Command | Service | Feeds it | Checks |
|---|---|---|---|
| `docker compose run --rm asr_test` | `asr_service` | `asr/inputs/audio.mp3`, in real time | transcript word error rate against the ground truth (at most `--max-wer`, 0.15 by default) and segment boundaries |
| `docker compose run --rm video_test` | `video_service` | `video/inputs/video.mp4`, in real time | `video/wounds/*.jpg` frames were written, each with a `.box` file of bounding boxes |
| `docker compose run --rm server_test` | `server` | `asr/inputs/audio.mp3` and `video/inputs/video_short.mp4` (looped), published over RTSP in real time for 6 s at a time (`--seconds`), as the phone app does | errors for an unknown command, a line that isn't JSON and JSON that isn't an object, with the connection kept; `busy` for a second client; for a session: the `start_session` reply, `session.json`, the subfolders and the MediaMTX path, then after `stop_session` about 6 s of 16 kHz audio in `audio.pcm` and of H.264 in `video.ts`, both marked `ok`, `audio.mp3` and `video.mp4` exported, the path removed; a publisher that reconnects appended to the same files; `start_session` during a session ending that one first; `stop_session` without a session failing |
| `docker compose run --rm llm_extraction_test` | `llm_extraction_service` | `extraction/inputs/segments.json`, one segment every 5 s (`--interval`) | a `<category>.json` of records per category (each for an input segment, in order, not empty), every quote time inside its record's segment, a unique id for each medication administration and an earlier one for each update, `medication_administrations.json` with each once, `progress.json` naming the last segment, `extractions.done`; then a report of the extractions' score against `extraction/ground_truth/` |

### Run a test

```bash
docker compose build e2e_test                  # once: builds the test image every test uses
docker compose up -d llm_extraction_service    # the service under test
docker compose logs -f llm_extraction_service  # wait for "waiting for a session"
docker compose run --rm llm_extraction_test
```

The test services reuse the image the `e2e_test` build makes, so rebuild it after changing
`tests/Dockerfile`; `tests/integration/` is mounted into the container, so edits to the tests
need no rebuild. The model services' tests don't need the server; a service that is busy with
a real session finishes it before it picks up the test's.

`server_test` is the other way round: it drives the server like the phone app and needs only
the server (`docker compose up -d server`), no GPU and no model services. Don't run it during
a real session, since its `start_session` ends the active one. The server's sessions name
every model service, so any that are running also work on the test's sessions (a few seconds
of media each). Its sessions stay in `sessions/<id>/`; their ids are in its log.

A test exits non-zero if any check fails. Its output goes to a log file, not the console (which
only shows the file's path): `logs/tests/<service>-<YYYYmmdd-HHMMSS>.log`, one per run, with each
check as `[PASS]` or `[FAIL]`, the score reports and ffmpeg's warnings. `--test-log-dir` (or
`$TEST_LOG_DIR`) puts it elsewhere. The logger is in `tests/integration/testlog.py`. Its
session stays in `sessions/it-<id>/` for inspection. Model weights are downloaded on the
service's first start, so allow for that before the first run.

### Options

Options go after the test's name, e.g. `docker compose run --rm llm_extraction_test --interval 0`.

| Test | Option | Effect |
|---|---|---|
| all but `server_test` | `--dry-run` | print what would be fed to the service, without contacting it |
| all | `--test-log-dir <dir>` | where to write the run's log (default: `logs/tests`) |
| all | `--timeout <s>` | how long to wait for the service to finish after the session ends (`server_test`: for each of the server's replies) |
| all | `--help` | every option |
| `asr_test` | `--max-wer <x>` | the highest word error rate that passes |
| `asr_test` | `--audio-file <path>` | another clip (skips the ground-truth checks) |
| `video_test` | `--video-file <path>` | another clip |
| `llm_extraction_test` | `--interval <s>` | seconds between segments; `0` writes them all at once |
| `llm_extraction_test` | `--score /sessions/<id>` | only log the score report for an earlier session; the service isn't contacted |
| `llm_extraction_test` | `--ground-truth <dir>` | score against other ground truth |
| `server_test` | `--seconds <s>` | how long each publish lasts (default 6) |
| `server_test` | `--host <ip>`, `--port <n>` | where the server's control port is (default `127.0.0.1:8000`) |

Paths are inside the test container, where `tests/integration/` is at `/tests/integration/`
and the sessions folder at `/sessions/`.

### The extraction score

The model's extractions aren't expected to reproduce the ground truth exactly, so the score is
a report and never fails the test. It compares all the extracted items with the ground truth,
`ground_truth/<category>.json`, which holds records in the same shape as the service's output.
An item matches another of the same category when their quotes share at least half their words
(preferring one with the same entity, since one utterance often gives several items); an item
without quotes matches one with the same entity instead. The score counts
coverage: recall is the share of ground-truth items some extracted item matches, precision the
share of extracted items that match some ground-truth item. For each category it also prints
how often a matched pair agrees on the entity and on the value (vital signs, demographics) or
dose (medications).

### What they don't cover

The integration tests check one service each. The handoff between the services (ASR to
extraction) is only exercised by the full stack: see
[Testing the live stack](#testing-the-live-stack).
