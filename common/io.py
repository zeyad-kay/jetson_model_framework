import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Union


class TimedTranscriptWriter:
    """Writes timestamped transcript text as JSON lines: {"t": seconds, "text": ...}.

    t is the position in the audio, in seconds from its start. Each line is
    flushed immediately.
    """

    def __init__(self, path: Union[str, Path], append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "a" if append else "w", encoding="utf-8")

    def write(self, text: str, seconds: float) -> None:
        self._file.write(json.dumps({"t": round(seconds, 3), "text": text}, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TimedTranscriptWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_text(path: Union[str, Path], text: str) -> None:
    """Writes text, replacing the file atomically so readers never see a partial write."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_json(path: Union[str, Path], obj) -> None:
    """Writes obj as JSON, replacing the file atomically so readers never see a partial write."""
    write_text(path, json.dumps(obj, indent=1, ensure_ascii=False))


def done_marker(path: Union[str, Path]) -> Path:
    """The marker that says `path` is complete: <path>.done next to it (audio.pcm -> audio.pcm.done).

    Its writer creates it once it will write `path` no more; whoever reads `path` as it
    grows can then read the rest and stop. Check the marker before reading `path`, so
    the last write before the marker isn't missed.
    """
    path = Path(path)
    return path.with_name(path.name + ".done")


def write_done(path: Union[str, Path], status: str = "ok") -> None:
    """Marks `path` (a file or a folder) complete: writes done_marker(path) as {"status": status}.

    status is "ok", or why it ended otherwise ("failed"). Written even after a failure, so
    whoever waits for `path` stops waiting.
    """
    write_json(done_marker(path), {"status": status})


def write_jpeg(path: Union[str, Path], image, quality: int = 90) -> None:
    """Writes an HxWx3 uint8 RGB array as a JPEG, replacing the file atomically so readers never see a partial write."""
    from PIL import Image  # only the services that write images need Pillow

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(image).save(tmp, format="JPEG", quality=quality)
    os.replace(tmp, path)


def _ffmpeg_to(output: Path, args: list[str], what: str, log: logging.Logger) -> bool:
    """Runs ffmpeg with `args` (its input options), writing to `output` atomically: to
    <stem>.tmp<suffix> (ffmpeg picks the format from the suffix), renamed over `output` once
    ffmpeg succeeds, so whoever sees `output` sees all of it. True on success; a failure is
    logged to `log`, and leaves no partial file."""
    output = Path(output)
    tmp = output.with_name(f"{output.stem}.tmp{output.suffix}")
    result = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", *args, str(tmp)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        log.warning("%s failed: %s", what, result.stderr.decode(errors="replace").strip())
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, output)
    return True


def pcm_to_mp3(
    pcm_path: Union[str, Path],
    mp3_path: Union[str, Path],
    sampling_rate: int,
    log: logging.Logger,
) -> bool:
    """Converts raw 16-bit mono PCM to an mp3, written atomically. True on success; failures are logged to `log`."""
    pcm_path = Path(pcm_path)
    if not pcm_path.exists() or pcm_path.stat().st_size == 0:
        log.warning("no audio to convert at %s", pcm_path)
        return False
    return _ffmpeg_to(Path(mp3_path), ["-f", "s16le", "-ac", "1", "-ar", str(sampling_rate), "-i", str(pcm_path)],
                      "pcm to mp3", log)


def ts_to_mp4(ts_path: Union[str, Path], mp4_path: Union[str, Path], log: logging.Logger) -> bool:
    """Converts raw mpeg-ts to an mp4, written atomically. True on success; failures are logged to `log`."""
    ts_path = Path(ts_path)
    if not ts_path.exists() or ts_path.stat().st_size == 0:
        log.warning("no video to convert at %s", ts_path)
        return False
    return _ffmpeg_to(Path(mp4_path), ["-i", str(ts_path)], "ts to mp4", log)
