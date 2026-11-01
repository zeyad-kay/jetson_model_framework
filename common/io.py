import json
import logging
import subprocess
from pathlib import Path
from typing import Union


class TranscriptWriter:
    """Writes transcript text from an ASR model to a text file as it is generated.

    Each piece of text is followed by `separator` and flushed immediately, so
    the file always reflects everything transcribed so far.
    """

    def __init__(
        self, path: Union[str, Path], separator: str = "\n", append: bool = False
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.separator = separator
        self._file = open(self.path, "a" if append else "w", encoding="utf-8")

    def write(self, text: str) -> None:
        self._file.write(text + self.separator)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TranscriptWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


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


def pcm_to_mp3(
    pcm_path: Union[str, Path],
    mp3_path: Union[str, Path],
    log: logging.Logger,
    sample_rate: int = 16000,
) -> bool:
    """Converts raw 16-bit mono PCM to an mp3. True on success; failures are logged to `log`."""
    pcm_path = Path(pcm_path)
    if not pcm_path.exists() or pcm_path.stat().st_size == 0:
        log.warning("no audio to convert at %s", pcm_path)
        return False
    result = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "s16le", "-ac", "1", "-ar", str(sample_rate),
            "-i", str(pcm_path), str(mp3_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        log.warning("pcm to mp3 failed: %s", result.stderr.decode(errors="replace").strip())
        return False
    return True


def ts_to_mp4(ts_path: Union[str, Path], mp4_path: Union[str, Path], log: logging.Logger) -> bool:
    """Converts raw mpeg-ts to an mp4. True on success; failures are logged to `log`."""
    ts_path = Path(ts_path)
    if not ts_path.exists() or ts_path.stat().st_size == 0:
        log.warning("no video to convert at %s", ts_path)
        return False
    result = subprocess.run(
        [
            "ffmpeg", "-loglevel", "error", "-y",
            "-i", str(ts_path), str(mp4_path),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        log.warning("ts to mp4 failed: %s", result.stderr.decode(errors="replace").strip())
        return False
    return True
