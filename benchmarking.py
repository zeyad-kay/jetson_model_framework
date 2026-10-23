"""Transcribe `live_audio.pcm` in near-real-time while `mic_record.py` writes to it.

Reads the raw PCM file written by mic_record.py, staying DELAY_SECONDS behind
the writer so it only ever reads bytes that have already been flushed to disk.
"""

import os
import queue
import signal
import time
from threading import Event, Thread

import numpy as np
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

stop_event = Event()


class StopStreaming(Exception):
    """Raised inside the audio reader when a shutdown was requested."""


def _request_stop(signum, frame):
    if stop_event.is_set():
        # Second signal: user really wants out, don't wait for cleanup.
        raise KeyboardInterrupt
    print("\nInterrupt received, shutting down...", flush=True)
    stop_event.set()


signal.signal(signal.SIGINT, _request_stop)
signal.signal(signal.SIGTERM, _request_stop)






def find_mp3_audio_start(file_path):
    with open(file_path, 'rb') as f:
        # 1. Read the 10-byte ID3v2 header
        header = f.read(10)
        if len(header) < 10:
            return 0
        
        # 2. Check for ID3v2 tag and calculate its size
        if header[:3] == b'ID3':
            # Decode the 4-byte synchsafe integer (7 bits per byte)
            size_bytes = header[6:10]
            tag_size = (size_bytes[0] << 21) | (size_bytes[1] << 14) | (size_bytes[2] << 7) | size_bytes[3]
            
            # The total size to skip is the 10-byte header + the tag's payload
            total_id3_size = 10 + tag_size
            f.seek(total_id3_size)
        else:
            # No ID3v2 tag found, start searching from the beginning
            f.seek(0)
            
        # 3. Scan forward for the 11-bit or 12-bit MP3 sync word (0xFFE or 0xFFF)
        # Usually checking for 0xFF followed by 0xE0 or higher works best
        while True:
            current_pos = f.tell()
            byte1 = f.read(1)
            if not byte1:
                break  # End of file reached
                
            if byte1 == b'\xff':
                byte2 = f.read(1)
                if not byte2:
                    break
                    
                # MP3 frames start with 11 or 12 bits of 1s (0xFF and 0xE0+ or 0xF0+)
                if (byte2[0] & 0xE0) == 0xE0:
                    return current_pos  # Found the first audio frame offset
                    
                # If byte2 didn't match, back up 1 byte to check it as a potential 0xFF
                f.seek(current_pos + 1)
                
    return 0  # Fallback to start if no valid header is found

PCM_PATH = "output.wav"
# PCM_PATH = "C:\\Users\\zeyad\\Downloads\\mediamtx_v1.21.2_windows_amd64\\output.pcm"
# PCM_PATH = "live_audio_corrupt.pcm"
DELAY_SECONDS = 2.0
BYTES_PER_SAMPLE = 2  # mono, int16
# BYTES_PER_SAMPLE = 4  # mono, f32

model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForRNNT.from_pretrained(model_id).cuda()
print(model.device)
processor.set_num_lookahead_tokens(6)
print(f"Streaming latency: {processor.streaming_latency_ms} ms")

language = "en-US"
sampling_rate = processor.feature_extractor.sampling_rate
delay_samples = int(DELAY_SECONDS * sampling_rate)




def wait_for_samples(end_sample: int) -> None:
    """Block until the PCM file has `end_sample` samples available to read,
    honoring the DELAY_SECONDS safety margin behind the writer."""
    needed_bytes = (end_sample + delay_samples) * BYTES_PER_SAMPLE
    # print(os.path.getsize(PCM_PATH) < needed_bytes)
    while not os.path.exists(PCM_PATH) or os.path.getsize(PCM_PATH) < needed_bytes:
        if stop_event.is_set():
            raise StopStreaming
        time.sleep(0.1)


def read_samples(f, start_sample: int, num_samples: int) -> np.ndarray:
    wait_for_samples(start_sample + num_samples)
    f.seek(start_sample * BYTES_PER_SAMPLE)
    data = f.read(num_samples * BYTES_PER_SAMPLE)
    # return np.frombuffer(data, dtype=np.float32)
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


print("Waiting for recording to start...")
while not os.path.exists(PCM_PATH):
    if stop_event.is_set():
        raise SystemExit(0)
    time.sleep(0.1)
pcm_file = open(PCM_PATH, "rb")


start_byte = 44
try:
    first_chunk_audio = read_samples(pcm_file, start_byte, processor.num_samples_first_audio_chunk)
except StopStreaming:
    pcm_file.close()
    print("Stopped.")
    raise SystemExit(0)

first_chunk_inputs = processor(
    first_chunk_audio,
    sampling_rate=sampling_rate,
    is_streaming=True,
    is_first_audio_chunk=True,
    language=language,
    return_tensors="pt",
)
first_chunk_inputs = first_chunk_inputs.to(model.device, dtype=model.dtype)


def input_features_generator():
    yield first_chunk_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]

    mel_frame_idx = processor.num_mel_frames_first_audio_chunk
    hop_length = processor.feature_extractor.hop_length
    n_fft = processor.feature_extractor.n_fft

    start_idx = mel_frame_idx * hop_length - n_fft // 2
    while not stop_event.is_set():
        end_idx = start_idx + processor.num_samples_per_audio_chunk
        try:
            chunk_audio = read_samples(pcm_file, max(start_idx, 0), end_idx - max(start_idx, 0))
        except StopStreaming:
            return

        inputs = processor(
            chunk_audio,
            sampling_rate=sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=False,
            language=language,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device, dtype=model.dtype)
        yield inputs.input_features

        mel_frame_idx += processor.num_mel_frames_per_audio_chunk
        start_idx = mel_frame_idx * hop_length - n_fft // 2


streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, timeout=0.5)
generate_kwargs = {
    **first_chunk_inputs,
    "input_features": input_features_generator(),
    "streamer": streamer,
}
thread = Thread(target=model.generate, kwargs=generate_kwargs, daemon=True)
thread.start()

print("Model output (streaming):", end=" ", flush=True)
try:
    stream_iter = iter(streamer)
    while not stop_event.is_set():
        try:
            text_chunk = next(stream_iter)
        except queue.Empty:
            continue  # no new text yet; re-check the stop flag
        except StopIteration:
            break
        print(text_chunk, end="", flush=True)
finally:
    stop_event.set()
    thread.join(timeout=5)
    pcm_file.close()
    print("\nStopped.")
