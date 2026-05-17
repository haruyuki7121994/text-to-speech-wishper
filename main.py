import argparse
import tempfile
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from lightning_whisper_mlx import LightningWhisperMLX


SAMPLE_RATE = 16_000
DEFAULT_MODEL = "distil-medium.en"
DEFAULT_CHUNK_SECONDS = 4.0
DEFAULT_SILENCE_THRESHOLD = 0.01


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time speech-to-text from your microphone using Lightning Whisper MLX."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model name to load.")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help="How many seconds of microphone audio to transcribe at a time.",
    )
    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=DEFAULT_SILENCE_THRESHOLD,
        help="Skip chunks below this RMS volume. Lower it if quiet speech is ignored.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Optional sounddevice input device id. Run `python -m sounddevice` to list devices.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size passed to LightningWhisperMLX.",
    )
    return parser.parse_args()


def write_wav(path: Path, audio: np.ndarray, sample_rate: int):
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def record_chunk(seconds: float, device: int | None):
    frames = int(seconds * SAMPLE_RATE)
    recording = sd.rec(
        frames,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    return recording.reshape(-1)


def transcribe_chunk(whisper: LightningWhisperMLX, audio: np.ndarray):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        write_wav(temp_path, audio, SAMPLE_RATE)
        result = whisper.transcribe(audio_path=str(temp_path))
        return result.get("text", "").strip()
    finally:
        temp_path.unlink(missing_ok=True)


def run_realtime_transcriber():
    args = parse_args()
    whisper = LightningWhisperMLX(model=args.model, batch_size=args.batch_size, quant=None)

    print("Real-time speech-to-text is running.")
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    try:
        while True:
            audio = record_chunk(args.chunk_seconds, args.device)
            rms = float(np.sqrt(np.mean(np.square(audio))))

            if rms < args.silence_threshold:
                print("...", flush=True)
                continue

            text = transcribe_chunk(whisper, audio)
            if text:
                print(text, flush=True)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    run_realtime_transcriber()
