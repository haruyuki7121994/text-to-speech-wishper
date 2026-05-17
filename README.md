# Real-Time Speech to Text

This project records audio from your microphone and transcribes it continuously with `lightning-whisper-mlx`.

## Requirements

Install Python packages:

```sh
pip install -r requirements.txt
```

Install `ffmpeg` on macOS:

```sh
brew install ffmpeg
```

`lightning-whisper-mlx` uses `ffmpeg` internally to load audio, so the app will not run without it.

## Run

```sh
python main.py
```

Stop the transcriber with `Ctrl+C`.

## Useful Options

Transcribe smaller chunks for lower latency:

```sh
python main.py --chunk-seconds 2
```

Lower the silence threshold if quiet speech is skipped:

```sh
python main.py --silence-threshold 0.005
```

Choose a specific microphone input device:

```sh
python -m sounddevice
python main.py --device 1
```
