#!/usr/bin/env python3
import os
import sys
import subprocess
import json

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")

def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)

def ensure_dependencies():
    # Upgrade pip
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Bersihkan dependency konflik
    run([
        sys.executable, "-m", "pip", "uninstall", "-y",
        "torch", "torchaudio", "torchvision",
        "pyannote.audio"
    ])

    # Install torch + torchaudio yang KOMPATIBEL (CPU, stabil di CI)
    run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])

    # Install WhisperX
    run([
        sys.executable, "-m", "pip", "install",
        "whisperx"
    ])

def find_audio_files():
    return [
        f for f in os.listdir(".")
        if f.lower().endswith(AUDIO_EXTENSIONS)
    ]

def transcribe(audio_file):
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # LOAD MODEL (MULTILINGUAL, MEDIUM, SILERO VAD)
    model = whisperx.load_model(
        "large-v3",
        device=device,
        compute_type=compute_type,
        language=None,
        vad_method="silero"  # <<< PENTING: bypass pyannote
    )

    result = model.transcribe(audio_file)

    base = os.path.splitext(audio_file)[0]
    json_path = f"{base}.json"

    # SIMPAN JSON KE FILE
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # PRINT RAW JSON KE STDOUT (BIAR KELIHATAN DI ACTIONS LOG)
    print("\n" + "=" * 100)
    print(f"RAW JSON OUTPUT :: {json_path}")
    print("=" * 100)
    with open(json_path, "r", encoding="utf-8") as f:
        print(f.read())
    print("=" * 100 + "\n")

def main():
    ensure_dependencies()

    audio_files = find_audio_files()
    if not audio_files:
        print("Tidak ada file audio ditemukan.")
        return

    for audio in audio_files:
        transcribe(audio)

if __name__ == "__main__":
    main()
