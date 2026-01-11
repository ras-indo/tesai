#!/usr/bin/env python3
import os
import sys
import subprocess
import json

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")

def run(cmd):
    subprocess.check_call(cmd)

def ensure_dependencies():
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([
        sys.executable, "-m", "pip", "install",
        "torch",
        "torchaudio",
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

    model = whisperx.load_model(
        "medium",
        device=device,
        compute_type=compute_type,
        language=None
    )

    result = model.transcribe(audio_file)

    if "segments" in result and result.get("language"):
        align_model, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )
        result = whisperx.align(
            result["segments"],
            align_model,
            metadata,
            audio_file,
            device
        )

    base = os.path.splitext(audio_file)[0]
    json_path = f"{base}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # === PRINT RAW JSON KE STDOUT ===
    print("\n" + "=" * 80)
    print(f"RAW JSON OUTPUT: {json_path}")
    print("=" * 80)
    with open(json_path, "r", encoding="utf-8") as f:
        print(f.read())
    print("=" * 80 + "\n")

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
