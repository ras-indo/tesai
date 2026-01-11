#!/usr/bin/env python3
"""
WHISPERX + PYANNOTE DIARIZATION - Fixed Version
Compatible PyTorch + Pyannote.audio + WhisperX
"""
import os
import sys
import subprocess
import json
from datetime import timedelta

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")

def run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.check_call(cmd)

def ensure_dependencies():
    """Install dependencies versi kompatibel"""
    print("📦 Installing compatible dependencies...")

    # Upgrade pip
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Uninstall versi lama / konflik
    run([sys.executable, "-m", "pip", "uninstall", "-y",
         "torch", "torchaudio", "torchvision", "pyannote.audio", "whisperx"])

    # Install versi stabil
    run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])

    run([
        sys.executable, "-m", "pip", "install",
        "whisperx==3.1.1",
        "pyannote.audio==3.1.1",
        "huggingface-hub"
    ])
    print("✅ Dependencies installed")

def setup_huggingface():
    """Setup Hugging Face token dari environment"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN tidak ditemukan")
        sys.exit(1)

    os.environ["HF_TOKEN"] = hf_token
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Logged in to Hugging Face Hub")
    except Exception as e:
        print(f"❌ Hugging Face login failed: {e}")
        sys.exit(1)

def find_audio_files():
    return [f for f in os.listdir(".") if f.lower().endswith(AUDIO_EXTENSIONS)]

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours = int(td.seconds // 3600)
    minutes = int((td.seconds % 3600) // 60)
    seconds = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def transcribe_with_diarization(audio_file):
    import whisperx
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"\n🎯 Processing: {audio_file} | Device: {device} | Compute: {compute_type}")

    # Load whisper model
    model = whisperx.load_model(
        "medium",
        device=device,
        compute_type=compute_type,
        language=None,  # auto-detect
        vad_parameters={"vad_onset": 0.5, "vad_offset": 0.363}
    )

    # Load audio & transcribe
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=8)

    # Alignment
    model_a, metadata = whisperx.load_align_model(
        language_code=result["language"],
        device=device
    )
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    # Diarization
    print("   👥 Diarizing speakers...")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=os.getenv("HF_TOKEN"),
        device=device
    )
    diarize_segments = diarize_model(audio)

    # Assign speaker to words
    result = whisperx.assign_word_speakers(diarize_segments, result)

    return result, audio_file

def save_outputs(result, audio_file):
    base = os.path.splitext(audio_file)[0]

    # JSON
    json_path = f"{base}_diarized.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # TXT
    txt_path = f"{base}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Audio: {audio_file}\nLanguage: {result.get('language', 'unknown')}\n")
        f.write("="*50 + "\n\n")
        for seg in result.get("segments", []):
            start, end = seg.get("start", 0), seg.get("end", 0)
            speaker, text = seg.get("speaker", "UNKNOWN"), seg.get("text", "").strip()
            f.write(f"[{format_timestamp(start)} → {format_timestamp(end)}] Speaker {speaker}:\n{text}\n\n")

    # SRT
    srt_path = f"{base}_subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.get("segments", []), 1):
            start, end = seg.get("start", 0), seg.get("end", 0)
            speaker, text = seg.get("speaker", "UNKNOWN"), seg.get("text", "").strip()
            f.write(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n[Speaker {speaker}] {text}\n\n")

    print(f"📁 Outputs: {json_path}, {txt_path}, {srt_path}")
    return json_path, txt_path, srt_path

def main():
    print("🚀 WhisperX + Pyannote Diarization (Fixed)")
    ensure_dependencies()
    setup_huggingface()

    audio_files = find_audio_files()
    if not audio_files:
        print("⚠️  No audio files found")
        return

    print(f"🎯 Found {len(audio_files)} audio file(s): {audio_files}")
    all_outputs = []
    for audio in audio_files:
        try:
            result, _ = transcribe_with_diarization(audio)
            outputs = save_outputs(result, audio)
            all_outputs.append(outputs)
        except Exception as e:
            print(f"❌ Error processing {audio}: {type(e).__name__}: {e}")
            continue

    print(f"✅ PROCESSING COMPLETE - {len(all_outputs)} file(s)")

if __name__ == "__main__":
    main()
