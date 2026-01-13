#!/usr/bin/env python3
"""
FASTER-WHISPER + PYANNOTE DIARIZATION - FINAL STABLE VERSION
"""

import os
import sys
import subprocess
import json
from datetime import timedelta

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")

def run(cmd):
    """Helper: run shell command dengan logging"""
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
    result.check_returncode()
    return result

def ensure_dependencies():
    """Install dependencies dengan urutan yang benar dan pre-flight check"""
    print("📦 Installing and verifying all dependencies...")

    # 1. Core packages
    core_packages = ["regex", "numpy<2", "packaging", "coloredlogs", "flatbuffers", "protobuf"]
    for pkg in core_packages:
        run([sys.executable, "-m", "pip", "install", pkg])

    # 2. PyTorch CPU compatible
    run([sys.executable, "-m", "pip", "install",
         "torch==2.9.1", "torchaudio==2.9.1", "--index-url", "https://download.pytorch.org/whl/cpu"])

    # 3. pyannote.audio + deps
    pyannote_deps = ["pyannote.core==5.0.0", "pyannote.audio==4.1.1"]
    for pkg in pyannote_deps:
        run([sys.executable, "-m", "pip", "install", pkg])

    # 4. Faster Whisper + other NLP/audio deps
    other_packages = [
        "faster-whisper",
        "librosa==0.10.0",
        "soundfile==0.12.1",
        "pandas==2.0.3",
        "nltk==3.8.1",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "tqdm==4.66.1",
        "numba==0.58.1",
        "jiwer==3.0.3",
        "huggingface-hub==0.20.3",
        "ffmpeg-python==0.2.0",
        "safetensors",
        "tokenizers"
    ]
    for pkg in other_packages:
        run([sys.executable, "-m", "pip", "install", pkg])

    # 5. Download NLTK punkt
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("✅ NLTK punkt downloaded")
    except Exception as e:
        print(f"⚠️ NLTK warning: {e}")

    # 6. Pre-flight check
    required_modules = ["numpy", "torch", "faster_whisper", "regex", "pyannote.audio"]
    for m in required_modules:
        try:
            __import__(m)
            print(f"✅ {m} loaded")
        except Exception as e:
            print(f"❌ Failed to import {m}: {e}")

def setup_huggingface():
    """Setup Hugging Face token dari environment"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN tidak ditemukan di environment")
        sys.exit(1)
    os.environ["HF_TOKEN"] = hf_token
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print("✅ Logged in to Hugging Face Hub")

def find_audio_files():
    """Auto-detect file audio"""
    return [f for f in os.listdir(".") if f.lower().endswith(AUDIO_EXTENSIONS)]

def format_timestamp(seconds):
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def transcribe_with_diarization(audio_file):
    """Transcribe dengan faster-whisper dan pyannote audio"""
    import torch
    from faster_whisper import WhisperModel
    from pyannote.audio import Pipeline

    print(f"\n🎯 Processing: {audio_file}")
    device = "cpu"

    # 1. Load faster-whisper
    print("📥 Loading Faster Whisper model...")
    model = WhisperModel("large-v3", device=device, compute_type="int8")

    # 2. Transcribe
    print("📝 Transcribing...")
    segments, _ = model.transcribe(audio_file, beam_size=5)
    result = {"segments": [], "language": "id"}  # default language id
    for s in segments:
        result["segments"].append({
            "start": s.start,
            "end": s.end,
            "text": s.text
        })

    # 3. Diarization
    print("👥 Running speaker diarization...")
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                            use_auth_token=os.getenv("HF_TOKEN"))
        diarization = pipeline(audio_file)
        result_segments = []
        for seg, _, speaker in diarization.itertracks(yield_label=True):
            for r in result["segments"]:
                if seg.start <= r["start"] <= seg.end:
                    r["speaker"] = speaker
                    result_segments.append(r)
        result["segments"] = result_segments
        print(f"✅ Diarization complete: {len(diarization)} segments")
    except Exception as e:
        print(f"⚠️ Diarization failed: {e}")
        # fallback speaker
        for r in result["segments"]:
            r["speaker"] = "SPEAKER_00"

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
        for seg in result["segments"]:
            txt_line = f"[{format_timestamp(seg['start'])} → {format_timestamp(seg['end'])}] "
            txt_line += f"{seg.get('speaker', 'UNKNOWN')}: {seg['text']}\n"
            f.write(txt_line)

    # SRT
    srt_path = f"{base}_subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"[{seg.get('speaker', 'UNKNOWN')}] {seg['text']}\n\n")

    print(f"\n📁 Output files saved: {json_path}, {txt_path}, {srt_path}")
    return json_path, txt_path, srt_path

def main():
    print("🚀 Faster-Whisper + Pyannote Diarization")

    ensure_dependencies()
    setup_huggingface()

    audio_files = find_audio_files()
    if not audio_files:
        print("⚠️ No audio files found")
        return

    all_outputs = []
    for audio in audio_files:
        try:
            result, _ = transcribe_with_diarization(audio)
            outputs = save_outputs(result, audio)
            all_outputs.append(outputs)
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue

    if all_outputs:
        print(f"✅ Processed {len(all_outputs)} audio files successfully")
    else:
        print("❌ No files were successfully processed")

if __name__ == "__main__":
    main()
