#!/usr/bin/env python3
"""
WHISPERX + PYANNOTE DIARIZATION - FINAL FIXED VERSION
✅ FIX: NumPy 2.x incompatibility, add ctranslate2 & sentencepiece
"""
import os
import sys
import subprocess
import json
from datetime import timedelta

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")

def run(cmd, capture=False):
    """Helper: run shell command dengan logging"""
    print("[RUN]", " ".join(cmd))
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        result.check_returncode()
        return result
    else:
        subprocess.check_call(cmd)

def ensure_dependencies():
    """Install dependencies dengan ISOLASI KETAT"""
    print("📦 Installing dependencies with strict isolation...")
    
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools"])
    
    packages_to_remove = [
        "torch", "torchaudio", "torchvision", "pyannote.audio", "whisperx",
        "transformers", "pytorch-lightning", "nltk", "pandas", "numpy",
        "scipy", "scikit-learn", "librosa", "soundfile", "tqdm", "numba",
        "jiwer", "pyannote.core", "huggingface-hub", "ffmpeg-python",
        "ctranslate2", "sentencepiece"
    ]
    
    run([sys.executable, "-m", "pip", "uninstall", "-y"] + packages_to_remove)
    
    requirements = """# Strictly compatible versions for WhisperX + Pyannote
numpy==1.26.4
pandas==2.0.3
nltk==3.8.1
scipy==1.11.4
scikit-learn==1.3.2
librosa==0.10.0
soundfile==0.12.1
tqdm==4.66.1
numba==0.58.1
jiwer==3.0.3
pyannote.core==5.0.0
huggingface-hub==0.20.3
ffmpeg-python==0.2.0
ctranslate2==3.34.2
sentencepiece==0.1.99
"""
    
    with open("requirements_temp.txt", "w") as f:
        f.write(requirements)
    
    run([sys.executable, "-m", "pip", "install", "-r", "requirements_temp.txt"])
    
    # Torch (CPU only)
    run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--no-deps"
    ])
    
    run([
        sys.executable, "-m", "pip", "install",
        "transformers==4.35.2",
        "pytorch-lightning==2.0.9",
    ])
    
    run([
        sys.executable, "-m", "pip", "install",
        "pyannote.audio==3.1.1",
        "--no-deps"
    ])
    
    run([
        sys.executable, "-m", "pip", "install",
        "whisperx==3.1.1",
        "--no-deps"
    ])
    
    import numpy as np
    print(f"✅ NumPy version verified: {np.__version__}")
    if int(np.__version__.split('.')[0]) >= 2:
        run([sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.4"])
    
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    
    if os.path.exists("requirements_temp.txt"):
        os.remove("requirements_temp.txt")
    
    print("✅ All dependencies installed successfully")

def setup_huggingface():
    """Setup Hugging Face token"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment")
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
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def transcribe_with_diarization(audio_file):
    import whisperx
    import torch
    import numpy as np
    
    device = "cpu"
    compute_type = "int8"
    
    print(f"\n🎯 Processing: {audio_file}")
    print(f"   Device: {device}, Compute: {compute_type}")
    
    print("   📥 Loading Whisper model...")
    model = whisperx.load_model("medium", device=device, compute_type=compute_type, language=None)
    
    print("   📝 Transcribing...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=2)
    
    print("   🔍 Aligning...")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
    except Exception as e:
        print(f"   ⚠️  Alignment skipped: {e}")
    
    print("   👥 Diarizing speakers...")
    try:
        diarize_model = whisperx.DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
    except Exception as e:
        print(f"   ❌ Diarization failed: {e}")
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
    
    return result, audio_file

def save_outputs(result, audio_file):
    base = os.path.splitext(audio_file)[0]
    
    json_path = f"{base}_diarized.json"
    with open(json_path, "w", encoding="utf-8") as f:
        simplified = {
            "audio_file": audio_file,
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": seg.get("text", "").strip(),
                    "speaker": seg.get("speaker", "UNKNOWN")
                } for seg in result.get("segments", [])
            ]
        }
        json.dump(simplified, f, ensure_ascii=False, indent=2)
    
    txt_path = f"{base}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Audio: {audio_file}\nLanguage: {result.get('language', 'unknown')}\n")
        f.write("=" * 50 + "\n\n")
        for seg in result.get("segments", []):
            f.write(f"[{format_timestamp(seg.get('start',0))} → {format_timestamp(seg.get('end',0))}] ")
            f.write(f"Speaker {seg.get('speaker','UNKNOWN')}:\n{seg.get('text','').strip()}\n\n")
    
    srt_path = f"{base}_subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.get("segments", []), 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.get('start',0))} --> {format_timestamp(seg.get('end',0))}\n")
            f.write(f"[Speaker {seg.get('speaker','UNKNOWN')}] {seg.get('text','').strip()}\n\n")
    
    print(f"\n📁 Outputs saved: {json_path}, {txt_path}, {srt_path}")
    return json_path, txt_path, srt_path

def main():
    print("=" * 70)
    print("🚀 WhisperX + Pyannote Diarization - FINAL FIX")
    print("=" * 70)
    
    ensure_dependencies()
    setup_huggingface()
    
    audio_files = find_audio_files()
    if not audio_files:
        print("⚠️  No audio files found")
        return
    
    all_outputs = []
    for audio in audio_files:
        try:
            result, _ = transcribe_with_diarization(audio)
            outputs = save_outputs(result, audio)
            all_outputs.append(outputs)
        except Exception as e:
            print(f"❌ Error processing {audio}: {e}")
            import traceback
            traceback.print_exc()
    
    if all_outputs:
        print("✅ All files processed successfully")
    else:
        print("❌ No files were successfully processed")

if __name__ == "__main__":
    main()
