#!/usr/bin/env python3
"""
WHISPERX + PYANNOTE DIARIZATION - ULTIMATE FIX
FIX: Isolate dependencies dengan requirements.txt
"""
import os
import sys
import subprocess
import json
from datetime import timedelta

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")

def run(cmd):
    """Helper: run shell command"""
    print("[RUN]", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(f"STDOUT: {result.stdout[:500]}...")
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}...")
    result.check_returncode()
    return result

def ensure_dependencies():
    """Install dependencies dengan requirements.txt"""
    print("📦 Installing dependencies from requirements.txt...")
    
    # 1. Buat requirements.txt
    requirements = """numpy==1.26.4
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
ctranslate2==4.3.1
onnxruntime==1.16.3
torch==2.1.2
torchaudio==2.1.2
transformers==4.35.2
pytorch-lightning==2.0.9
torchmetrics==1.3.2
pyannote.audio==3.1.1
openai-whisper==20231117
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    
    # 2. Hapus semua package yang konflik
    print("🧹 Cleaning up conflicting packages...")
    run([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio", "numpy"])
    
    # 3. Install PyTorch terlebih dahulu dengan index yang benar
    print("🔧 Installing PyTorch first...")
    run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2", "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    # 4. Install numpy 1.26.4 secara paksa
    print("🔧 Forcing numpy 1.26.4...")
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.4"])
    
    # 5. Install sisa requirements
    print("🔧 Installing remaining requirements...")
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--no-deps"])
    
    # 6. Install WhisperX dari GitHub
    print("🔧 Installing WhisperX from GitHub...")
    run([
        sys.executable, "-m", "pip", "install",
        "git+https://github.com/m-bain/whisperX.git@v3.1.0",
        "--no-deps"
    ])
    
    # 7. Install dependencies yang diperlukan
    run([sys.executable, "-m", "pip", "install", "safetensors", "tokenizers"])
    
    # 8. Download NLTK data
    print("📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ NLTK warning: {e}")
    
    # 9. Verifikasi versi
    print("🔍 Verifying versions...")
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
    
    if int(np.__version__.split('.')[0]) >= 2:
        print("❌ ERROR: NumPy 2.x detected!")
        print("   Forcing numpy 1.26.4 again...")
        run([sys.executable, "-m", "pip", "install", "--force-reinstall", "numpy==1.26.4"])
    
    # Hapus requirements.txt
    if os.path.exists("requirements.txt"):
        os.remove("requirements.txt")
    
    print("✅ All dependencies installed!")

def setup_huggingface():
    """Setup Hugging Face token"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment")
        print("   Make sure it's set in GitHub Secrets")
        sys.exit(1)
    
    os.environ["HF_TOKEN"] = hf_token
    
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Logged in to Hugging Face")
    except Exception as e:
        print(f"❌ Hugging Face login failed: {e}")
        sys.exit(1)

def find_audio_files():
    """Auto-detect audio files"""
    return [f for f in os.listdir(".") if f.lower().endswith(AUDIO_EXTENSIONS)]

def format_timestamp(seconds):
    """Convert seconds to timestamp"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace('.', ',')

def transcribe_with_diarization(audio_file):
    """Transcribe with speaker diarization"""
    import whisperx
    import torch
    import numpy as np
    
    print(f"\n🎯 Processing: {audio_file}")
    print(f"   NumPy version: {np.__version__}")
    
    # Force CPU
    device = "cpu"
    compute_type = "int8"
    
    print(f"   Device: {device}, Compute: {compute_type}")
    
    # 1. Load model
    print("   📥 Loading model...")
    model = whisperx.load_model(
        "medium",
        device=device,
        compute_type=compute_type,
        language=None,
        vad_parameters={"vad_onset": 0.5, "vad_offset": 0.363}
    )
    
    # 2. Transcribe
    print("   📝 Transcribing...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=2)
    
    # 3. Alignment
    print("   🔍 Aligning...")
    try:
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )
        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False
        )
    except Exception as e:
        print(f"   ⚠️ Alignment skipped: {e}")
    
    # 4. Diarization
    print("   👥 Diarizing speakers...")
    try:
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=os.getenv("HF_TOKEN"),
            device=device
        )
        
        diarize_segments = diarize_model(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(f"   ✅ Found {len(diarize_segments)} speaker segments")
    except Exception as e:
        print(f"   ❌ Diarization failed: {e}")
        print("   💡 Make sure you accepted terms:")
        print("      https://hf.co/pyannote/speaker-diarization-3.1")
        print("      https://hf.co/pyannote/segmentation-3.0")
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
    
    return result, audio_file

def save_outputs(result, audio_file):
    """Save results"""
    base = os.path.splitext(audio_file)[0]
    
    # JSON
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
                }
                for seg in result.get("segments", [])
            ]
        }
        json.dump(simplified, f, ensure_ascii=False, indent=2)
    
    # TXT
    txt_path = f"{base}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Audio: {audio_file}\nLanguage: {result.get('language', 'unknown')}\n")
        f.write("=" * 50 + "\n\n")
        
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            f.write(f"[{format_timestamp(start)} → {format_timestamp(end)}] ")
            f.write(f"Speaker {speaker}:\n{text}\n\n")
    
    # SRT
    srt_path = f"{base}_subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result.get("segments", []), 1):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(start)} --> {format_timestamp(end)}\n")
            f.write(f"[Speaker {speaker}] {text}\n\n")
    
    print(f"\n📁 Output files saved:")
    print(f"   • {json_path}")
    print(f"   • {txt_path}")
    print(f"   • {srt_path}")
    
    return json_path, txt_path, srt_path

def main():
    """Main workflow"""
    print("=" * 70)
    print("🚀 WhisperX + Pyannote Diarization - ULTIMATE FIX")
    print("🔐 Using HF Token for speaker detection")
    print("=" * 70)
    
    ensure_dependencies()
    setup_huggingface()
    
    audio_files = find_audio_files()
    if not audio_files:
        print("⚠️ No audio files found")
        return
    
    print(f"🎯 Found {len(audio_files)} audio file(s)")
    
    all_outputs = []
    for audio in audio_files:
        try:
            print(f"\n{'='*70}")
            result, _ = transcribe_with_diarization(audio)
            outputs = save_outputs(result, audio)
            all_outputs.append(outputs)
        except Exception as e:
            print(f"❌ Error processing {audio}:")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    if all_outputs:
        print("✅ PROCESSING COMPLETE")
        print(f"   Processed: {len(all_outputs)} file(s)")
    else:
        print("❌ No files processed successfully")

if __name__ == "__main__":
    main()
