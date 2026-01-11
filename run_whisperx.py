#!/usr/bin/env python3
"""
WHISPERX + PYANNOTE DIARIZATION - FINAL WORKING VERSION
SOLUSI: Gunakan whisperx 3.1.0 resmi + ctranslate2 + numpy 1.x
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
    """Install dependencies dengan versi yang KOMPATIBEL 100%"""
    print("🔧 Installing COMPATIBLE dependencies...")
    
    # 1. Upgrade pip dan clear cache
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    
    # 2. HAPUS semua packages yang bermasalah
    problematic_packages = [
        "torch", "torchaudio", "torchvision", "pyannote.audio", "whisperx",
        "transformers", "pytorch-lightning", "numpy", "ctranslate2",
        "nltk", "pandas", "scipy", "librosa", "soundfile"
    ]
    
    for pkg in problematic_packages:
        try:
            run([sys.executable, "-m", "pip", "uninstall", "-y", pkg])
        except:
            pass
    
    # 3. BUAT requirement.txt yang KOMPATIBEL
    requirements = """# COMPATIBLE VERSIONS for WhisperX + Pyannote
# MUST: numpy<2.0.0, whisperx 3.1.0 (official), ctranslate2
numpy==1.24.4
pandas==1.5.3
nltk==3.8.1
scipy==1.10.1
librosa==0.10.0
soundfile==0.12.1
tqdm==4.65.0
huggingface-hub==0.20.3
ffmpeg-python==0.2.0
jiwer==3.0.3
pyannote.core==5.0.0
pyannote.metrics==3.2.1
torch==2.1.2
torchaudio==2.1.2
ctranslate2==3.24.0
transformers==4.35.2
pytorch-lightning==2.0.9
accelerate==0.24.1
tokenizers==0.15.0
"""
    
    with open("requirements_compatible.txt", "w") as f:
        f.write(requirements)
    
    # 4. INSTALL semua sekaligus dengan constraint
    print("📦 Installing all compatible packages...")
    
    # Install torch CPU version first (compatible dengan numpy 1.24.4)
    run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "numpy==1.24.4",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--no-deps"
    ])
    
    # Install requirements lainnya
    run([sys.executable, "-m", "pip", "install", "-r", "requirements_compatible.txt"])
    
    # 5. Install pyannote.audio 2.1.1 (lebih stabil untuk numpy 1.x)
    run([sys.executable, "-m", "pip", "install", "pyannote.audio==2.1.1"])
    
    # 6. Install whisperx 3.1.0 (VERSI RESMI, bukan 3.1.1 yang di-yank)
    run([sys.executable, "-m", "pip", "install", "whisperx==3.1.0"])
    
    # 7. Download NLTK data
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK data downloaded")
    except:
        pass
    
    # 8. VERIFIKASI
    import numpy as np
    print(f"✅ NumPy: {np.__version__}")
    print("✅ All dependencies installed successfully!")
    
    # Cleanup
    if os.path.exists("requirements_compatible.txt"):
        os.remove("requirements_compatible.txt")

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
    """Find audio files"""
    return [f for f in os.listdir(".") if f.lower().endswith(AUDIO_EXTENSIONS)]

def format_timestamp(seconds):
    """Convert seconds to timestamp"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def transcribe_with_diarization(audio_file):
    """Transcribe with speaker diarization"""
    # Import di dalam fungsi untuk memastikan dependencies sudah terinstall
    import whisperx
    import torch
    
    # Force CPU mode untuk konsistensi
    device = "cpu"
    compute_type = "int8"
    
    print(f"\n🎯 Processing: {audio_file}")
    print(f"   Device: {device}, Compute: {compute_type}")
    
    # 1. LOAD MODEL
    print("   📥 Loading Whisper model...")
    model = whisperx.load_model(
        "medium",
        device=device,
        compute_type=compute_type,
        language=None,
        vad_parameters={"vad_onset": 0.5, "vad_offset": 0.363}
    )
    
    # 2. TRANSCRIBE
    print("   📝 Transcribing...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=2)
    
    # 3. ALIGNMENT
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
        print(f"   ⚠️  Alignment skipped: {e}")
    
    # 4. DIARIZATION
    print("   👥 Diarizing speakers...")
    try:
        # Pyannote 2.x menggunakan syntax berbeda
        from pyannote.audio import Pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.getenv("HF_TOKEN")
        )
        
        # Run diarization
        diarization = pipeline(audio_file)
        
        # Convert diarization ke format whisperx
        diarize_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarize_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        # Assign speakers
        result = whisperx.assign_word_speakers(diarize_segments, result)
        print(f"   ✅ Diarization successful")
        
    except Exception as e:
        print(f"   ❌ Diarization failed: {e}")
        print("   💡 Ensure you've accepted terms at:")
        print("      https://hf.co/pyannote/speaker-diarization-3.1")
        print("      https://hf.co/pyannote/segmentation-3.0")
        # Default speaker
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
    
    return result, audio_file

def save_outputs(result, audio_file):
    """Save results"""
    base = os.path.splitext(audio_file)[0]
    
    # 1. JSON
    json_path = f"{base}_diarized.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 2. TXT
    txt_path = f"{base}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Audio: {audio_file}\n")
        f.write(f"Language: {result.get('language', 'unknown')}\n")
        f.write("=" * 50 + "\n\n")
        
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "UNKNOWN")
            text = segment.get("text", "").strip()
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            f.write(f"[{format_timestamp(start)} → {format_timestamp(end)}] ")
            f.write(f"Speaker {speaker}:\n")
            f.write(f"{text}\n\n")
    
    # 3. SRT
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
    
    print(f"\n📁 Output saved:")
    print(f"   • {json_path}")
    print(f"   • {txt_path}")
    print(f"   • {srt_path}")
    
    return json_path, txt_path, srt_path

def main():
    """Main workflow"""
    print("=" * 70)
    print("🚀 WHISPERX + PYANNOTE - WORKING VERSION")
    print("🔐 Using HF Token for speaker detection")
    print("🔒 Compatible versions: numpy=1.24.4, whisperx=3.1.0")
    print("=" * 70)
    
    # Setup
    ensure_dependencies()
    setup_huggingface()
    
    # Find audio files
    audio_files = find_audio_files()
    if not audio_files:
        print("⚠️  No audio files found")
        return
    
    print(f"🎯 Found {len(audio_files)} audio file(s):")
    for af in audio_files:
        print(f"   • {af}")
    
    # Process
    all_outputs = []
    for audio in audio_files:
        try:
            print(f"\n{'='*70}")
            result, _ = transcribe_with_diarization(audio)
            outputs = save_outputs(result, audio)
            all_outputs.append(outputs)
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*70}")
    if all_outputs:
        print(f"✅ SUCCESS: Processed {len(all_outputs)} file(s)")
        for files in all_outputs:
            for f in files:
                print(f"   📄 {f}")
    else:
        print("❌ FAILED: No files processed")

if __name__ == "__main__":
    main()
