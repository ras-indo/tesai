#!/usr/bin/env python3
"""
WHISPERX + PYANNOTE DIARIZATION - FIXED DEPENDENCY VERSION
Dengan HF Token Auth & Speaker Detection
FIX: Mengatasi error missing numpy, nltk, pandas, dan huggingface-hub version conflict.
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
    """Install dependencies dengan VERSI YANG DIKUNCI dan urutan yang benar"""
    print("📦 Installing dependencies with proper version locking...")
    
    # 1. Upgrade pip & setuptools
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools"])
    
    # 2. HAPUS SEMUA dependency yang berpotensi konflik
    run([
        sys.executable,
        "-m",
        "pip",
        "uninstall", "-y",
        "torch",
        "torchaudio",
        "torchvision",
        "pyannote.audio",
        "whisperx",
        "transformers",
        "pytorch-lightning",
        "nltk",
        "pandas",
        "numpy"
    ])
    
    # 3. INSTALL DEPENDENSI DASAR TERLEBIH DAHULU
    # Install numpy, pandas, nltk terlebih dahulu
    run([
        sys.executable, "-m", "pip", "install",
        "numpy==1.24.4",
        "pandas==2.0.3",
        "nltk==3.8.1",
        "huggingface-hub==0.20.3",  # Versi yang kompatibel dengan transformers 4.35.2
    ])
    
    # 4. INSTALL TORCH VERSI SPESIFIK
    # Gunakan versi 2.1.2 yang stabil dan kompatibel
    run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    # 5. INSTALL TRANSFORMERS dan dependensi terkait
    run([
        sys.executable, "-m", "pip", "install",
        "transformers==4.35.2",
        "pytorch-lightning==2.0.9",
    ])
    
    # 6. INSTALL WHISPERX dan PYANNOTE dengan dependencies
    # Tanpa --no-deps agar semua dependencies terinstall
    run([
        sys.executable, "-m", "pip", "install",
        "whisperx==3.1.1",
        "pyannote.audio==3.1.1",
    ])
    
    # 7. Install dependencies tambahan
    run([sys.executable, "-m", "pip", "install", "ffmpeg-python", "soundfile"])
    
    # 8. Download NLTK data yang diperlukan
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️  NLTK download warning: {e}")
    
    print("✅ Dependencies installed successfully.")

def setup_huggingface():
    """Setup Hugging Face token dari environment"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN tidak ditemukan di environment")
        print("   Pastikan sudah set di GitHub Secrets")
        sys.exit(1)
    
    # Set token untuk Hugging Face CLI
    os.environ["HF_TOKEN"] = hf_token
    
    # Login ke Hugging Face (untuk download model pyannote)
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Logged in to Hugging Face Hub")
    except Exception as e:
        print(f"❌ Hugging Face login failed: {e}")
        sys.exit(1)

def find_audio_files():
    """Auto-detect file audio"""
    return [
        f
        for f in os.listdir(".")
        if f.lower().endswith(AUDIO_EXTENSIONS)
    ]

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def transcribe_with_diarization(audio_file):
    """Transcribe dengan speaker diarization"""
    import whisperx
    import torch
    
    # Config device - PAKAI CPU di GitHub Actions untuk stabil
    device = "cpu"  # Force CPU untuk konsistensi di CI
    compute_type = "int8"
    
    print(f"\n🎯 Processing: {audio_file}")
    print(f"   Device: {device}, Compute: {compute_type}")
    print(f"   Using pyannote for speaker diarization")
    
    # 1. LOAD WHISPER MODEL
    model = whisperx.load_model(
        "medium",
        device=device,
        compute_type=compute_type,
        language=None,  # Auto-detect
        vad_parameters={
            "vad_onset": 0.5,
            "vad_offset": 0.363
        }
    )
    
    # 2. TRANSCRIBE
    print("   📝 Transcribing...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=4)  # Batch size lebih kecil untuk CPU
    
    # 3. ALIGNMENT (jika bukan English)
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
    
    # 4. DIARIZATION (SPEAKER DETECTION)
    print("   👥 Diarizing speakers...")
    try:
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=os.getenv("HF_TOKEN"),
            device=device
        )
        
        # Run diarization
        diarize_segments = diarize_model(audio)
        
        # Assign speakers to transcript
        result = whisperx.assign_word_speakers(diarize_segments, result)
    except Exception as e:
        print(f"   ❌ Diarization failed: {e}")
        print("   💡 Ensure you've accepted terms at:")
        print("      https://hf.co/pyannote/speaker-diarization-3.1")
        print("      https://hf.co/pyannote/segmentation-3.0")
        # Continue without diarization
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
    
    return result, audio_file

def save_outputs(result, audio_file):
    """Save results in multiple formats"""
    base = os.path.splitext(audio_file)[0]
    
    # 1. Save JSON (full output)
    json_path = f"{base}_diarized.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # Simplify JSON structure untuk mengurangi size
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
    
    # 2. Save TXT (readable transcript)
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
    
    # 3. Save SRT (for video subtitles)
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
    
    # Print summary
    print(f"\n📁 Output files saved:")
    print(f"   • {json_path} (Full JSON)")
    print(f"   • {txt_path} (Readable transcript)")
    print(f"   • {srt_path} (SRT subtitles)")
    
    # Print sample of transcript
    print(f"\n📄 Sample transcript (first 2 segments):")
    print("=" * 60)
    segments = result.get("segments", [])
    for segment in segments[:2]:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        start = segment.get("start", 0)
        
        print(f"[{format_timestamp(start)}] Speaker {speaker}: {text}")
    
    if len(segments) > 2:
        print(f"... and {len(segments) - 2} more segments")
    
    print("=" * 60)
    
    return json_path, txt_path, srt_path

def main():
    """Main workflow"""
    print("🚀 WhisperX + Pyannote Diarization - DEPENDENCY FIXED VERSION")
    print("🔐 Using HF Token for speaker detection")
    print("🔒 Locked versions: torch==2.1.2, whisperx==3.1.1, pyannote.audio==3.1.1")
    
    # Setup
    ensure_dependencies()
    setup_huggingface()
    
    # Find audio files
    audio_files = find_audio_files()
    
    if not audio_files:
        print("⚠️  No audio files found")
        print(f"   Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        return
    
    print(f"🎯 Found {len(audio_files)} audio file(s):")
    for af in audio_files:
        print(f"   • {af}")
    
    # Process each file
    all_outputs = []
    for audio in audio_files:
        try:
            print(f"\n{'='*60}")
            result, _ = transcribe_with_diarization(audio)
            outputs = save_outputs(result, audio)
            all_outputs.append(outputs)
        except Exception as e:
            print(f"❌ Error processing {audio}:")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ PROCESSING COMPLETE")
    print(f"   Processed: {len(all_outputs)} file(s)")
    for i, (json_file, txt_file, srt_file) in enumerate(all_outputs, 1):
        print(f"\n   File {i}:")
        print(f"      JSON: {json_file}")
        print(f"      TXT:  {txt_file}")
        print(f"      SRT:  {srt_file}")

if __name__ == "__main__":
    main()
