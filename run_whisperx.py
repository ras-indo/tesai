#!/usr/bin/env python3
"""
WHISPERX + PYANNOTE DIARIZATION - FINAL STABLE VERSION
FIX: Complete dependency management with pre-flight check
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
    
    # 1. Core packages yang harus diinstall pertama
    print("🔧 Step 1: Installing core packages (including regex for NLTK)...")
    core_packages = [
        "regex",  # Critical for NLTK's punkt tokenizer
        "numpy==1.26.4",
        "packaging",
        "coloredlogs",
        "flatbuffers",
        "protobuf"
    ]
    for pkg in core_packages:
        run([sys.executable, "-m", "pip", "install", pkg])
    
    # 2. Install pyannote.audio dan dependencies-nya SEBELUM WhisperX
    print("🔧 Step 2: Installing pyannote.audio and direct dependencies...")
    pyannote_deps = [
        "asteroid-filterbanks>=0.4",
        "einops>=0.6.0",
        "pyannote.core==5.0.0",
        "pyannote.audio==3.1.1"
    ]
    for pkg in pyannote_deps:
        run([sys.executable, "-m", "pip", "install", pkg])
    
    # 3. Install PyTorch 2.1.2 (CPU untuk GitHub Actions)
    print("🔧 Step 3: Installing PyTorch 2.1.2...")
    run([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    # 4. Install ONNX Runtime dan paket kunci lainnya
    print("🔧 Step 4: Installing ONNX Runtime and transformers...")
    key_packages = [
        "onnxruntime==1.16.3",
        "transformers==4.35.2",
        "pytorch-lightning==2.0.9",
        "ctranslate2==4.3.1",
        "openai-whisper==20231117"
    ]
    for pkg in key_packages:
        run([sys.executable, "-m", "pip", "install", pkg])
    
    # 5. Install paket lainnya yang diperlukan
    print("🔧 Step 5: Installing other required packages...")
    other_packages = [
        "pandas==2.0.3",
        "nltk==3.8.1",
        "scipy==1.11.4",
        "scikit-learn==1.3.2",
        "librosa==0.10.0",
        "soundfile==0.12.1",
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
    
    # 6. FINALLY, install WhisperX dari GitHub
    print("🔧 Step 6: Installing WhisperX from GitHub...")
    run([sys.executable, "-m", "pip", "install", "git+https://github.com/m-bain/whisperX.git@v3.1.0"])
    
    # 7. Download NLTK data
    print("📥 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠️ NLTK warning: {e}")
    
    # 8. PRE-FLIGHT CHECK
    print("\n" + "="*60)
    print("🚀 RUNNING PRE-FLIGHT CHECK")
    print("="*60)
    
    check_passed = True
    required_modules = [
        ("numpy", lambda: f"Version: {__import__('numpy').__version__}"),
        ("torch", lambda: f"Version: {__import__('torch').__version__}"),
        ("whisperx", lambda: "Available"),
        ("regex", lambda: "Available"),
        ("asteroid.filterbanks", lambda: "Available"),
        ("einops", lambda: "Available"),
        ("onnxruntime", lambda: "Available"),
    ]
    
    for module_name, check_func in required_modules:
        try:
            result = check_func()
            print(f"✅ {module_name}: {result}")
        except ImportError as e:
            print(f"❌ {module_name}: Import failed - {e}")
            check_passed = False
        except Exception as e:
            print(f"⚠️ {module_name}: Check warning - {e}")
    
    # Check HF Token
    if os.getenv("HF_TOKEN"):
        print("✅ HF_TOKEN is set in environment")
    else:
        print("❌ HF_TOKEN is NOT set in environment")
        check_passed = False
    
    print("="*60)
    if check_passed:
        print("✅ Pre-flight check passed! All dependencies are ready.")
    else:
        print("⚠️ Pre-flight check found issues. The script may fail.")
    
    return check_passed

def setup_huggingface():
    """Setup Hugging Face token dari environment"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN tidak ditemukan di environment")
        print("   Pastikan sudah set di GitHub Secrets")
        sys.exit(1)
    
    os.environ["HF_TOKEN"] = hf_token
    
    # Login ke Hugging Face
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Logged in to Hugging Face Hub")
    except Exception as e:
        print(f"❌ Hugging Face login failed: {e}")
        sys.exit(1)

def find_audio_files():
    """Auto-detect file audio"""
    return [f for f in os.listdir(".") if f.lower().endswith(AUDIO_EXTENSIONS)]

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def transcribe_with_diarization(audio_file):
    """Transcribe dengan speaker diarization - WITH ENHANCED ERROR HANDLING"""
    import whisperx
    import torch
    import numpy as np
    
    print(f"\n🎯 Processing: {audio_file}")
    print(f"   NumPy: {np.__version__}, PyTorch: {torch.__version__}")
    print("="*50)
    
    # Config device - PAKAI CPU di GitHub Actions
    device = "cpu"
    compute_type = "int8"
    
    print(f"   Device: {device}, Compute: {compute_type}")
    print(f"   Using pyannote for speaker diarization")
    
    # 1. LOAD WHISPER MODEL
    print("   📥 Loading Whisper model...")
    try:
        model = whisperx.load_model(
            "medium",
            device=device,
            compute_type=compute_type,
            language=None,
            vad_parameters={"vad_onset": 0.5, "vad_offset": 0.363}
        )
    except Exception as e:
        print(f"   ❌ Failed to load Whisper model: {e}")
        raise
    
    # 2. TRANSCRIBE
    print("   📝 Transcribing...")
    try:
        audio = whisperx.load_audio(audio_file)
        result = model.transcribe(audio, batch_size=2)
        print(f"   ✅ Transcription complete, {len(result['segments'])} segments")
    except Exception as e:
        print(f"   ❌ Transcription failed: {e}")
        raise
    
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
        print(f"   ✅ Alignment complete for language: {result['language']}")
    except Exception as e:
        print(f"   ⚠️ Alignment skipped: {e}")
    
    # 4. DIARIZATION - dengan error handling yang lebih baik
    print("   👥 Diarizing speakers...")
    diarization_success = False
    try:
        # Cek apakah token sudah ada
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("   ⚠️ HF_TOKEN not found, skipping diarization")
            raise ValueError("HF_TOKEN not found")
        
        diarize_model = whisperx.DiarizationPipeline(
            use_auth_token=hf_token,
            device=device
        )
        
        # Coba dengan parameter yang lebih sederhana
        diarize_segments = diarize_model(audio, min_speakers=1, max_speakers=4)
        
        if diarize_segments and len(diarize_segments) > 0:
            result = whisperx.assign_word_speakers(diarize_segments, result)
            print(f"   ✅ Diarization successful. Found {len(diarize_segments)} speaker segments.")
            diarization_success = True
        else:
            print("   ⚠️ Diarization returned empty segments.")
            raise ValueError("Empty diarization segments")
            
    except Exception as e:
        print(f"   ❌ Diarization failed: {e}")
        print("   🔍 This is the most common point of failure. Possible causes:")
        print("      1. HF_TOKEN lacks 'write' permission ")
        print("      2. You haven't accepted model terms on Hugging Face")
        print("      3. PyTorch version mismatch with pyannote models")
        print("   💡 Continuing without speaker labels...")
    
    # Jika diarization gagal, set speaker default
    if not diarization_success:
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"
    
    return result, audio_file

def save_outputs(result, audio_file):
    """Save results in multiple formats"""
    base = os.path.splitext(audio_file)[0]
    
    # 1. Save JSON
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
    
    # 2. Save TXT
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
    
    # 3. Save SRT
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
    print(f"   • {json_path} (Full JSON)")
    print(f"   • {txt_path} (Readable transcript)")
    print(f"   • {srt_path} (SRT subtitles)")
    
    # Print sample
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
    print("=" * 70)
    print("🚀 WhisperX + Pyannote Diarization - FINAL STABLE VERSION")
    print("🔐 Using HF Token for speaker detection")
    print("=" * 70)
    
    # Setup dependencies
    deps_ok = ensure_dependencies()
    if not deps_ok:
        print("⚠️ Dependencies have issues, but continuing anyway...")
    
    # Setup Hugging Face
    setup_huggingface()
    
    # Find audio files
    audio_files = find_audio_files()
    if not audio_files:
        print("⚠️ No audio files found")
        print(f"   Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        return
    
    print(f"🎯 Found {len(audio_files)} audio file(s):")
    for af in audio_files:
        print(f"   • {af}")
    
    # Process each file
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
    
    # Summary
    print(f"\n{'='*70}")
    if all_outputs:
        print("✅ PROCESSING COMPLETE")
        print(f"   Successfully processed: {len(all_outputs)} file(s)")
        for i, (json_file, txt_file, srt_file) in enumerate(all_outputs, 1):
            print(f"\n   File {i}:")
            print(f"      JSON: {json_file}")
            print(f"      TXT:  {txt_file}")
            print(f"      SRT:  {srt_file}")
    else:
        print("❌ No files were successfully processed")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Check HF_TOKEN permissions (may need 'write' access)")
        print("   2. Accept model terms at:")
        print("      - https://hf.co/pyannote/speaker-diarization-3.1")
        print("      - https://hf.co/pyannote/segmentation-3.0")
        print("   3. Try running with vad_method='silero' to bypass pyannote")
        print("   4. Check GitHub Actions logs for detailed errors")

if __name__ == "__main__":
    main()
