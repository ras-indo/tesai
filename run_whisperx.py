#!/usr/bin/env python3
"""
WHISPERX + PYANNOTE DIARIZATION - STABLE GitHub Actions Version
FIXED: torch 1.13.1 for pyannote compatibility
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
    subprocess.check_call(cmd)

def ensure_dependencies():
    """Install STABLE dependencies compatible with pyannote"""
    print("📦 Installing compatible dependencies...")
    
    # 1. Upgrade pip
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    
    # 2. Clean conflicting dependencies
    run([
        sys.executable,
        "-m",
        "pip",
        "uninstall", "-y",
        "torch",
        "torchaudio",
        "torchvision",
        "pyannote.audio",
        "whisperx"
    ])
    
    # 3. Install COMPATIBLE torch + torchaudio (CPU for CI)
    # PYANNOTE REQUIRES OLDER TORCH. This is the key fix.
    run([
        sys.executable,
        "-m", "pip", "install",
        "torch==1.13.1",
        "torchaudio==0.13.1",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    
    # 4. Install specific, compatible versions of WhisperX and Pyannote
    run([
        sys.executable,
        "-m", "pip", "install",
        "whisperx==3.1.1",
        "pyannote.audio==2.1.1"
    ])
    
    print("✅ Compatible dependencies installed")

def setup_huggingface():
    """Setup Hugging Face token from environment"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment")
        print("   Ensure it's set in GitHub Secrets")
        sys.exit(1)
    
    # Set token for Hugging Face CLI
    os.environ["HF_TOKEN"] = hf_token
    
    # Login to Hugging Face (for pyannote model download)
    from huggingface_hub import login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("✅ Logged in to Hugging Face Hub")
    except Exception as e:
        print(f"❌ Hugging Face login failed: {e}")
        sys.exit(1)

def find_audio_files():
    """Auto-detect audio files"""
    return [
        f
        for f in os.listdir(".")
        if f.lower().endswith(AUDIO_EXTENSIONS)
    ]

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp"""
    td = timedelta(seconds=seconds)
    hours = int(td.seconds // 3600)
    minutes = int((td.seconds % 3600) // 60)
    seconds = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace('.', ',')

def transcribe_with_diarization(audio_file):
    """Transcribe with speaker diarization"""
    import whisperx
    import torch
    
    # Config device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    
    print(f"\n🎯 Processing: {audio_file}")
    print(f"   Device: {device}, Compute: {compute_type}")
    print(f"   Using pyannote for speaker diarization")
    
    # 1. LOAD WHISPER MODEL
    model = whisperx.load_model(
        "medium",
        device=device,
        compute_type=compute_type,
        language=None,  # Auto-detect
    )
    
    # 2. TRANSCRIBE
    print("   📝 Transcribing...")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=8)
    
    # 3. ALIGNMENT (if not English)
    print("   🔍 Aligning...")
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
    
    # 4. DIARIZATION (SPEAKER DETECTION)
    print("   👥 Diarizing speakers...")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=os.getenv("HF_TOKEN"),
        device=device
    )
    
    # Run diarization
    diarize_segments = diarize_model(audio)
    
    # Assign speakers to transcript
    result = whisperx.assign_word_speakers(diarize_segments, result)
    
    return result, audio_file

def save_outputs(result, audio_file):
    """Save results in multiple formats"""
    base = os.path.splitext(audio_file)[0]
    
    # 1. Save JSON (full output)
    json_path = f"{base}_diarized.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
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
    print(f"\n📄 Sample transcript (first 3 segments):")
    print("=" * 60)
    for segment in result.get("segments", [])[:3]:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        start = segment.get("start", 0)
        
        print(f"[{format_timestamp(start)}] Speaker {speaker}: {text}")
    
    if len(result.get("segments", [])) > 3:
        print(f"... and {len(result['segments']) - 3} more segments")
    
    print("=" * 60)
    
    return json_path, txt_path, srt_path

def main():
    """Main workflow"""
    print("🚀 WhisperX + Pyannote Diarization (STABLE VERSION)")
    print("🔐 Using HF Token for speaker detection")
    print("🔧 Fixed: PyTorch 1.13.1 for pyannote compatibility")
    
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
            print(f"   {type(e).__name__}: {e}")
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
