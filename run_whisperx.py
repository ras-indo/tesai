#!/usr/bin/env python3
"""
WHISPERX + PYANNOTE DIARIZATION - CLEAN ENV FIX
Menghindari register_pytree_node error
"""
import os
import sys
import subprocess
import json
from datetime import timedelta

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")

def run(cmd, check=True):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=check)

def setup_virtualenv():
    """Buat virtual environment bersih"""
    venv_dir = ".venv_whisperx"
    if not os.path.exists(venv_dir):
        print("⚡ Creating virtual environment...")
        run([sys.executable, "-m", "venv", venv_dir])
    activate = os.path.join(venv_dir, "bin", "activate_this.py")
    with open(activate) as f:
        exec(f.read(), {'__file__': activate})
    print("✅ Virtualenv activated")
    return venv_dir

def ensure_dependencies():
    """Install versi kompatibel di virtualenv"""
    print("📦 Installing dependencies (torch 2.1.2 + pyannote 3.1.1 + whisperx)...")

    # Upgrade pip
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])

    # Install fixed versions
    run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall",
        "torch==2.1.2",
        "torchaudio==2.1.2",
        "--index-url", "https://download.pytorch.org/whl/cpu"
    ])
    run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall",
        "whisperx==3.1.1",
        "pyannote.audio==3.1.1",
        "huggingface-hub"
    ])
    print("✅ Dependencies installed")

def setup_huggingface():
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found")
        sys.exit(1)
    os.environ["HF_TOKEN"] = hf_token
    from huggingface_hub import login
    login(token=hf_token, add_to_git_credential=False)
    print("✅ Logged in to Hugging Face")

def find_audio_files():
    return [f for f in os.listdir(".") if f.lower().endswith(AUDIO_EXTENSIONS)]

def format_timestamp(seconds):
    from datetime import timedelta
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")

def transcribe_with_diarization(audio_file):
    import torch
    import whisperx

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"\n🎯 Processing: {audio_file} | Device: {device} | Compute: {compute_type}")

    # Load whisper model
    model = whisperx.load_model("medium", device=device, compute_type=compute_type, language=None,
                                vad_parameters={"vad_onset":0.5, "vad_offset":0.363})

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=8)

    # Alignment
    model_a, metadata = whisperx.load_align_model(result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device,
                            return_char_alignments=False)

    # Diarization
    from whisperx import DiarizationPipeline
    diarizer = DiarizationPipeline(use_auth_token=os.getenv("HF_TOKEN"), device=device)
    diar_segments = diarizer(audio)

    # Assign speakers
    result = whisperx.assign_word_speakers(diar_segments, result)
    return result, audio_file

def save_outputs(result, audio_file):
    base = os.path.splitext(audio_file)[0]
    json_path = f"{base}_diarized.json"
    txt_path = f"{base}_transcript.txt"
    srt_path = f"{base}_subtitles.srt"

    # JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # TXT
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Audio: {audio_file}\nLanguage: {result.get('language','unknown')}\n")
        f.write("="*50 + "\n\n")
        for seg in result.get("segments", []):
            start, end = seg.get("start",0), seg.get("end",0)
            speaker, text = seg.get("speaker","UNKNOWN"), seg.get("text","").strip()
            f.write(f"[{format_timestamp(start)} → {format_timestamp(end)}] Speaker {speaker}:\n{text}\n\n")

    # SRT
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result.get("segments", []), 1):
            start, end = seg.get("start",0), seg.get("end",0)
            speaker, text = seg.get("speaker","UNKNOWN"), seg.get("text","").strip()
            f.write(f"{i}\n{format_timestamp(start)} --> {format_timestamp(end)}\n[Speaker {speaker}] {text}\n\n")

    print(f"📁 Outputs: {json_path}, {txt_path}, {srt_path}")
    return json_path, txt_path, srt_path

def main():
    print("🚀 WhisperX + Pyannote Diarization (Clean Env Fix)")
    setup_virtualenv()
    ensure_dependencies()
    setup_huggingface()

    audio_files = find_audio_files()
    if not audio_files:
        print("⚠️ No audio files found")
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
