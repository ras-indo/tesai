#!/usr/bin/env python3
"""
run_diarization.py

Stable transcription + optional speaker diarization runner for CI (GitHub Actions)
- Uses faster-whisper for transcription (offline, CPU-friendly)
- Two-step approach: detect language first, then transcribe with language-specific settings
- Attempts to use pyannote.audio for speaker diarization (if available & HF_TOKEN provided)
- Produces outputs: <audio>_diarized.json, <audio>_transcript.txt, <audio>_subtitles.srt
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import traceback
from datetime import timedelta
from typing import List, Dict, Tuple, Optional, Any

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")


# -------------------------
# Helpers
# -------------------------
def run(cmd: List[str]) -> subprocess.CompletedProcess:
    """Run shell command and return CompletedProcess. Print output streams."""
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr)
    return proc


def format_timestamp(seconds: float) -> str:
    """Format seconds float to SRT timestamp 'HH:MM:SS,mmm'"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def find_audio_files() -> List[str]:
    return [f for f in os.listdir(".") if f.lower().endswith(AUDIO_EXTENSIONS)]


# -------------------------
# Dependency / Preflight
# -------------------------
def ensure_dependencies() -> Dict[str, bool]:
    """
    Try to ensure minimal dependencies are present.
    Returns dict of availability flags.
    """
    flags = {
        "numpy": False,
        "torch": False,
        "faster_whisper": False,
        "pyannote": False,
        "onnxruntime": False,
        "nltk": False,
    }

    # Lightweight packages we can safely pip install here if needed
    to_ensure = [
        ("numpy", ["numpy<2"]),
        ("nltk", ["nltk==3.8.1"]),
    ]
    for name, specs in to_ensure:
        try:
            __import__(name)
            flags[name] = True
            print(f"✅ {name} available")
        except Exception:
            for spec in specs:
                cp = run([sys.executable, "-m", "pip", "install", spec])
                if cp.returncode == 0:
                    try:
                        __import__(name)
                        flags[name] = True
                        print(f"✅ Installed & imported {name} ({spec})")
                        break
                    except Exception as e:
                        print(f"⚠️ Installed {spec} but import failed: {e}")
                else:
                    print(f"⚠️ pip install {spec} failed (rc={cp.returncode})")
    
    # Check heavy modules
    try:
        import torch
        flags["torch"] = True
        print("✅ torch available:", torch.__version__)
    except Exception:
        print("⚠️ torch not available. Please preinstall torch in CI for best results.")

    try:
        import faster_whisper
        flags["faster_whisper"] = True
        print("✅ faster_whisper available")
    except Exception:
        print("⚠️ faster-whisper not available. Attempting to install it now...")
        cp = run([sys.executable, "-m", "pip", "install", "faster-whisper"])
        if cp.returncode == 0:
            try:
                import faster_whisper
                flags["faster_whisper"] = True
                print("✅ faster-whisper installed")
            except Exception as e:
                print("⚠️ faster-whisper import failed after install:", e)

    # pyannote.audio
    pyannote_versions_to_try = ["4.0.3", "4.1.1", "3.1.1", "2.1.1"]
    try:
        import importlib.util
        if importlib.util.find_spec("pyannote.audio") is not None:
            flags["pyannote"] = True
            print("✅ pyannote.audio already installed")
        else:
            for v in pyannote_versions_to_try:
                print(f"ℹ️ Attempting to pip install pyannote.audio=={v}")
                cp = run([sys.executable, "-m", "pip", "install", f"pyannote.audio=={v}"])
                if cp.returncode == 0:
                    try:
                        __import__("pyannote.audio")
                        flags["pyannote"] = True
                        print(f"✅ Installed pyannote.audio=={v}")
                        break
                    except Exception as e:
                        print(f"⚠️ Installed pyannote.audio=={v} but import failed: {e}")
                else:
                    print(f"⚠️ pip install pyannote.audio=={v} failed (rc={cp.returncode})")
    except Exception as e:
        print("⚠️ Exception while probing pyannote:", e)

    print("---- preflight summary ----")
    for k, v in flags.items():
        print(f"{k}: {'OK' if v else 'MISSING/FAIL'}")
    print("---------------------------")
    return flags


# -------------------------
# Hugging Face helpers
# -------------------------
def get_hf_token() -> Optional[str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    return token


def setup_hf_cli_login(token: Optional[str]) -> None:
    """Optionally login using huggingface_hub.login if token present."""
    if not token:
        print("ℹ️ No HF_TOKEN found in environment; gated HF models will not be accessible.")
        return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✅ Hugging Face login (via huggingface_hub.login) succeeded")
    except Exception as e:
        print("⚠️ huggingface_hub.login failed:", e)


# -------------------------
# Enhanced Language Detection
# -------------------------
def detect_audio_language(audio_path: str, model) -> Tuple[str, float]:
    """
    Detect language of audio file with confidence score.
    Returns (language_code, confidence)
    """
    try:
        # Use faster-whisper's built-in language detection
        # Transcribe first 30 seconds for faster detection
        print(f"Detecting language for {audio_path}...")
        
        # Method 1: Use faster-whisper's language detection
        try:
            from faster_whisper.transcribe import TranscriptionOptions
            
            # Create a temporary transcription for language detection
            segments, info = model.transcribe(
                audio_path,
                beam_size=1,  # Faster for detection
                task="transcribe",
                language=None,  # Auto-detect
                vad_filter=True,  # Use VAD for better detection
                vad_parameters=dict(min_silence_duration_ms=500),
                word_timestamps=False,  # Not needed for detection
                best_of=1  # Faster
            )
            
            # Get the first segment to trigger processing
            first_segment = next(segments, None)
            
            language = info.language if hasattr(info, 'language') else "unknown"
            language_probability = getattr(info, 'language_probability', 0.0)
            
            print(f"ℹ️ Detected language: {language} (confidence: {language_probability:.2%})")
            
            # Map to language codes if needed
            language_map = {
                "indonesian": "id",
                "english": "en",
                "japanese": "ja",
                "korean": "ko",
                "chinese": "zh",
                "spanish": "es",
                "french": "fr",
                "german": "de",
                "russian": "ru",
                "arabic": "ar",
                "portuguese": "pt",
                "italian": "it",
                "dutch": "nl",
                "hindi": "hi",
                "turkish": "tr"
            }
            
            if language in language_map:
                language = language_map[language]
            
            return language, language_probability
            
        except Exception as e:
            print(f"⚠️ Language detection error: {e}")
            
            # Fallback method: transcribe first 10 seconds
            try:
                import subprocess
                import tempfile
                
                # Extract first 10 seconds for language detection
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    temp_path = f.name
                
                # Use ffmpeg to extract first 10 seconds
                cmd = [
                    "ffmpeg", "-i", audio_path,
                    "-t", "10",  # 10 seconds
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-y",  # Overwrite
                    temp_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    segments, info = model.transcribe(
                        temp_path,
                        beam_size=1,
                        task="transcribe",
                        language=None,
                        word_timestamps=False
                    )
                    
                    # Consume generator
                    list(segments)
                    
                    language = info.language if hasattr(info, 'language') else "unknown"
                    language_probability = getattr(info, 'language_probability', 0.0)
                    
                    # Clean up temp file
                    os.unlink(temp_path)
                    
                    print(f"ℹ️ Fallback detection: {language} (confidence: {language_probability:.2%})")
                    return language, language_probability
                    
            except Exception as e2:
                print(f"⚠️ Fallback detection failed: {e2}")
    
    except Exception as e:
        print(f"⚠️ Language detection completely failed: {e}")
    
    return "unknown", 0.0


# -------------------------
# Core processing with enhanced language detection
# -------------------------
def transcribe_with_faster_whisper(audio_path: str, device: str = "cpu") -> Tuple[Dict, str]:
    """
    Enhanced transcription with two-step language detection.
    Returns (result_dict, audio_path).
    """
    try:
        from faster_whisper import WhisperModel
    except Exception:
        raise RuntimeError("faster_whisper is not installed or import failed.")

    print(f"Transcribing {audio_path} with enhanced language detection")
    
    # Load model for language detection (use smaller model if available)
    model_size = os.getenv("FW_MODEL", "large-v3")
    compute_type = os.getenv("FW_COMPUTE", "int8")
    
    # For language detection, we could use a smaller model but let's use the same
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    # Step 1: Detect language with confidence
    detected_language, confidence = detect_audio_language(audio_path, model)
    
    # Step 2: Configure transcription based on detected language
    transcription_params = {
        "beam_size": 5,
        "word_timestamps": True,
        "vad_filter": True,
        "vad_parameters": dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
            threshold=0.5
        )
    }
    
    # Language-specific optimizations
    language_configs = {
        "id": {  # Indonesian
            "initial_prompt": "Transkripsi percakapan dalam Bahasa Indonesia.",
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.0,
            "best_of": 3,
        },
        "en": {  # English
            "initial_prompt": "Transcription of English conversation.",
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "best_of": 5,
        },
        "ja": {  # Japanese
            "initial_prompt": "日本語の会話の文字起こし。",
            "condition_on_previous_text": False,
            "best_of": 3,
        },
        "ko": {  # Korean
            "initial_prompt": "한국어 대화 전사.",
            "condition_on_previous_text": False,
            "best_of": 3,
        },
        "zh": {  # Chinese
            "initial_prompt": "中文对话转录。",
            "condition_on_previous_text": False,
            "best_of": 3,
        }
    }
    
    # Apply language-specific config if available
    if detected_language in language_configs and confidence > 0.3:
        config = language_configs[detected_language]
        transcription_params.update(config)
        print(f"ℹ️ Applying language-specific config for {detected_language}")
    else:
        print(f"ℹ️ Using generic transcription settings")
    
    # Set language parameter if detected with reasonable confidence
    if detected_language != "unknown" and confidence > 0.1:
        transcription_params["language"] = detected_language
        print(f"ℹ️ Transcribing with detected language: {detected_language}")
    else:
        print("⚠️ Language detection uncertain, using auto-detection")
    
    # Perform transcription
    try:
        transcription = model.transcribe(audio_path, **transcription_params)
    except Exception as e:
        print(f"⚠️ Transcription with language detection failed: {e}")
        # Fallback to basic transcription
        transcription = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    
    # Normalize transcription into a result dict
    segments = []
    language = detected_language  # Use detected language as default
    
    if isinstance(transcription, tuple) and len(transcription) >= 1:
        segs = transcription[0]
        info = transcription[1] if len(transcription) > 1 else {}
        
        for s in segs:
            start = getattr(s, "start", getattr(s, "start_time", 0.0))
            end = getattr(s, "end", getattr(s, "end_time", start))
            text = getattr(s, "text", getattr(s, "content", "")).strip()
            
            words = []
            wlist = getattr(s, "words", None)
            if wlist:
                for w in wlist:
                    wstart = getattr(w, "start", None)
                    wend = getattr(w, "end", None)
                    wtext = getattr(w, "word", getattr(w, "text", ""))
                    words.append({"word": wtext, "start": wstart, "end": wend})
            
            segments.append({
                "start": float(start),
                "end": float(end),
                "text": text,
                "words": words,
                "confidence": getattr(s, "confidence", 0.0)
            })
        
        # Override with detected language from transcription if different
        if hasattr(info, 'language'):
            trans_lang = info.language
            if trans_lang and trans_lang != language:
                print(f"ℹ️ Transcription detected language: {trans_lang}")
                language = trans_lang
    
    result = {
        "language": language or "unknown",
        "language_confidence": confidence,
        "segments": segments,
        "detected_language": detected_language
    }
    
    print(f"Transcription produced {len(segments)} segments; language={result['language']} (detected: {detected_language})")
    
    # Calculate average confidence
    if segments:
        avg_confidence = sum(s.get("confidence", 0) for s in segments) / len(segments)
        print(f"Average segment confidence: {avg_confidence:.2%}")
    
    return result, audio_path


# -------------------------
# Diarization (fixed for pyannote 4.x)
# -------------------------
def run_diarization_map(transcript_segments: List[Dict], audio_path: str, hf_token: Optional[str]) -> Tuple[List[Dict], bool]:
    """Run diarization and map speakers to transcript segments."""
    try:
        from pyannote.audio import Pipeline
        import pyannote.audio
        print(f"ℹ️ pyannote.audio version: {pyannote.audio.__version__}")
    except Exception as e:
        print(f"⚠️ pyannote.audio not available: {e}")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    if not hf_token:
        print("⚠️ No HF_TOKEN provided — skipping diarization.")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        print("✅ Diarization pipeline loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load pipeline: {e}")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    # Run diarization
    try:
        diarization_result = pipeline(audio_path)
        print(f"ℹ️ Diarization result type: {type(diarization_result)}")
    except Exception as e:
        print(f"⚠️ Diarization failed: {e}")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    # Extract speaker turns from DiarizeOutput
    turns = []
    try:
        # Access the annotation from DiarizeOutput
        if hasattr(diarization_result, 'annotation'):
            annotation = diarization_result.annotation
            for segment, track, label in annotation.itertracks(yield_label=True):
                turns.append((segment.start, segment.end, label))
        elif hasattr(diarization_result, 'speaker_diarization'):
            annotation = diarization_result.speaker_diarization
            for segment, track, label in annotation.itertracks(yield_label=True):
                turns.append((segment.start, segment.end, label))
        else:
            print(f"⚠️ Unknown diarization result structure")
            for s in transcript_segments:
                s["speaker"] = "SPEAKER_00"
            return transcript_segments, False
    except Exception as e:
        print(f"⚠️ Could not extract diarization turns: {e}")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    if not turns:
        print("⚠️ No speaker turns found.")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    print(f"ℹ️ Extracted {len(turns)} speaker turns")

    # Map speaker labels to transcript segments by overlap
    def overlap(a_start, a_end, b_start, b_end) -> float:
        return max(0.0, min(a_end, b_end) - max(a_start, b_start))

    for seg in transcript_segments:
        s0 = float(seg.get("start", 0.0))
        e0 = float(seg.get("end", s0 + 0.001))
        best_label = None
        best_ov = 0.0
        
        for (t0, t1, lbl) in turns:
            ov = overlap(s0, e0, float(t0), float(t1))
            if ov > best_ov:
                best_ov = ov
                best_label = lbl
        
        # Only assign speaker if there's reasonable overlap (>0.1s)
        if best_ov > 0.1:
            seg["speaker"] = best_label
        else:
            seg["speaker"] = "SPEAKER_00"

    unique_speakers = len(set([t[2] for t in turns]))
    print(f"✅ Diarization complete. Found {unique_speakers} speaker(s).")
    return transcript_segments, True


# -------------------------
# Output serialization
# -------------------------
def save_outputs(result: Dict, audio_file: str) -> Tuple[str, str, str]:
    base = os.path.splitext(audio_file)[0]
    json_path = f"{base}_diarized.json"
    txt_path = f"{base}_transcript.txt"
    srt_path = f"{base}_subtitles.srt"

    # JSON output with enhanced info
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "audio_file": audio_file,
            "language": result.get("language", "unknown"),
            "detected_language": result.get("detected_language", "unknown"),
            "language_confidence": result.get("language_confidence", 0.0),
            "segments": result.get("segments", [])
        }, fh, ensure_ascii=False, indent=2)

    # TXT output
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Audio: {audio_file}\n")
        fh.write(f"Language: {result.get('language', 'unknown')} ")
        fh.write(f"(detected: {result.get('detected_language', 'unknown')}, ")
        fh.write(f"confidence: {result.get('language_confidence', 0):.2%})\n")
        fh.write("=" * 60 + "\n\n")
        
        for seg in result.get("segments", []):
            speaker = seg.get("speaker", "UNKNOWN")
            confidence = seg.get("confidence", 0)
            fh.write(f"[{format_timestamp(seg.get('start', 0.0))} → {format_timestamp(seg.get('end', 0.0))}] ")
            fh.write(f"Speaker {speaker} (conf: {confidence:.2%}):\n")
            fh.write(seg.get("text", "").strip() + "\n\n")

    # SRT output
    with open(srt_path, "w", encoding="utf-8") as fh:
        for idx, seg in enumerate(result.get("segments", []), start=1):
            fh.write(f"{idx}\n")
            fh.write(f"{format_timestamp(seg.get('start', 0.0))} --> {format_timestamp(seg.get('end', 0.0))}\n")
            speaker = seg.get("speaker", "UNKNOWN")
            confidence = seg.get("confidence", 0)
            fh.write(f"[{speaker}] {seg.get('text', '').strip()}\n\n")

    print(f"Outputs written: {json_path}, {txt_path}, {srt_path}")
    return json_path, txt_path, srt_path


# -------------------------
# Main orchestration
# -------------------------
def main():
    print("=" * 70)
    print("Enhanced Faster-Whisper + Pyannote Diarization runner")
    print("=" * 70)

    # Step 0: preflight ensure minimal deps
    flags = ensure_dependencies()

    # Setup HF login optionally (non-fatal)
    hf = get_hf_token()
    setup_hf_cli_login(hf)

    # Locate audio files
    audio_files = find_audio_files()
    if not audio_files:
        print("No audio files found in working directory. Supported extensions:", ", ".join(AUDIO_EXTENSIONS))
        sys.exit(0)

    print(f"Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(" -", f)

    processed = 0
    for audio_path in audio_files:
        print("\n" + "=" * 40)
        print("Processing:", audio_path)
        try:
            # Transcribe with enhanced language detection
            result, _ = transcribe_with_faster_whisper(audio_path, device="cpu")
            
            # Try diarization mapping if pyannote available
            segments_with_speakers, diar_ok = run_diarization_map(
                result.get("segments", []), audio_path, hf
            )
            
            result["segments"] = segments_with_speakers
            result["diarization_ok"] = bool(diar_ok)
            
            # Save outputs
            save_outputs(result, audio_path)
            processed += 1
            
        except Exception as e:
            print("❌ Error processing", audio_path, ":", e)
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print(f"Processing complete. Successfully processed {processed}/{len(audio_files)} audio files.")
    print("=" * 70)


if __name__ == "__main__":
    main()