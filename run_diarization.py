#!/usr/bin/env python3
"""
run_diarization.py

Stable transcription + optional speaker diarization runner for CI (GitHub Actions)
- Uses faster-whisper for transcription (offline, CPU-friendly)
- Attempts to use pyannote.audio for speaker diarization (if available & HF_TOKEN provided)
- Safe installation fallback for pyannote.audio (tries several known versions)
- Produces outputs: <audio>_diarized.json, <audio>_transcript.txt, <audio>_subtitles.srt

Usage:
 - Ensure HF_TOKEN is set in environment if you want diarization with gated HF models:
     export HF_TOKEN="hf_xxx"
 - Prefer installing dependencies in workflow; this script will try safe installs/fallbacks
   for pyannote.audio but will continue even if diarization is unavailable.

Notes:
 - This script is written to be robust in CI / ephemeral environments.
 - It favors safe continuations over hard failures so transcription completes even if diarization fails.
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
import traceback
from datetime import timedelta
from typing import List, Dict, Tuple, Optional

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
    Try to ensure minimal dependencies are present. Install light-weight packages if missing.
    For heavy packages (torch / pyannote) it's better to pre-install in workflow.
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
            # try install
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
    # Check heavy modules (we do NOT force-install torch/pyannote here except safe fallback attempts)
    try:
        import torch  # type: ignore
        flags["torch"] = True
        print("✅ torch available:", torch.__version__)
    except Exception:
        print("⚠️ torch not available. Please preinstall torch in CI for best results.")

    try:
        import faster_whisper  # type: ignore
        flags["faster_whisper"] = True
        print("✅ faster_whisper available")
    except Exception:
        print("⚠️ faster_whisper not available. Attempting to install it now...")
        cp = run([sys.executable, "-m", "pip", "install", "faster-whisper"])
        if cp.returncode == 0:
            try:
                import faster_whisper  # type: ignore
                flags["faster_whisper"] = True
                print("✅ faster_whisper installed")
            except Exception as e:
                print("⚠️ faster_whisper import failed after install:", e)

    # ONNX runtime check
    try:
        import onnxruntime  # type: ignore
        flags["onnxruntime"] = True
        print("✅ onnxruntime available")
    except Exception:
        print("⚠️ onnxruntime not available (install only if you need it)")

    # pyannote.audio: try several fallbacks but DO NOT abort if unavailable
    # KEY FIX: Target pyannote.audio 2.1.1 for API compatibility with your script
    pyannote_versions_to_try = ["2.1.1", "4.1.1", "4.0.3", "3.4.0"]
    pyannote_ok = False
    try:
        import importlib.util

        if importlib.util.find_spec("pyannote.audio") is not None:
            flags["pyannote"] = True
            pyannote_ok = True
            print("✅ pyannote.audio already installed")
        else:
            # Try pip installing one of pinned versions (best-effort)
            for v in pyannote_versions_to_try:
                print(f"ℹ️ Attempting to pip install pyannote.audio=={v} (best-effort; may fail on some runners)")
                cp = run([sys.executable, "-m", "pip", "install", f"pyannote.audio=={v}"])
                if cp.returncode == 0:
                    try:
                        __import__("pyannote.audio")
                        flags["pyannote"] = True
                        pyannote_ok = True
                        print(f"✅ Installed pyannote.audio=={v}")
                        break
                    except Exception as e:
                        print(f"⚠️ Installed pyannote.audio=={v} but import failed: {e}")
                else:
                    print(f"⚠️ pip install pyannote.audio=={v} failed (rc={cp.returncode})")
    except Exception as e:
        print("⚠️ Exception while probing pyannote:", e)

    # Final: report
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
    """
    Optionally login using huggingface_hub.login if token present.
    Use add_to_git_credential=False to avoid altering git config in CI.
    """
    if not token:
        print("ℹ️ No HF_TOKEN found in environment; gated HF models will not be accessible.")
        return
    try:
        from huggingface_hub import login  # type: ignore

        login(token=token, add_to_git_credential=False)
        print("✅ Hugging Face login (via huggingface_hub.login) succeeded")
    except Exception as e:
        print("⚠️ huggingface_hub.login failed:", e)
        # continue without failing


# -------------------------
# Core processing
# -------------------------
def transcribe_with_faster_whisper(audio_path: str, device: str = "cpu") -> Tuple[Dict, str]:
    """
    Transcribe using faster-whisper. Returns (result_dict, audio_path).
    result_dict: { "language": str, "segments": [ {start,end,text,words?} ] }
    """
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception:
        raise RuntimeError("faster_whisper is not installed or import failed.")

    print(f"Transcribing {audio_path} with faster-whisper (device={device})")
    model_size = os.getenv("FW_MODEL", "large-v3")  # allow override
    compute_type = os.getenv("FW_COMPUTE", "int8")  # int8 on CPU is common
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    # The faster-whisper transcribe API can accept an audio path
    # It returns segments and info (library versions vary). We'll handle both styles.
    try:
        # prefer file path API (returns segments, info)
        transcription = model.transcribe(audio_path, beam_size=5, word_timestamps=True)
    except TypeError:
        # fallback if signature differs
        transcription = model.transcribe(audio_path, beam_size=5)

    # Normalize transcription into a result dict
    segments = []
    language = "unknown"
    # transcription might be (segments, info) or a dict
    if isinstance(transcription, tuple) and len(transcription) >= 1:
        segs = transcription[0]
        info = transcription[1] if len(transcription) > 1 else {}
        # iterate segments; segments might be dataclass-like objects with start,end,text,words
        for s in segs:
            start = getattr(s, "start", getattr(s, "start_time", 0.0))
            end = getattr(s, "end", getattr(s, "end_time", start))
            text = getattr(s, "text", getattr(s, "content", "")).strip()
            # words may be available as s.words; otherwise leave empty
            words = []
            wlist = getattr(s, "words", None)
            if wlist:
                for w in wlist:
                    wstart = getattr(w, "start", None)
                    wend = getattr(w, "end", None)
                    wtext = getattr(w, "word", getattr(w, "text", ""))
                    words.append({"word": wtext, "start": wstart, "end": wend})
            segments.append({"start": float(start), "end": float(end), "text": text, "words": words})
        language = info.get("language") if isinstance(info, dict) else getattr(info, "language", language)
    elif isinstance(transcription, dict):
        # some newer APIs may already return dict
        language = transcription.get("language", "unknown")
        for s in transcription.get("segments", []):
            segments.append({
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", 0.0)),
                "text": s.get("text", "").strip(),
                "words": s.get("words", []),
            })
    else:
        raise RuntimeError("Unexpected transcription output format from faster-whisper")

    result = {"language": language or "unknown", "segments": segments}
    print(f"Transcription produced {len(segments)} segments; language={result['language']}")
    return result, audio_path


def run_diarization_map(transcript_segments: List[Dict], audio_path: str, hf_token: Optional[str]) -> Tuple[List[Dict], bool]:
    """
    If pyannote.audio is available and HF token is provided, run diarization pipeline and map
    speaker labels to transcript segments. Returns (segments_with_speakers, diarization_success_flag).
    
    FIXED: Uses correct API for newer pyannote versions (2.1.1+) with compatibility fallback.
    """
    try:
        from pyannote.audio import Pipeline  # type: ignore
    except Exception as e:
        print("⚠️ pyannote.audio not available/importable:", e)
        # fallback: tag default speaker for all
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    if not hf_token:
        print("⚠️ No HF_TOKEN provided — skipping HF gated diarization. Using default speaker labels.")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    try:
        # KEY FIX: Use correct pipeline name and parameter name for auth token
        # Older versions use use_auth_token, newer ones might use token
        # The '@2.1' suffix ensures compatibility with the correct API
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization@2.1",  # Explicit version for API compatibility[citation:8]
            use_auth_token=hf_token  # Parameter name for older API[citation:1][citation:10]
        )
    except Exception as e:
        print(f"⚠️ Failed to load pyannote pipeline from HF: {e}")
        # Fallback: try without version specifier
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", token=hf_token)
        except Exception as e2:
            print(f"⚠️ Fallback pipeline load also failed: {e2}")
            for s in transcript_segments:
                s["speaker"] = "SPEAKER_00"
            return transcript_segments, False

    # Run diarization
    try:
        diarization = pipeline(audio_path)
    except Exception as e:
        print(f"⚠️ Diarization pipeline run failed: {e}")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    # KEY FIX: Updated API for accessing diarization results
    # Newer pyannote versions (2.1.1+) have different API than older ones[citation:1][citation:8]
    # Build list of diarization turns with compatibility for different API versions
    turns = []
    
    try:
        # Method 1: Try the OLD API first (for pyannote.audio < 2.1)
        for turn, _, label in diarization.itertracks(yield_label=True):
            turns.append((turn.start, turn.end, label))
        print("ℹ️ Used legacy itertracks() API")
        
    except AttributeError:
        try:
            # Method 2: Try the NEWER API (pyannote.audio >= 2.1)
            # The diarization object itself is iterable[citation:8]
            for segment, track, label in diarization.iterturns(yield_label=True):
                turns.append((segment.start, segment.end, label))
            print("ℹ️ Used iterturns() API")
            
        except AttributeError:
            try:
                # Method 3: Try the LATEST API - direct iteration
                # In some versions, diarization is directly iterable as (segment, speaker) pairs
                for segment, speaker in diarization:
                    turns.append((segment.start, segment.end, speaker))
                print("ℹ️ Used direct iteration API")
                
            except Exception as e:
                print(f"⚠️ Could not parse diarization output with any known API: {e}")
                for s in transcript_segments:
                    s["speaker"] = "SPEAKER_00"
                return transcript_segments, False

    if not turns:
        print("⚠️ Diarization returned no turns. Falling back to single speaker.")
        for s in transcript_segments:
            s["speaker"] = "SPEAKER_00"
        return transcript_segments, False

    # For each transcript segment, choose the label with maximum overlap
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
        seg["speaker"] = best_label if best_label is not None else "SPEAKER_00"

    print(f"✅ Diarization mapping complete. Found {len(set([t[2] for t in turns]))} speakers.")
    return transcript_segments, True


# -------------------------
# Output serialization
# -------------------------
def save_outputs(result: Dict, audio_file: str) -> Tuple[str, str, str]:
    base = os.path.splitext(audio_file)[0]
    json_path = f"{base}_diarized.json"
    txt_path = f"{base}_transcript.txt"
    srt_path = f"{base}_subtitles.srt"

    # Simplified JSON (store segments as-is)
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"audio_file": audio_file, "language": result.get("language", "unknown"), "segments": result.get("segments", [])}, fh, ensure_ascii=False, indent=2)

    # TXT
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Audio: {audio_file}\nLanguage: {result.get('language', 'unknown')}\n")
        fh.write("=" * 60 + "\n\n")
        for seg in result.get("segments", []):
            speaker = seg.get("speaker", "UNKNOWN")
            fh.write(f"[{format_timestamp(seg.get('start', 0.0))} → {format_timestamp(seg.get('end', 0.0))}] Speaker {speaker}:\n")
            fh.write(seg.get("text", "").strip() + "\n\n")

    # SRT
    with open(srt_path, "w", encoding="utf-8") as fh:
        for idx, seg in enumerate(result.get("segments", []), start=1):
            fh.write(f"{idx}\n")
            fh.write(f"{format_timestamp(seg.get('start', 0.0))} --> {format_timestamp(seg.get('end', 0.0))}\n")
            fh.write(f"[{seg.get('speaker', 'UNKNOWN')}] {seg.get('text', '').strip()}\n\n")

    print(f"Outputs written: {json_path}, {txt_path}, {srt_path}")
    return json_path, txt_path, srt_path


# -------------------------
# Main orchestration
# -------------------------
def main():
    print("=" * 70)
    print("Faster-Whisper + (optional) Pyannote Diarization runner")
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
            # Transcribe
            result, _ = transcribe_with_faster_whisper(audio_path, device="cpu")
            # Try diarization mapping if pyannote available
            segments_with_speakers, diar_ok = run_diarization_map(result.get("segments", []), audio_path, hf)
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