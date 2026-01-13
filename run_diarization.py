#!/usr/bin/env python3
"""
run_diarization.py

Stable transcription + optional speaker diarization runner for CI (GitHub Actions)
- Uses faster-whisper for transcription (offline, CPU-friendly)
- Optional pyannote.audio speaker diarization if HF_TOKEN provided
- Detects language first → switches to language-specific model for accuracy
- Produces outputs: <audio>_diarized.json, <audio>_transcript.txt, <audio>_subtitles.srt
"""

from __future__ import annotations
import os, sys, json, subprocess, traceback
from datetime import timedelta
from typing import List, Dict, Tuple, Optional

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")

# -------------------------
# Helpers
# -------------------------
def run(cmd: List[str]) -> subprocess.CompletedProcess:
    print("[RUN]", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.stdout: print(proc.stdout, end="")
    if proc.stderr: print(proc.stderr, end="", file=sys.stderr)
    return proc

def format_timestamp(seconds: float) -> str:
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
    flags = { "numpy": False, "torch": False, "faster_whisper": False, "pyannote": False, "onnxruntime": False, "nltk": False }

    to_ensure = [("numpy", ["numpy<2"]), ("nltk", ["nltk==3.8.1"])]
    for name, specs in to_ensure:
        try: __import__(name); flags[name]=True; print(f"✅ {name} available")
        except Exception:
            for spec in specs:
                cp = run([sys.executable, "-m", "pip", "install", spec])
                if cp.returncode==0:
                    try: __import__(name); flags[name]=True; print(f"✅ Installed & imported {name} ({spec})"); break
                    except Exception as e: print(f"⚠️ Installed {spec} but import failed: {e}")
                else: print(f"⚠️ pip install {spec} failed (rc={cp.returncode})")

    try: import torch; flags["torch"]=True; print("✅ torch available:", torch.__version__)
    except Exception: print("⚠️ torch not available. Preinstall for best results.")

    try: import faster_whisper; flags["faster_whisper"]=True; print("✅ faster_whisper available")
    except Exception:
        print("⚠️ faster_whisper not available. Attempting install...")
        cp = run([sys.executable, "-m", "pip", "install", "faster-whisper"])
        if cp.returncode==0:
            try: import faster_whisper; flags["faster_whisper"]=True; print("✅ faster_whisper installed")
            except Exception as e: print("⚠️ faster_whisper import failed after install:", e)

    try: import onnxruntime; flags["onnxruntime"]=True; print("✅ onnxruntime available")
    except Exception: print("⚠️ onnxruntime not available (optional)")

    pyannote_versions_to_try = ["4.0.3", "4.1.1", "3.1.1", "2.1.1"]
    try:
        import importlib.util
        if importlib.util.find_spec("pyannote.audio") is not None: flags["pyannote"]=True; print("✅ pyannote.audio already installed")
        else:
            for v in pyannote_versions_to_try:
                print(f"ℹ️ Attempting pip install pyannote.audio=={v}")
                cp = run([sys.executable, "-m", "pip", "install", f"pyannote.audio=={v}"])
                if cp.returncode==0:
                    try: __import__("pyannote.audio"); flags["pyannote"]=True; print(f"✅ Installed pyannote.audio=={v}"); break
                    except Exception as e: print(f"⚠️ Installed pyannote.audio=={v} but import failed: {e}")
                else: print(f"⚠️ pip install pyannote.audio=={v} failed")
    except Exception as e: print("⚠️ Exception while probing pyannote:", e)

    print("---- preflight summary ----")
    for k,v in flags.items(): print(f"{k}: {'OK' if v else 'MISSING/FAIL'}")
    print("---------------------------")
    return flags

# -------------------------
# Hugging Face helpers
# -------------------------
def get_hf_token() -> Optional[str]:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

def setup_hf_cli_login(token: Optional[str]) -> None:
    if not token: print("ℹ️ No HF_TOKEN found; gated HF models will not be accessible."); return
    try:
        from huggingface_hub import login
        login(token=token, add_to_git_credential=False)
        print("✅ Hugging Face login succeeded")
    except Exception as e: print("⚠️ huggingface_hub.login failed:", e)

# -------------------------
# Language detection & mapping
# -------------------------
def detect_language(audio_path: str, device: str = "cpu") -> str:
    from faster_whisper import WhisperModel
    model = WhisperModel(os.getenv("FW_MODEL","large-v3"), device=device, compute_type="int8")
    segments, info = model.transcribe(audio_path, language=None, beam_size=2, word_timestamps=False)
    lang = getattr(info,"language",None) or info.get("language",None) or "unknown"
    print(f"🔍 Detected language: {lang}")
    return lang

LANG_MAP = {
    "en":"english","ar":"arabic","az":"azerbaijani","bn":"bengali","bg":"bulgarian",
    "ca":"catalan","zh":"chinese","hr":"croatian","cs":"czech","da":"danish",
    "nl":"dutch","et":"estonian","fi":"finnish","fr":"french","de":"german",
    "el":"greek","gu":"gujarati","ht":"haitian","he":"hebrew","hi":"hindi",
    "hu":"hungarian","is":"icelandic","id":"indonesian","it":"italian","ja":"japanese",
    "kn":"kannada","kk":"kazakh","ko":"korean","lv":"latvian","lt":"lithuanian",
    "mk":"macedonian","ms":"malay","ml":"malayalam","mr":"marathi","mn":"mongolian",
    "ne":"nepali","no":"norwegian","pa":"punjabi","fa":"persian","pl":"polish",
    "pt":"portuguese","ro":"romanian","ru":"russian","sr":"serbian","si":"sinhala",
    "sk":"slovak","sl":"slovenian","es":"spanish","sv":"swedish","ta":"tamil",
    "te":"telugu","th":"thai","tr":"turkish","uk":"ukrainian","ur":"urdu",
    "uz":"uzbek","vi":"vietnamese"
}

# -------------------------
# Transcription with language-specific model
# -------------------------
def transcribe_with_faster_whisper(audio_path: str, device: str = "cpu") -> Tuple[Dict,str]:
    from faster_whisper import WhisperModel
    detected_lang = detect_language(audio_path, device=device)
    specific_lang = LANG_MAP.get(detected_lang.lower(), None)
    if specific_lang: print(f"⚡ Using language-specific model: {specific_lang}")
    else: print(f"⚠️ Unknown language '{detected_lang}', fallback to multilingual"); specific_lang=None

    model_size = os.getenv("FW_MODEL","large-v3")
    compute_type = os.getenv("FW_COMPUTE","int8")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    kwargs = {"beam_size":5,"word_timestamps":True}
    if specific_lang: kwargs["language"]=specific_lang
    transcription = model.transcribe(audio_path, **kwargs)

    segments=[]
    language = specific_lang or detected_lang or "unknown"
    if isinstance(transcription, tuple) and len(transcription)>=1:
        segs = transcription[0]; info = transcription[1] if len(transcription)>1 else {}
        for s in segs:
            start = getattr(s,"start",getattr(s,"start_time",0.0))
            end = getattr(s,"end",getattr(s,"end_time",start))
            text = getattr(s,"text",getattr(s,"content","")).strip()
            words=[]
            wlist=getattr(s,"words",None)
            if wlist:
                for w in wlist:
                    wstart=getattr(w,"start",None)
                    wend=getattr(w,"end",None)
                    wtext=getattr(w,"word",getattr(w,"text",""))
                    words.append({"word":wtext,"start":wstart,"end":wend})
            segments.append({"start":float(start),"end":float(end),"text":text,"words":words})
        language = info.get("language") if isinstance(info, dict) else getattr(info,"language",language)
    elif isinstance(transcription, dict):
        language = transcription.get("language",language)
        for s in transcription.get("segments",[]):
            segments.append({"start":float(s.get("start",0.0)),"end":float(s.get("end",0.0)),"text":s.get("text","").strip(),"words":s.get("words",[])})
    result={"language":language,"segments":segments}
    print(f"✅ Transcription done: {len(segments)} segments, language={result['language']}")
    return result,audio_path

# -------------------------
# Diarization mapping
# -------------------------
def run_diarization_map(transcript_segments: List[Dict], audio_path: str, hf_token: Optional[str]) -> Tuple[List[Dict], bool]:
    try:
        from pyannote.audio import Pipeline
        import pyannote.audio
    except Exception as e:
        for s in transcript_segments: s["speaker"]="SPEAKER_00"
        return transcript_segments, False

    if not hf_token:
        for s in transcript_segments: s["speaker"]="SPEAKER_00"
        return transcript_segments, False

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token=hf_token)
    except Exception:
        for s in transcript_segments: s["speaker"]="SPEAKER_00"
        return transcript_segments, False

    try: diarization_result = pipeline(audio_path)
    except Exception:
        for s in transcript_segments: s["speaker"]="SPEAKER_00"
        return transcript_segments, False

    turns=[]
    try:
        if hasattr(diarization_result,'speaker_diarization'):
            annotation = diarization_result.speaker_diarization
            for segment, track, label in annotation.itertracks(yield_label=True):
                turns.append((segment.start, segment.end, label))
        elif hasattr(diarization_result,'exclusive_speaker_diarization'):
            annotation = diarization_result.exclusive_speaker_diarization
            for segment, track, label in annotation.itertracks(yield_label=True):
                turns.append((segment.start, segment.end, label))
        elif hasattr(diarization_result,'__iter__'):
            for segment, track, label in diarization_result.itertracks(yield_label=True):
                turns.append((segment.start, segment.end, label))
    except Exception:
        for s in transcript_segments: s["speaker"]="SPEAKER_00"
        return transcript_segments, False

    if not turns:
        for s in transcript_segments: s["speaker"]="SPEAKER_00"
        return transcript_segments, False

    def overlap(a_start,a_end,b_start,b_end) -> float:
        return max(0.0, min(a_end,b_end)-max(a_start,b_start))

    for seg in transcript_segments:
        s0=float(seg.get("start",0.0))
        e0=float(seg.get("end",s0+0.001))
        best_label=None
        best_ov=0.0
        for t0,t1,lbl in turns:
            ov=overlap(s0,e0,float(t0),float(t1))
            if ov>best_ov: best_ov=ov; best_label=lbl
        seg["speaker"]=best_label if best_label else "SPEAKER_00"

    return transcript_segments, True

# -------------------------
# Save outputs
# -------------------------
def save_outputs(result: Dict, audio_file: str) -> Tuple[str,str,str]:
    base = os.path.splitext(audio_file)[0]
    json_path = f"{base}_diarized.json"
    txt_path = f"{base}_transcript.txt"
    srt_path = f"{base}_subtitles.srt"

    with open(json_path,"w",encoding="utf-8") as fh:
        json.dump({"audio_file":audio_file,"language":result.get("language","unknown"),"segments":result.get("segments",[])}, fh, ensure_ascii=False, indent=2)

    with open(txt_path,"w",encoding="utf-8") as fh:
        fh.write(f"Audio: {audio_file}\nLanguage: {result.get('language','unknown')}\n")
        fh.write("="*60+"\n\n")
        for seg in result.get("segments",[]):
            speaker=seg.get("speaker","UNKNOWN")
            fh.write(f"[{format_timestamp(seg.get('start',0.0))} → {format_timestamp(seg.get('end',0.0))}] Speaker {speaker}:\n")
            fh.write(seg.get("text","").strip()+"\n\n")

    with open(srt_path,"w",encoding="utf-8") as fh:
        for idx,seg in enumerate(result.get("segments",[]),start=1):
            fh.write(f"{idx}\n")
            fh.write(f"{format_timestamp(seg.get('start',0.0))} --> {format_timestamp(seg.get('end',0.0))}\n")
            fh.write(f"[{seg.get('speaker','UNKNOWN')}] {seg.get('text','').strip()}\n\n")

    print(f"Outputs written: {json_path}, {txt_path}, {srt_path}")
    return json_path,txt_path,srt_path

# -------------------------
# Main orchestration
# -------------------------
def main():
    print("="*70)
    print("Faster-Whisper + (optional) Pyannote Diarization runner")
    print("="*70)

    flags = ensure_dependencies()
    hf=get_hf_token()
    setup_hf_cli_login(hf)
    audio_files=find_audio_files()
    if not audio_files:
        print("No audio files found. Supported extensions:",", ".join(AUDIO_EXTENSIONS))
        sys.exit(0)

    processed=0
    for audio_path in audio_files:
        print("\n"+"="*40)
        print("Processing:",audio_path)
        try:
            result,_=transcribe_with_faster_whisper(audio_path,device="cpu")
            segments_with_speakers, diar_ok=run_diarization_map(result.get("segments",[]), audio_path, hf)
            result["segments"]=segments_with_speakers
            result["diarization_ok"]=bool(diar_ok)
            save_outputs(result,audio_path)
            processed+=1
        except Exception as e:
            print("❌ Error processing",audio_path,":",e)
            traceback.print_exc()
            continue

    print("\n"+"="*70)
    print(f"Processing complete. Successfully processed {processed}/{len(audio_files)} audio files.")
    print("="*70)

if __name__=="__main__":
    main()