#!/usr/bin/env python3
"""
audio_processor.py
Simple & Powerful Audio Transcription + Diarization
Compatible with pyannote.audio 4.0.3
"""

import os
import sys
import json
from datetime import timedelta
from typing import List, Dict, Tuple, Optional

# ==================== KONFIGURASI ====================
AUDIO_EXTS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")
WHISPER_MODEL = "large-v3"  # Best multilingual model
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

# ==================== FUNGSI UTAMA ====================

def find_audio_files() -> List[str]:
    """Cari semua file audio di direktori saat ini"""
    return [f for f in os.listdir(".") 
            if f.lower().endswith(AUDIO_EXTS)]


def format_timestamp(seconds: float) -> str:
    """Konversi detik ke format timestamp SRT"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")


def transcribe_audio(audio_path: str) -> Dict:
    """Transkripsi audio menggunakan faster-whisper"""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ ERROR: faster-whisper tidak terinstal")
        print("   Install dengan: pip install faster-whisper")
        sys.exit(1)
    
    print(f"🎙️  Transcribing: {audio_path}")
    
    # Load model
    model = WhisperModel(WHISPER_MODEL, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    # Transkripsi dengan parameter optimal
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 300,
            "threshold": 0.5
        }
    )
    
    # Konversi ke list
    segments_list = []
    for seg in segments:
        segments_list.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text.strip(),
            "confidence": getattr(seg, "confidence", 0.0),
            "speaker": "SPEAKER_00"  # Default
        })
    
    print(f"✅ Transkripsi selesai: {len(segments_list)} segmen")
    print(f"   Bahasa terdeteksi: {info.language} ({info.language_probability:.1%})")
    
    return {
        "audio_file": audio_path,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segments_list
    }


def diarize_audio(audio_path: str, hf_token: Optional[str]) -> List[Tuple]:
    """Diarisasi pembicara menggunakan pyannote.audio 4.0.3"""
    if not hf_token:
        print("⚠️  HF_TOKEN tidak ada, skip diarisasi")
        return []
    
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("⚠️  pyannote.audio tidak terinstal")
        return []
    
    try:
        print("👥  Running diarization...")
        
        # Load pipeline - gunakan parameter yang benar untuk pyannote.audio 4.0.3
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )
        except TypeError as e:
            # Fallback untuk parameter lama jika diperlukan
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
        
        # Run diarization
        diarization_result = pipeline(audio_path)
        
        # Ekstrak speaker turns dari DiarizeOutput (pyannote.audio 4.0.3)
        turns = []
        
        # Method 1: Coba akses annotation langsung
        if hasattr(diarization_result, 'annotation'):
            annotation = diarization_result.annotation
            print("ℹ️  Menggunakan 'annotation' attribute")
        elif hasattr(diarization_result, 'speaker_diarization'):
            annotation = diarization_result.speaker_diarization
            print("ℹ️  Menggunakan 'speaker_diarization' attribute")
        else:
            # Method 2: Coba iterasi langsung
            print("ℹ️  Mencoba iterasi langsung")
            annotation = diarization_result
        
        # Extract turns dari annotation
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            turns.append((segment.start, segment.end, speaker))
        
        print(f"✅ Diarisasi selesai: {len(turns)} speaker turns ditemukan")
        unique_speakers = len(set([t[2] for t in turns]))
        print(f"   Jumlah pembicara unik: {unique_speakers}")
        
        return turns
        
    except Exception as e:
        print(f"❌ Diarisasi gagal: {str(e)}")
        return []


def map_speakers(segments: List[Dict], speaker_turns: List[Tuple]) -> List[Dict]:
    """Map pembicara ke segmen transkripsi"""
    if not speaker_turns:
        return segments
    
    for seg in segments:
        best_speaker = "SPEAKER_00"
        best_overlap = 0
        
        for spk_start, spk_end, speaker in speaker_turns:
            # Hitung overlap
            overlap_start = max(seg["start"], spk_start)
            overlap_end = min(seg["end"], spk_end)
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        
        # Assign speaker jika ada overlap signifikan (>0.1 detik)
        seg["speaker"] = best_speaker if best_overlap > 0.1 else "SPEAKER_00"
    
    return segments


def save_outputs(result: Dict, base_name: str):
    """Simpan hasil dalam 3 format"""
    
    # 1. JSON (data lengkap)
    json_file = f"{base_name}_diarized.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 2. TXT (format terbaca)
    txt_file = f"{base_name}_transcript.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"File Audio: {result['audio_file']}\n")
        f.write(f"Bahasa: {result['language']} ")
        f.write(f"(confidence: {result.get('language_probability', 0):.1%})\n")
        f.write("=" * 60 + "\n\n")
        
        for seg in result["segments"]:
            f.write(f"[{format_timestamp(seg['start'])} → {format_timestamp(seg['end'])}] ")
            f.write(f"{seg['speaker']}:\n")
            f.write(f"{seg['text']}\n\n")
    
    # 3. SRT (subtitle)
    srt_file = f"{base_name}_subtitles.srt"
    with open(srt_file, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg['start'])} --> {format_timestamp(seg['end'])}\n")
            f.write(f"[{seg['speaker']}] {seg['text']}\n\n")
    
    print(f"💾 Output disimpan:")
    print(f"   📄 {json_file}")
    print(f"   📝 {txt_file}")
    print(f"   🎬 {srt_file}")


def main():
    """Fungsi utama"""
    print("=" * 60)
    print("🎧 AUDIO TRANSCRIPTION & DIARIZATION")
    print("=" * 60)
    print("Versi: Faster-Whisper + Pyannote.audio 4.0.3")
    print("=" * 60)
    
    # Cek HF Token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ℹ️  Catatan: HF_TOKEN tidak diset")
        print("   Diarisasi akan dilewati, hanya transkripsi")
        print("   Set HF_TOKEN untuk diarisasi pembicara")
    
    # Cari file audio
    audio_files = find_audio_files()
    if not audio_files:
        print("❌ Tidak ada file audio ditemukan")
        print(f"   Format yang didukung: {', '.join(AUDIO_EXTS)}")
        return
    
    print(f"📁 Ditemukan {len(audio_files)} file audio:")
    for f in audio_files:
        print(f"   • {f}")
    
    # Proses setiap file
    processed = 0
    for audio_file in audio_files:
        print(f"\n{'='*40}")
        print(f"🔄 Memproses: {audio_file}")
        
        try:
            # 1. Transkripsi
            result = transcribe_audio(audio_file)
            
            # 2. Diarisasi (jika token ada)
            speaker_turns = diarize_audio(audio_file, hf_token)
            
            # 3. Map pembicara
            if speaker_turns:
                result["segments"] = map_speakers(result["segments"], speaker_turns)
                result["diarization"] = "success"
            else:
                result["diarization"] = "skipped"
            
            # 4. Simpan hasil
            base_name = os.path.splitext(audio_file)[0]
            save_outputs(result, base_name)
            
            processed += 1
            
        except Exception as e:
            print(f"❌ Error memproses {audio_file}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print(f"✅ SELESAI: {processed}/{len(audio_files)} file berhasil diproses")
    print("=" * 60)


if __name__ == "__main__":
    main()