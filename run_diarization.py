#!/usr/bin/env python3
"""
Audio Transcription & Diarization Script
Simple but powerful: faster-whisper + pyannote.audio
"""

import os
import sys
import json
from datetime import timedelta
from typing import List, Dict, Tuple, Optional

# --------------------------------------------------
# Konfigurasi
# --------------------------------------------------
AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus")
MODEL_SIZE = "large-v3"  # Best for multilingual
DEVICE = "cpu"           # Change to "cuda" if GPU available
COMPUTE_TYPE = "int8"    # Optimized for CPU

# --------------------------------------------------
# Fungsi Utilitas
# --------------------------------------------------
def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp format"""
    td = timedelta(seconds=seconds)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    secs = td.seconds % 60 + td.microseconds / 1_000_000
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")

def find_audio_files() -> List[str]:
    """Find all audio files in current directory"""
    return [f for f in os.listdir(".") 
            if f.lower().endswith(AUDIO_EXTENSIONS)]

# --------------------------------------------------
# Transkripsi dengan faster-whisper
# --------------------------------------------------
def transcribe_audio(audio_path: str) -> Tuple[Dict, str]:
    """
    Transcribe audio using faster-whisper
    Returns: (result_dict, audio_path)
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ faster_whisper not installed")
        print("   Install: pip install faster-whisper")
        sys.exit(1)
    
    print(f"📝 Transcribing: {audio_path}")
    
    # Load model
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    
    # Transcribe with optimized settings
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=300
        )
    )
    
    # Convert segments to list
    segments_list = []
    for segment in segments:
        segments_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip(),
            "confidence": getattr(segment, 'confidence', 0.0),
            "words": [
                {
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                }
                for word in getattr(segment, 'words', [])
            ]
        })
    
    result = {
        "audio_file": audio_path,
        "language": info.language,
        "language_probability": info.language_probability,
        "segments": segments_list
    }
    
    print(f"✅ Transcription complete: {len(segments_list)} segments")
    print(f"   Language: {info.language} ({info.language_probability:.1%})")
    
    return result, audio_path

# --------------------------------------------------
# Diarisasi dengan pyannote.audio
# --------------------------------------------------
def run_diarization(audio_path: str, hf_token: Optional[str]) -> List[Tuple[float, float, str]]:
    """
    Run speaker diarization on audio file
    Returns: List of (start, end, speaker) tuples
    """
    if not hf_token:
        print("⚠️ No HF_TOKEN - skipping diarization")
        return []
    
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        print("⚠️ pyannote.audio not installed")
        print("   Install: pip install pyannote.audio")
        return []
    
    try:
        # Load pipeline
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token
        )
        
        print("🔊 Running diarization...")
        diarization_result = pipeline(audio_path)
        
        # Extract speaker turns
        turns = []
        
        # Try different attribute names for different pyannote versions
        if hasattr(diarization_result, 'annotation'):
            annotation = diarization_result.annotation
        elif hasattr(diarization_result, 'speaker_diarization'):
            annotation = diarization_result.speaker_diarization
        else:
            print("⚠️ Cannot extract diarization data")
            return []
        
        for segment, track, speaker in annotation.itertracks(yield_label=True):
            turns.append((segment.start, segment.end, speaker))
        
        print(f"✅ Diarization complete: {len(turns)} speaker turns")
        print(f"   Speakers: {len(set([s[2] for s in turns]))} unique")
        
        return turns
        
    except Exception as e:
        print(f"❌ Diarization failed: {e}")
        return []

# --------------------------------------------------
# Map speaker to transcription segments
# --------------------------------------------------
def map_speakers_to_segments(segments: List[Dict], speaker_turns: List[Tuple]) -> List[Dict]:
    """Map speaker labels to transcription segments"""
    
    def calculate_overlap(seg_start, seg_end, spk_start, spk_end):
        overlap_start = max(seg_start, spk_start)
        overlap_end = min(seg_end, spk_end)
        return max(0, overlap_end - overlap_start)
    
    for segment in segments:
        best_speaker = "SPEAKER_00"
        best_overlap = 0
        
        for spk_start, spk_end, speaker in speaker_turns:
            overlap = calculate_overlap(
                segment["start"], segment["end"],
                spk_start, spk_end
            )
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker
        
        # Only assign if there's meaningful overlap (>0.05s)
        segment["speaker"] = best_speaker if best_overlap > 0.05 else "SPEAKER_00"
    
    return segments

# --------------------------------------------------
# Save Output Files
# --------------------------------------------------
def save_output_files(result: Dict, base_name: str):
    """Save results in multiple formats"""
    
    # 1. JSON (full data)
    json_path = f"{base_name}_diarized.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 2. Text (readable format)
    txt_path = f"{base_name}_transcript.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Audio: {result['audio_file']}\n")
        f.write(f"Language: {result['language']} ({result.get('language_probability', 0):.1%})\n")
        f.write("=" * 60 + "\n\n")
        
        for segment in result["segments"]:
            f.write(f"[{format_timestamp(segment['start'])} → {format_timestamp(segment['end'])}] ")
            f.write(f"{segment['speaker']}:\n")
            f.write(f"{segment['text']}\n\n")
    
    # 3. SRT (subtitles)
    srt_path = f"{base_name}_subtitles.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(result["segments"], 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(segment['start'])} --> {format_timestamp(segment['end'])}\n")
            f.write(f"[{segment['speaker']}] {segment['text']}\n\n")
    
    print(f"📁 Output files saved:")
    print(f"   • {json_path}")
    print(f"   • {txt_path}")
    print(f"   • {srt_path}")
    
    return json_path, txt_path, srt_path

# --------------------------------------------------
# Main Function
# --------------------------------------------------
def main():
    """Main execution function"""
    
    print("=" * 60)
    print("🎙️  Audio Transcription & Diarization")
    print("=" * 60)
    
    # Check for Hugging Face token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("ℹ️  HF_TOKEN not set (diarization will be skipped)")
        print("   Set: export HF_TOKEN='your_token_here'")
    
    # Find audio files
    audio_files = find_audio_files()
    if not audio_files:
        print("❌ No audio files found")
        print(f"   Supported formats: {', '.join(AUDIO_EXTENSIONS)}")
        return
    
    print(f"📂 Found {len(audio_files)} audio file(s):")
    for f in audio_files:
        print(f"   • {f}")
    
    # Process each file
    for audio_file in audio_files:
        print(f"\n{'='*40}")
        print(f"🎯 Processing: {audio_file}")
        
        try:
            # Step 1: Transcribe
            result, _ = transcribe_audio(audio_file)
            
            # Step 2: Diarize (if token available)
            speaker_turns = run_diarization(audio_file, hf_token)
            
            # Step 3: Map speakers to segments
            if speaker_turns:
                result["segments"] = map_speakers_to_segments(
                    result["segments"], speaker_turns
                )
                result["diarization_success"] = True
            else:
                # Add default speaker if diarization failed
                for segment in result["segments"]:
                    segment["speaker"] = "SPEAKER_00"
                result["diarization_success"] = False
            
            # Step 4: Save outputs
            base_name = os.path.splitext(audio_file)[0]
            save_output_files(result, base_name)
            
        except Exception as e:
            print(f"❌ Error processing {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("✅ Processing complete!")
    print("=" * 60)

# --------------------------------------------------
# Entry Point
# --------------------------------------------------
if __name__ == "__main__":
    main()