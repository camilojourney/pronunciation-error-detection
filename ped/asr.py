"""ASR wrapper using faster-whisper for speech-to-text transcription."""

from typing import Optional, Dict, Any
import os


def transcribe(audio_path: str, model_size: str = "base", device: Optional[str] = None) -> str:
    """Transcribe an audio file and return text using faster-whisper.

    Args:
        audio_path: Path to the audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        device: Device to use for inference (cpu, cuda, auto)

    Returns:
        Transcribed text as a single string
    """
    from faster_whisper import WhisperModel

    if device is None:
        device = "cpu"  # Default to CPU for compatibility

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Initialize model
    model = WhisperModel(model_size, device=device, compute_type="int8")

    # Transcribe
    segments, info = model.transcribe(audio_path, beam_size=5)

    # Combine all segments into single text
    text = " ".join([segment.text.strip() for segment in segments])

    return text


def transcribe_with_details(
    audio_path: str,
    model_size: str = "base",
    device: Optional[str] = None
) -> Dict[str, Any]:
    """Transcribe with detailed segment information.

    Returns:
        Dictionary containing text, segments, and metadata
    """
    from faster_whisper import WhisperModel

    if device is None:
        device = "cpu"

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = WhisperModel(model_size, device=device, compute_type="int8")
    segments, info = model.transcribe(audio_path, beam_size=5)

    segment_list = []
    full_text = []

    for segment in segments:
        segment_list.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
        full_text.append(segment.text.strip())

    return {
        "text": " ".join(full_text),
        "segments": segment_list,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration
    }
