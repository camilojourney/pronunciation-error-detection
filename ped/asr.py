"""ASR wrapper placeholder.

Future: integrate faster-whisper or openai-whisper. For now, this module provides
an interface and a stub implementation that raises NotImplementedError.
"""

from typing import Optional


def transcribe(audio_path: str, model_size: str = "base", device: Optional[str] = None) -> str:
    """Transcribe an audio file and return text.

    TODO: implement with faster-whisper.
    """
    raise NotImplementedError("ASR not implemented yet. Install faster-whisper and implement.")
