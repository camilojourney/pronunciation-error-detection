"""Pipeline orchestrator for PED."""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from .text import tokenize, align_tokens
from .metrics import wer
from .errors import analyze_errors, ErrorAnalysis
from .asr import transcribe


@dataclass
class TextPipelineResult:
    ref: str
    hyp: str
    wer: float
    operations: list
    error_analysis: Optional[ErrorAnalysis] = None


@dataclass
class AudioPipelineResult:
    audio_path: str
    ref: str
    hyp: str
    wer: float
    operations: list
    error_analysis: ErrorAnalysis


def run_text_pipeline(ref_text: str, hyp_text: str, analyze: bool = True) -> TextPipelineResult:
    """Run text-based error detection pipeline.

    Args:
        ref_text: Reference (correct) text
        hyp_text: Hypothesis (transcribed or spoken) text
        analyze: Whether to perform detailed error analysis

    Returns:
        TextPipelineResult with alignment and optional error analysis
    """
    ref_toks = tokenize(ref_text)
    hyp_toks = tokenize(hyp_text)
    ops = align_tokens(ref_toks, hyp_toks)
    score = wer(ref_toks, hyp_toks)

    error_analysis = None
    if analyze:
        error_analysis = analyze_errors(ref_toks, hyp_toks)

    return TextPipelineResult(
        ref=ref_text,
        hyp=hyp_text,
        wer=score,
        operations=ops,
        error_analysis=error_analysis
    )


def run_audio_pipeline(
    audio_path: str,
    reference_text: str,
    model_size: str = "base",
    device: Optional[str] = None
) -> AudioPipelineResult:
    """Run complete audio-based pronunciation error detection pipeline.

    Args:
        audio_path: Path to audio file
        reference_text: Expected/reference text
        model_size: Whisper model size
        device: Compute device (cpu/cuda)

    Returns:
        AudioPipelineResult with transcription and error analysis
    """
    # Transcribe audio
    hyp_text = transcribe(audio_path, model_size=model_size, device=device)

    # Tokenize
    ref_toks = tokenize(reference_text)
    hyp_toks = tokenize(hyp_text)

    # Align and compute metrics
    ops = align_tokens(ref_toks, hyp_toks)
    score = wer(ref_toks, hyp_toks)

    # Analyze errors
    error_analysis = analyze_errors(ref_toks, hyp_toks)

    return AudioPipelineResult(
        audio_path=audio_path,
        ref=reference_text,
        hyp=hyp_text,
        wer=score,
        operations=ops,
        error_analysis=error_analysis
    )


def batch_process_audio(
    audio_files: List[Dict[str, str]],
    model_size: str = "base",
    device: Optional[str] = None
) -> List[AudioPipelineResult]:
    """Process multiple audio files.

    Args:
        audio_files: List of dicts with 'audio_path' and 'reference_text'
        model_size: Whisper model size
        device: Compute device

    Returns:
        List of AudioPipelineResult objects
    """
    results = []
    for item in audio_files:
        try:
            result = run_audio_pipeline(
                audio_path=item["audio_path"],
                reference_text=item["reference_text"],
                model_size=model_size,
                device=device
            )
            results.append(result)
        except Exception as e:
            print(f"Error processing {item['audio_path']}: {e}")
            continue

    return results
