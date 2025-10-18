"""Pipeline orchestrator placeholder for PED.

Provides a function to run the text-only path and a future audio path.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .text import tokenize, align_tokens
from .metrics import wer


@dataclass
class TextPipelineResult:
    ref: str
    hyp: str
    wer: float
    operations: list


def run_text_pipeline(ref_text: str, hyp_text: str) -> TextPipelineResult:
    ref_toks = tokenize(ref_text)
    hyp_toks = tokenize(hyp_text)
    ops = align_tokens(ref_toks, hyp_toks)
    score = wer(ref_toks, hyp_toks)
    return TextPipelineResult(ref=ref_text, hyp=hyp_text, wer=score, operations=ops)


def run_audio_pipeline(audio_path: str, reference_text: str, **kwargs: Dict[str, Any]):
    """Future: transcribe with ASR, then run text pipeline."""
    raise NotImplementedError
