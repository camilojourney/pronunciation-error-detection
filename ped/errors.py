"""Error detection and classification for pronunciation analysis."""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from collections import Counter
from .text import align_tokens, TokenOp


@dataclass
class PronunciationError:
    """Represents a single pronunciation error."""
    error_type: str  # 'insertion', 'deletion', 'substitution'
    position: int
    reference_word: str
    hypothesis_word: str
    context_before: List[str]
    context_after: List[str]


@dataclass
class ErrorAnalysis:
    """Complete error analysis results."""
    errors: List[PronunciationError]
    total_words: int
    total_errors: int
    insertion_count: int
    deletion_count: int
    substitution_count: int
    error_rate: float
    common_errors: List[Tuple[str, str, int]]  # (ref, hyp, count)


def classify_errors(ref_tokens: List[str], hyp_tokens: List[str]) -> List[PronunciationError]:
    """Classify pronunciation errors from aligned tokens.

    Args:
        ref_tokens: Reference (correct) tokens
        hyp_tokens: Hypothesis (ASR output) tokens

    Returns:
        List of PronunciationError objects
    """
    ops = align_tokens(ref_tokens, hyp_tokens)
    errors = []
    position = 0

    for i, op in enumerate(ops):
        if op.op == "equal":
            position += len(op.ref)
            continue

        # Get context
        context_before = []
        context_after = []

        # Look back for context
        for j in range(i - 1, max(-1, i - 3), -1):
            if ops[j].op == "equal" and ops[j].ref:
                context_before = ops[j].ref[-2:] + context_before

        # Look forward for context
        for j in range(i + 1, min(len(ops), i + 3)):
            if ops[j].op == "equal" and ops[j].ref:
                context_after += ops[j].ref[:2]

        if op.op == "replace":
            # Substitution error
            for ref_word, hyp_word in zip(op.ref, op.hyp):
                errors.append(PronunciationError(
                    error_type="substitution",
                    position=position,
                    reference_word=ref_word,
                    hypothesis_word=hyp_word,
                    context_before=context_before[-2:],
                    context_after=context_after[:2]
                ))
                position += 1
        elif op.op == "delete":
            # Deletion error (word in reference but not in hypothesis)
            for ref_word in op.ref:
                errors.append(PronunciationError(
                    error_type="deletion",
                    position=position,
                    reference_word=ref_word,
                    hypothesis_word="<deleted>",
                    context_before=context_before[-2:],
                    context_after=context_after[:2]
                ))
                position += 1
        elif op.op == "insert":
            # Insertion error (word in hypothesis but not in reference)
            for hyp_word in op.hyp:
                errors.append(PronunciationError(
                    error_type="insertion",
                    position=position,
                    reference_word="<none>",
                    hypothesis_word=hyp_word,
                    context_before=context_before[-2:],
                    context_after=context_after[:2]
                ))

    return errors


def analyze_errors(ref_tokens: List[str], hyp_tokens: List[str]) -> ErrorAnalysis:
    """Perform complete error analysis.

    Args:
        ref_tokens: Reference tokens
        hyp_tokens: Hypothesis tokens

    Returns:
        ErrorAnalysis object with statistics
    """
    errors = classify_errors(ref_tokens, hyp_tokens)

    insertion_count = sum(1 for e in errors if e.error_type == "insertion")
    deletion_count = sum(1 for e in errors if e.error_type == "deletion")
    substitution_count = sum(1 for e in errors if e.error_type == "substitution")

    total_errors = len(errors)
    total_words = len(ref_tokens)
    error_rate = total_errors / max(1, total_words)

    # Find common error patterns
    error_pairs = []
    for e in errors:
        if e.error_type == "substitution":
            error_pairs.append((e.reference_word, e.hypothesis_word))

    common_errors = [(ref, hyp, count) for (ref, hyp), count in Counter(error_pairs).most_common(10)]

    return ErrorAnalysis(
        errors=errors,
        total_words=total_words,
        total_errors=total_errors,
        insertion_count=insertion_count,
        deletion_count=deletion_count,
        substitution_count=substitution_count,
        error_rate=error_rate,
        common_errors=common_errors
    )


def get_error_summary(analysis: ErrorAnalysis) -> Dict[str, any]:
    """Get a summary dictionary of error analysis.

    Args:
        analysis: ErrorAnalysis object

    Returns:
        Dictionary with summary statistics
    """
    return {
        "total_words": analysis.total_words,
        "total_errors": analysis.total_errors,
        "error_rate": analysis.error_rate,
        "insertions": analysis.insertion_count,
        "deletions": analysis.deletion_count,
        "substitutions": analysis.substitution_count,
        "common_substitutions": analysis.common_errors[:5]
    }
