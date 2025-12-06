"""
Unified Pronunciation Error Detection Pipeline
===============================================

This module combines Whisper ASR and MFA forced alignment into a single
pipeline for detecting and analyzing pronunciation errors.

Pipeline Architecture:
    AUDIO (L2 speaker)
         │
         ▼
    ┌─────────────┐
    │   WHISPER   │  → Gets correct WORDS (robust to accent)
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │     MFA     │  → Extracts actual L2 PHONEMES from audio
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  NATIVE IPA │  → Generates reference phonemes from dictionary
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  COMPARATOR │  → Finds phoneme differences
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │  FEEDBACK   │  → Provides actionable pronunciation feedback
    └─────────────┘

Author: Camilo Martinez
Course: Natural Language Processing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import difflib

# Import our existing modules
from analysis_utils import (
    transcribe_audio,
    word_to_phonemes,
    arpabet_to_ipa,
    format_phonemes_ipa,
    clean_text,
    tokenize
)
from forced_alignment import (
    get_phonemes_from_audio,
    mfa_phone_to_ipa,
    UtteranceAlignment,
    WordAlignment,
    PhonemeInterval
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PhonemeError:
    """Represents a single phoneme-level error."""
    error_type: str         # 'substitution', 'deletion', 'insertion'
    expected_phoneme: str   # Expected IPA phoneme
    actual_phoneme: str     # Actual IPA phoneme (empty for deletion)
    position: int           # Position in the word
    severity: str           # 'high', 'medium', 'low'
    explanation: str        # Human-readable explanation


@dataclass
class WordAnalysis:
    """Complete analysis for a single word."""
    word: str                           # The word
    expected_ipa: str                   # Expected IPA from dictionary
    actual_ipa: str                     # Actual IPA from MFA
    expected_phonemes: List[str]        # List of expected phonemes
    actual_phonemes: List[str]          # List of actual phonemes
    errors: List[PhonemeError]          # List of detected errors
    is_correct: bool                    # Whether pronunciation is correct
    accuracy_score: float               # 0.0 to 1.0
    timing_ms: Optional[float] = None   # Duration in milliseconds


@dataclass
class PronunciationResult:
    """Complete result from the pronunciation analysis pipeline."""
    audio_path: str
    whisper_transcript: str             # What Whisper heard (words)
    expected_text: Optional[str]        # Original expected text (if provided)
    word_analyses: List[WordAnalysis]   # Per-word analysis
    overall_accuracy: float             # 0.0 to 1.0
    total_phoneme_errors: int
    error_summary: Dict[str, int]       # {'substitution': n, 'deletion': n, ...}
    feedback: List[str]                 # List of feedback messages
    raw_mfa_alignment: Optional[UtteranceAlignment] = None


# ============================================================================
# PHONEME COMPARISON ENGINE
# ============================================================================

# Phoneme similarity groups (for severity assessment)
VOWEL_PHONEMES = {'i', 'ɪ', 'e', 'ɛ', 'æ', 'ə', 'ʌ', 'ɑ', 'ɔ', 'o', 'ʊ', 'u', 'ɚ'}
DIPHTHONGS = {'eɪ', 'aɪ', 'oʊ', 'aʊ', 'ɔɪ'}

VOICED_CONSONANTS = {'b', 'd', 'ɡ', 'v', 'ð', 'z', 'ʒ', 'dʒ', 'm', 'n', 'ŋ', 'l', 'ɹ', 'w', 'j'}
VOICELESS_CONSONANTS = {'p', 't', 'k', 'f', 'θ', 's', 'ʃ', 'tʃ', 'h'}

# Common L2 English confusion pairs
COMMON_CONFUSIONS = {
    # TH sounds (common for many L1s)
    ('θ', 't'): 'TH→T confusion (common for Spanish, French speakers)',
    ('θ', 's'): 'TH→S confusion (common for German, French speakers)',
    ('ð', 'd'): 'DH→D confusion (common for Spanish, Japanese speakers)',
    ('ð', 'z'): 'DH→Z confusion',

    # R/L confusion (common for East Asian speakers)
    ('ɹ', 'l'): 'R/L confusion (common for Japanese, Korean, Chinese speakers)',
    ('l', 'ɹ'): 'L/R confusion (common for Japanese, Korean, Chinese speakers)',

    # Vowel confusions
    ('ɪ', 'i'): 'Short I/Long E confusion (ship vs sheep)',
    ('i', 'ɪ'): 'Long E/Short I confusion (sheep vs ship)',
    ('æ', 'ɛ'): 'Short A/E confusion (bad vs bed)',
    ('ʌ', 'ɑ'): 'Short U/AH confusion (cup vs cop)',

    # V/W confusion (common for German, Hindi speakers)
    ('v', 'w'): 'V/W confusion',
    ('w', 'v'): 'W/V confusion',

    # Final consonant cluster reduction
    ('d', ''): 'Final D deletion (common in consonant clusters)',
    ('t', ''): 'Final T deletion (common in consonant clusters)',
    ('s', ''): 'Final S deletion',
    ('z', ''): 'Final Z deletion',
}


def align_phoneme_sequences(
    expected: List[str],
    actual: List[str]
) -> List[Tuple[str, str, str]]:
    """
    Align two phoneme sequences using dynamic programming.

    Returns list of (operation, expected_phoneme, actual_phoneme) tuples.
    Operations: 'equal', 'substitution', 'deletion', 'insertion'
    """
    matcher = difflib.SequenceMatcher(None, expected, actual)
    alignments = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            for i, j in zip(range(i1, i2), range(j1, j2)):
                alignments.append(('equal', expected[i], actual[j]))
        elif tag == 'replace':
            # Handle replace as substitution
            for i, j in zip(range(i1, i2), range(j1, j2)):
                alignments.append(('substitution', expected[i], actual[j]))
            # Handle length mismatches
            if i2 - i1 > j2 - j1:
                # More expected than actual = deletions
                for i in range(i1 + (j2 - j1), i2):
                    alignments.append(('deletion', expected[i], ''))
            elif j2 - j1 > i2 - i1:
                # More actual than expected = insertions
                for j in range(j1 + (i2 - i1), j2):
                    alignments.append(('insertion', '', actual[j]))
        elif tag == 'delete':
            for i in range(i1, i2):
                alignments.append(('deletion', expected[i], ''))
        elif tag == 'insert':
            for j in range(j1, j2):
                alignments.append(('insertion', '', actual[j]))

    return alignments


def assess_error_severity(expected: str, actual: str, error_type: str) -> str:
    """
    Assess the severity of a phoneme error.

    Returns: 'high', 'medium', or 'low'
    """
    if error_type == 'equal':
        return 'none'

    if error_type == 'deletion':
        # Missing phonemes are usually high severity
        if expected in VOICED_CONSONANTS or expected in VOICELESS_CONSONANTS:
            return 'high'
        return 'medium'

    if error_type == 'insertion':
        # Extra phonemes are medium severity
        return 'medium'

    # Substitution severity depends on how different the phonemes are
    pair = (expected, actual)
    reverse_pair = (actual, expected)

    # Check if it's a common L2 confusion (lower severity - expected error)
    if pair in COMMON_CONFUSIONS or reverse_pair in COMMON_CONFUSIONS:
        return 'medium'

    # Vowel to vowel substitution is medium
    if expected in VOWEL_PHONEMES and actual in VOWEL_PHONEMES:
        return 'medium'

    # Consonant voicing change is medium
    if (expected in VOICED_CONSONANTS and actual in VOICELESS_CONSONANTS) or \
       (expected in VOICELESS_CONSONANTS and actual in VOICED_CONSONANTS):
        return 'medium'

    # Different phoneme classes = high severity
    return 'high'


def generate_error_explanation(expected: str, actual: str, error_type: str) -> str:
    """Generate a human-readable explanation for an error."""

    if error_type == 'deletion':
        return f"Missing sound /{expected}/. This sound should be pronounced."

    if error_type == 'insertion':
        return f"Extra sound /{actual}/ was added."

    if error_type == 'substitution':
        pair = (expected, actual)
        reverse_pair = (actual, expected)

        # Check for known confusion
        if pair in COMMON_CONFUSIONS:
            return COMMON_CONFUSIONS[pair]
        if reverse_pair in COMMON_CONFUSIONS:
            return COMMON_CONFUSIONS[reverse_pair]

        return f"Sound /{expected}/ was pronounced as /{actual}/."

    return ""


def compare_word_phonemes(
    word: str,
    expected_phonemes: List[str],
    actual_phonemes: List[str]
) -> Tuple[List[PhonemeError], float]:
    """
    Compare expected vs actual phonemes for a word.

    Returns:
        - List of PhonemeError objects
        - Accuracy score (0.0 to 1.0)
    """
    errors = []

    # Align the sequences
    alignments = align_phoneme_sequences(expected_phonemes, actual_phonemes)

    correct_count = 0
    position = 0

    for op, exp, act in alignments:
        if op == 'equal':
            correct_count += 1
        else:
            severity = assess_error_severity(exp, act, op)
            explanation = generate_error_explanation(exp, act, op)

            errors.append(PhonemeError(
                error_type=op,
                expected_phoneme=exp,
                actual_phoneme=act,
                position=position,
                severity=severity,
                explanation=explanation
            ))
        position += 1

    # Calculate accuracy
    total = len(expected_phonemes) if expected_phonemes else 1
    accuracy = correct_count / total if total > 0 else 1.0

    return errors, accuracy


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def analyze_pronunciation(
    audio_path: str,
    expected_text: Optional[str] = None,
    use_whisper: bool = True,
    whisper_model: str = "base"
) -> PronunciationResult:
    """
    Main entry point: Analyze pronunciation from audio file.

    This combines:
    1. Whisper ASR (to get the words spoken, robust to accent)
    2. MFA forced alignment (to get actual phonemes pronounced)
    3. Dictionary lookup (to get expected phonemes)
    4. Comparison and feedback generation

    Args:
        audio_path: Path to audio file (WAV format preferred)
        expected_text: Optional expected text. If None, uses Whisper output
        use_whisper: Whether to use Whisper for transcription
        whisper_model: Whisper model size ('tiny', 'base', 'small', 'medium')

    Returns:
        PronunciationResult with complete analysis
    """
    audio_path = str(Path(audio_path).absolute())

    # Step 1: Get words from audio using Whisper
    if use_whisper:
        whisper_transcript = transcribe_audio(audio_path, model_size=whisper_model)
        whisper_transcript = clean_text(whisper_transcript)
    else:
        whisper_transcript = expected_text or ""

    # Use Whisper transcript if no expected text provided
    working_text = expected_text if expected_text else whisper_transcript
    working_text = clean_text(working_text)

    # Step 2: Get actual phonemes from audio using MFA
    mfa_alignment = get_phonemes_from_audio(audio_path, working_text)

    if mfa_alignment is None:
        # MFA failed - return partial result
        return PronunciationResult(
            audio_path=audio_path,
            whisper_transcript=whisper_transcript,
            expected_text=expected_text,
            word_analyses=[],
            overall_accuracy=0.0,
            total_phoneme_errors=0,
            error_summary={},
            feedback=["Error: Could not perform forced alignment. Check MFA installation."],
            raw_mfa_alignment=None
        )

    # Step 3 & 4: Compare each word's expected vs actual phonemes
    word_analyses = []
    total_errors = 0
    error_counts = {'substitution': 0, 'deletion': 0, 'insertion': 0}

    for word_align in mfa_alignment.words:
        word = word_align.word.lower()

        # Get expected phonemes from dictionary
        expected_arpabet = word_to_phonemes(word)
        expected_ipa = [arpabet_to_ipa(p) for p in expected_arpabet]

        # Get actual phonemes from MFA
        actual_ipa = [mfa_phone_to_ipa(p.phoneme) for p in word_align.phonemes]

        # Compare phonemes
        errors, accuracy = compare_word_phonemes(word, expected_ipa, actual_ipa)

        # Count errors
        for error in errors:
            error_counts[error.error_type] = error_counts.get(error.error_type, 0) + 1
            total_errors += 1

        # Calculate timing
        timing_ms = None
        if word_align.phonemes:
            timing_ms = (word_align.end_time - word_align.start_time) * 1000

        word_analyses.append(WordAnalysis(
            word=word,
            expected_ipa=f"/{' '.join(expected_ipa)}/",
            actual_ipa=f"/{' '.join(actual_ipa)}/",
            expected_phonemes=expected_ipa,
            actual_phonemes=actual_ipa,
            errors=errors,
            is_correct=len(errors) == 0,
            accuracy_score=accuracy,
            timing_ms=timing_ms
        ))

    # Step 5: Calculate overall accuracy
    if word_analyses:
        overall_accuracy = sum(w.accuracy_score for w in word_analyses) / len(word_analyses)
    else:
        overall_accuracy = 0.0

    # Step 6: Generate feedback
    feedback = generate_feedback(word_analyses, error_counts)

    return PronunciationResult(
        audio_path=audio_path,
        whisper_transcript=whisper_transcript,
        expected_text=expected_text,
        word_analyses=word_analyses,
        overall_accuracy=overall_accuracy,
        total_phoneme_errors=total_errors,
        error_summary=error_counts,
        feedback=feedback,
        raw_mfa_alignment=mfa_alignment
    )


# ============================================================================
# FEEDBACK GENERATION
# ============================================================================

def generate_feedback(
    word_analyses: List[WordAnalysis],
    error_counts: Dict[str, int]
) -> List[str]:
    """
    Generate actionable feedback based on the analysis.
    """
    feedback = []

    # Overall score message
    if word_analyses:
        avg_accuracy = sum(w.accuracy_score for w in word_analyses) / len(word_analyses)

        if avg_accuracy >= 0.95:
            feedback.append("🌟 Excellent pronunciation! Nearly perfect.")
        elif avg_accuracy >= 0.85:
            feedback.append("👍 Good pronunciation with minor errors.")
        elif avg_accuracy >= 0.70:
            feedback.append("📝 Fair pronunciation. Some sounds need practice.")
        else:
            feedback.append("💪 Keep practicing! Several sounds need improvement.")

    # Summarize errors by type
    if error_counts.get('substitution', 0) > 0:
        feedback.append(f"⚠️ {error_counts['substitution']} sound substitution(s) detected.")

    if error_counts.get('deletion', 0) > 0:
        feedback.append(f"⚠️ {error_counts['deletion']} missing sound(s) detected.")

    if error_counts.get('insertion', 0) > 0:
        feedback.append(f"⚠️ {error_counts['insertion']} extra sound(s) detected.")

    # Identify problematic words
    problem_words = [w for w in word_analyses if not w.is_correct]

    if problem_words:
        feedback.append("\n📋 Words that need practice:")

        for wa in problem_words[:5]:  # Top 5 problematic words
            if wa.errors:
                error_details = []
                for err in wa.errors[:2]:  # First 2 errors per word
                    if err.error_type == 'substitution':
                        error_details.append(f"/{err.expected_phoneme}/→/{err.actual_phoneme}/")
                    elif err.error_type == 'deletion':
                        error_details.append(f"missing /{err.expected_phoneme}/")
                    elif err.error_type == 'insertion':
                        error_details.append(f"extra /{err.actual_phoneme}/")

                feedback.append(
                    f"  • '{wa.word}': {wa.expected_ipa} vs {wa.actual_ipa} "
                    f"({', '.join(error_details)})"
                )

    # Add specific tips based on common patterns
    all_errors = [e for w in word_analyses for e in w.errors]

    # Check for TH issues
    th_errors = [e for e in all_errors if 'θ' in (e.expected_phoneme, e.actual_phoneme)
                 or 'ð' in (e.expected_phoneme, e.actual_phoneme)]
    if th_errors:
        feedback.append("\n💡 Tip: Practice the TH sounds (/θ/ as in 'think', /ð/ as in 'this'). "
                       "Place your tongue between your teeth and blow air.")

    # Check for R/L issues
    rl_errors = [e for e in all_errors if ('ɹ' in (e.expected_phoneme, e.actual_phoneme)
                 and 'l' in (e.expected_phoneme, e.actual_phoneme))]
    if rl_errors:
        feedback.append("\n💡 Tip: Practice R vs L. For R, curl your tongue back without touching "
                       "the roof of your mouth. For L, touch the ridge behind your top teeth.")

    return feedback


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_result_for_display(result: PronunciationResult) -> str:
    """Format the result as a readable string."""
    lines = [
        "=" * 60,
        "PRONUNCIATION ANALYSIS RESULT",
        "=" * 60,
        f"Audio: {result.audio_path}",
        f"Transcript: {result.whisper_transcript}",
        f"Overall Accuracy: {result.overall_accuracy:.1%}",
        f"Total Errors: {result.total_phoneme_errors}",
        "",
        "-" * 60,
        "WORD-BY-WORD ANALYSIS",
        "-" * 60,
    ]

    for wa in result.word_analyses:
        status = "✓" if wa.is_correct else "✗"
        lines.append(f"\n{status} {wa.word.upper()}")
        lines.append(f"   Expected: {wa.expected_ipa}")
        lines.append(f"   Actual:   {wa.actual_ipa}")
        lines.append(f"   Accuracy: {wa.accuracy_score:.1%}")

        if wa.errors:
            lines.append("   Errors:")
            for err in wa.errors:
                lines.append(f"     - {err.error_type}: {err.explanation}")

    lines.extend([
        "",
        "-" * 60,
        "FEEDBACK",
        "-" * 60,
    ])
    lines.extend(result.feedback)
    lines.append("=" * 60)

    return "\n".join(lines)


def analyze_and_print(audio_path: str, expected_text: Optional[str] = None):
    """Convenience function to analyze and print results."""
    result = analyze_pronunciation(audio_path, expected_text)
    print(format_result_for_display(result))
    return result


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("UNIFIED PRONUNCIATION ANALYSIS PIPELINE")
    print("=" * 60)
    print()
    print("Pipeline: Audio → Whisper → MFA → Native IPA → Compare → Feedback")
    print()

    # Example usage
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        expected = sys.argv[2] if len(sys.argv) > 2 else None

        print(f"Analyzing: {audio_file}")
        if expected:
            print(f"Expected text: {expected}")
        print()

        result = analyze_pronunciation(audio_file, expected)
        print(format_result_for_display(result))
    else:
        # Demo with L2-ARCTIC sample
        demo_audio = "l2arctic_release_v5/ABA/wav/arctic_a0005.wav"

        if Path(demo_audio).exists():
            print(f"Demo: Analyzing {demo_audio}")
            print()
            result = analyze_pronunciation(demo_audio)
            print(format_result_for_display(result))
        else:
            print("Usage: python pronunciation_pipeline.py <audio_file> [expected_text]")
            print()
            print("Example:")
            print("  python pronunciation_pipeline.py sample.wav")
            print('  python pronunciation_pipeline.py sample.wav "the weather is nice"')
