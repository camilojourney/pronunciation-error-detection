"""
NLP Analysis Utilities for Pronunciation Error Detection
=========================================================

This module contains all the core functions needed for the pronunciation
error detection pipeline, including:
- Text preprocessing and normalization
- Sequence alignment using Levenshtein distance
- Error detection and classification
- Evaluation metrics (WER)
- ASR transcription using Whisper

Author: Camilo Martinez
Course: Natural Language Processing
"""

import re
from dataclasses import dataclass
from typing import List, Tuple
import difflib


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class AlignmentOperation:
    """Represents a single alignment operation between reference and hypothesis."""
    op: str  # 'equal', 'replace', 'delete', 'insert'
    ref: str
    hyp: str


@dataclass
class ErrorAnalysis:
    """Contains detailed error analysis results."""
    total_errors: int
    substitution_count: int
    deletion_count: int
    insertion_count: int
    substitutions: List[Tuple[str, str]]  # [(ref_word, hyp_word), ...]
    deletions: List[str]
    insertions: List[str]


@dataclass
class PipelineResult:
    """Complete result from the pronunciation error detection pipeline."""
    ref: str
    hyp: str
    wer: float
    error_analysis: ErrorAnalysis
    alignment_ops: List[AlignmentOperation]


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.

    Steps:
    1. Convert to lowercase
    2. Remove punctuation
    3. Normalize whitespace

    Args:
        text: Raw text string

    Returns:
        Cleaned text string

    Example:
        >>> clean_text("The Weather is VERY nice today!")
        'the weather is very nice today'
    """
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # Normalize whitespace (collapse multiple spaces)
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into words.

    Args:
        text: Text string to tokenize

    Returns:
        List of word tokens

    Example:
        >>> tokenize("The weather is nice")
        ['the', 'weather', 'is', 'nice']
    """
    # Clean first, then split on whitespace
    cleaned = clean_text(text)
    tokens = cleaned.split()
    return tokens


# ============================================================================
# SEQUENCE ALIGNMENT
# ============================================================================

def align_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> List[AlignmentOperation]:
    """
    Align two token sequences using edit distance (Levenshtein algorithm).

    This function uses the SequenceMatcher from difflib to compute the optimal
    alignment between reference and hypothesis token sequences, identifying:
    - Equal: Matching tokens
    - Replace: Substitution errors
    - Delete: Deletion errors (word omitted)
    - Insert: Insertion errors (extra word added)

    Args:
        ref_tokens: Reference token sequence (expected)
        hyp_tokens: Hypothesis token sequence (actual/spoken)

    Returns:
        List of AlignmentOperation objects representing the alignment

    Example:
        >>> ref = ["the", "weather", "is", "nice"]
        >>> hyp = ["the", "weazer", "is", "nice"]
        >>> ops = align_tokens(ref, hyp)
        >>> ops[1].op
        'replace'
    """
    matcher = difflib.SequenceMatcher(None, ref_tokens, hyp_tokens)
    operations = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Matching tokens
            for i, j in zip(range(i1, i2), range(j1, j2)):
                operations.append(AlignmentOperation(
                    op='equal',
                    ref=ref_tokens[i],
                    hyp=hyp_tokens[j]
                ))
        elif tag == 'replace':
            # Substitution errors
            for i, j in zip(range(i1, i2), range(j1, j2)):
                operations.append(AlignmentOperation(
                    op='replace',
                    ref=ref_tokens[i],
                    hyp=hyp_tokens[j]
                ))
        elif tag == 'delete':
            # Deletion errors (in reference but not in hypothesis)
            for i in range(i1, i2):
                operations.append(AlignmentOperation(
                    op='delete',
                    ref=ref_tokens[i],
                    hyp=''
                ))
        elif tag == 'insert':
            # Insertion errors (in hypothesis but not in reference)
            for j in range(j1, j2):
                operations.append(AlignmentOperation(
                    op='insert',
                    ref='',
                    hyp=hyp_tokens[j]
                ))

    return operations


# ============================================================================
# ERROR DETECTION & CLASSIFICATION
# ============================================================================

def detect_errors(alignment_ops: List[AlignmentOperation]) -> ErrorAnalysis:
    """
    Detect and classify pronunciation errors from alignment operations.

    Classifies errors into three categories:
    - Substitutions: Word was pronounced incorrectly
    - Deletions: Word was omitted
    - Insertions: Extra word was added

    Args:
        alignment_ops: List of alignment operations from align_tokens()

    Returns:
        ErrorAnalysis object with detailed error statistics

    Example:
        >>> ops = align_tokens(["the", "cat"], ["the", "hat"])
        >>> errors = detect_errors(ops)
        >>> errors.substitution_count
        1
    """
    substitutions = []
    deletions = []
    insertions = []

    for op in alignment_ops:
        if op.op == 'replace':
            substitutions.append((op.ref, op.hyp))
        elif op.op == 'delete':
            deletions.append(op.ref)
        elif op.op == 'insert':
            insertions.append(op.hyp)

    total_errors = len(substitutions) + len(deletions) + len(insertions)

    return ErrorAnalysis(
        total_errors=total_errors,
        substitution_count=len(substitutions),
        deletion_count=len(deletions),
        insertion_count=len(insertions),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions
    )


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def wer(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """
    Calculate Word Error Rate (WER).

    WER = (S + D + I) / N
    where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = total number of words in reference

    Args:
        ref_tokens: Reference token sequence
        hyp_tokens: Hypothesis token sequence

    Returns:
        WER as a float between 0.0 and 1.0 (or higher if many insertions)

    Example:
        >>> ref = ["the", "weather", "is", "nice"]
        >>> hyp = ["the", "weazer", "is", "nice"]
        >>> wer(ref, hyp)
        0.25
    """
    if len(ref_tokens) == 0:
        return 0.0 if len(hyp_tokens) == 0 else float('inf')

    alignment_ops = align_tokens(ref_tokens, hyp_tokens)
    errors = detect_errors(alignment_ops)

    wer_score = errors.total_errors / len(ref_tokens)
    return wer_score


def cer(ref_text: str, hyp_text: str) -> float:
    """
    Calculate Character Error Rate (CER).

    Similar to WER but operates at character level.

    Args:
        ref_text: Reference text string
        hyp_text: Hypothesis text string

    Returns:
        CER as a float
    """
    ref_chars = list(ref_text)
    hyp_chars = list(hyp_text)

    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else float('inf')

    alignment_ops = align_tokens(ref_chars, hyp_chars)
    errors = detect_errors(alignment_ops)

    cer_score = errors.total_errors / len(ref_chars)
    return cer_score


# ============================================================================
# ASR INTEGRATION
# ============================================================================

def transcribe_audio(audio_path: str, model_size: str = "base") -> str:
    """
    Transcribe audio file using Whisper ASR.

    Args:
        audio_path: Path to audio file (.wav, .mp3, etc.)
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')

    Returns:
        Transcribed text string

    Example:
        >>> text = transcribe_audio("speaker_01.wav", model_size="base")
        >>> print(text)
        'the weather is very nice today'

    Note:
        Requires faster-whisper or openai-whisper installed.
        Install with: pip install faster-whisper
    """
    try:
        from faster_whisper import WhisperModel

        # Initialize model
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

        # Transcribe
        segments, info = model.transcribe(audio_path, beam_size=5)

        # Concatenate all segments
        transcription = " ".join([segment.text.strip() for segment in segments])

        return transcription.strip()

    except ImportError:
        # Fallback to openai-whisper if faster-whisper not available
        try:
            import whisper

            model = whisper.load_model(model_size)
            result = model.transcribe(audio_path)

            return result["text"].strip()

        except ImportError:
            raise ImportError(
                "Whisper not installed. Install with: "
                "pip install faster-whisper (recommended) or "
                "pip install openai-whisper"
            )


# ============================================================================
# END-TO-END PIPELINE
# ============================================================================

def run_text_pipeline(ref_text: str, hyp_text: str) -> PipelineResult:
    """
    Run the complete pronunciation error detection pipeline (text-based).

    Pipeline steps:
    1. Preprocess and tokenize both texts
    2. Align token sequences
    3. Detect and classify errors
    4. Calculate WER

    Args:
        ref_text: Reference text (expected)
        hyp_text: Hypothesis text (actual/transcribed)

    Returns:
        PipelineResult with complete analysis

    Example:
        >>> result = run_text_pipeline(
        ...     "The weather is nice",
        ...     "The weazer is nice"
        ... )
        >>> print(f"WER: {result.wer:.2%}")
        WER: 25.00%
    """
    # Tokenize
    ref_tokens = tokenize(ref_text)
    hyp_tokens = tokenize(hyp_text)

    # Align
    alignment_ops = align_tokens(ref_tokens, hyp_tokens)

    # Detect errors
    error_analysis = detect_errors(alignment_ops)

    # Calculate WER
    wer_score = wer(ref_tokens, hyp_tokens)

    return PipelineResult(
        ref=ref_text,
        hyp=hyp_text,
        wer=wer_score,
        error_analysis=error_analysis,
        alignment_ops=alignment_ops
    )


def run_audio_pipeline(audio_path: str, ref_text: str, model_size: str = "base") -> PipelineResult:
    """
    Run the complete pronunciation error detection pipeline (audio-based).

    Pipeline steps:
    1. Transcribe audio using Whisper
    2. Run text-based pipeline on transcription

    Args:
        audio_path: Path to audio file
        ref_text: Reference text (expected)
        model_size: Whisper model size

    Returns:
        PipelineResult with complete analysis

    Example:
        >>> result = run_audio_pipeline(
        ...     "audio/speaker_01.wav",
        ...     "The weather is very nice today"
        ... )
        >>> print(f"Transcribed: {result.hyp}")
        >>> print(f"WER: {result.wer:.2%}")
    """
    # Transcribe audio
    hyp_text = transcribe_audio(audio_path, model_size)

    # Run text pipeline
    result = run_text_pipeline(ref_text, hyp_text)

    return result


# ============================================================================
# HELPER FUNCTIONS FOR ANALYSIS
# ============================================================================

def get_speaker_language_mapping():
    """
    Get mapping of L2-ARCTIC speaker IDs to native languages.

    Returns:
        Dictionary mapping speaker_id to native language
    """
    return {
        'ABA': 'Arabic',
        'SKA': 'Arabic',
        'YBAA': 'Arabic',
        'ZHAA': 'Arabic',
        'BWC': 'Mandarin',
        'LXC': 'Mandarin',
        'NCC': 'Mandarin',
        'TXHC': 'Mandarin',
        'ASI': 'Hindi',
        'RRBI': 'Hindi',
        'SVBI': 'Hindi',
        'TNI': 'Hindi',
        'HJK': 'Korean',
        'HKK': 'Korean',
        'YDCK': 'Korean',
        'YKWK': 'Korean',
        'EBVS': 'Spanish',
        'ERMS': 'Spanish',
        'MBMPS': 'Spanish',
        'NJS': 'Spanish',
        'HQTV': 'Vietnamese',
        'PNV': 'Vietnamese',
        'THV': 'Vietnamese',
        'TLV': 'Vietnamese',
    }


def format_error_table(error_analysis: ErrorAnalysis) -> str:
    """
    Format error analysis as a readable table.

    Args:
        error_analysis: ErrorAnalysis object

    Returns:
        Formatted string table
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ERROR ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Total Errors: {error_analysis.total_errors}")
    lines.append("")
    lines.append(f"Substitutions: {error_analysis.substitution_count}")
    for ref, hyp in error_analysis.substitutions:
        lines.append(f"  '{ref}' → '{hyp}'")
    lines.append("")
    lines.append(f"Deletions: {error_analysis.deletion_count}")
    for word in error_analysis.deletions:
        lines.append(f"  '{word}' (deleted)")
    lines.append("")
    lines.append(f"Insertions: {error_analysis.insertion_count}")
    for word in error_analysis.insertions:
        lines.append(f"  '{word}' (inserted)")
    lines.append("=" * 60)

    return "\n".join(lines)


# ============================================================================
# INTELLIGIBILITY ANALYSIS (PHONEME-LEVEL)
# ============================================================================

def get_minimal_pairs_database() -> dict:
    """
    Return a database of minimal pairs - word pairs differing by one phoneme.

    These are critical for intelligibility because confusing them changes meaning.

    Returns:
        Dictionary mapping phoneme substitution patterns to minimal pairs

    Example:
        >>> minimal_pairs = get_minimal_pairs_database()
        >>> minimal_pairs['r_l_confusion']
        [('right', 'light'), ('read', 'lead'), ...]
    """
    return {
        'r_l_confusion': [
            ('right', 'light'), ('read', 'lead'), ('rent', 'lent'),
            ('correct', 'collect'), ('pirate', 'pilot'), ('rock', 'lock'),
            ('red', 'led'), ('road', 'load'), ('race', 'lace')
        ],
        'th_s_confusion': [
            ('think', 'sink'), ('thank', 'sank'), ('thick', 'sick'),
            ('path', 'pass'), ('faith', 'face'), ('myth', 'miss'),
            ('bath', 'bass'), ('tenth', 'tense')
        ],
        'th_z_confusion': [
            ('bathe', 'base'), ('breathe', 'breeze'), ('clothe', 'close')
        ],
        'th_t_confusion': [
            ('think', 'tink'), ('thick', 'tick'), ('thing', 'ting'),
            ('thought', 'taught'), ('thaw', 'taw')
        ],
        'th_f_confusion': [
            ('thin', 'fin'), ('thick', 'fick'), ('thought', 'fought'),
            ('three', 'free'), ('thirst', 'first')
        ],
        'sh_ch_confusion': [
            ('ship', 'chip'), ('shop', 'chop'), ('wash', 'watch'),
            ('ashes', 'hatches'), ('sheep', 'cheap'), ('shore', 'chore')
        ],
        'v_b_confusion': [
            ('van', 'ban'), ('vote', 'boat'), ('vest', 'best'),
            ('curve', 'curb'), ('very', 'berry'), ('vine', 'bine')
        ],
        'v_w_confusion': [
            ('vine', 'wine'), ('veal', 'wheel'), ('verse', 'worse'),
            ('vent', 'went'), ('vile', 'while')
        ],
        'v_f_confusion': [
            ('van', 'fan'), ('vat', 'fat'), ('vine', 'fine'),
            ('veil', 'fail'), ('very', 'ferry')
        ],
        'f_p_confusion': [
            ('fan', 'pan'), ('face', 'pace'), ('feel', 'peel'),
            ('fast', 'past'), ('foot', 'put')
        ],
        'p_b_confusion': [
            ('cap', 'cab'), ('rip', 'rib'), ('rope', 'robe'),
            ('pear', 'bear'), ('pack', 'back')
        ],
        'z_s_confusion': [
            ('zoo', 'sue'), ('zip', 'sip'), ('zone', 'sewn')
        ],
        'j_y_confusion': [  # /dʒ/ vs /j/
            ('jet', 'yet'), ('jail', 'yale'), ('jeer', 'year')
        ],
        'final_consonant_deletion': [
            ('fast', 'fas'), ('test', 'tes'), ('best', 'bes'),
            ('last', 'las'), ('must', 'mus')
        ]
    }


@dataclass
class IntelligibilityImpact:
    """Classification of error's impact on intelligibility."""
    level: str  # 'HIGH', 'MEDIUM', 'LOW'
    minimal_pair: bool
    phoneme_pattern: str  # e.g., 'r_l_confusion'
    explanation: str


def assess_intelligibility_impact(ref_word: str, hyp_word: str) -> IntelligibilityImpact:
    """
    Assess the intelligibility impact of a word substitution.

    Determines if the substitution creates a minimal pair (different meaning)
    or is likely just an accent feature.

    Args:
        ref_word: Reference word (expected)
        hyp_word: Hypothesis word (produced)

    Returns:
        IntelligibilityImpact classification

    Example:
        >>> impact = assess_intelligibility_impact('ship', 'chip')
        >>> impact.level
        'HIGH'
        >>> impact.minimal_pair
        True
    """
    minimal_pairs_db = get_minimal_pairs_database()

    # Check if this substitution is a known minimal pair
    for pattern, pairs in minimal_pairs_db.items():
        for word1, word2 in pairs:
            if (ref_word.lower() == word1 and hyp_word.lower() == word2) or \
               (ref_word.lower() == word2 and hyp_word.lower() == word1):
                return IntelligibilityImpact(
                    level='HIGH',
                    minimal_pair=True,
                    phoneme_pattern=pattern,
                    explanation=f"Minimal pair: '{ref_word}' ↔ '{hyp_word}' (different meanings)"
                )

    # Check for incomplete words (deletion patterns)
    if len(hyp_word) < len(ref_word) and ref_word.startswith(hyp_word):
        return IntelligibilityImpact(
            level='MEDIUM',
            minimal_pair=False,
            phoneme_pattern='final_consonant_deletion',
            explanation=f"Incomplete word: '{ref_word}' → '{hyp_word}' (missing ending)"
        )

    # Check for non-standard additions (e.g., 'speak' → 'espeak')
    if len(hyp_word) > len(ref_word) and hyp_word.endswith(ref_word):
        return IntelligibilityImpact(
            level='MEDIUM',
            minimal_pair=False,
            phoneme_pattern='initial_insertion',
            explanation=f"Non-standard prefix: '{ref_word}' → '{hyp_word}'"
        )

    # Otherwise, likely an accent feature (low impact)
    return IntelligibilityImpact(
        level='LOW',
        minimal_pair=False,
        phoneme_pattern='accent_feature',
        explanation=f"Likely accent variation: '{ref_word}' → '{hyp_word}' (understandable)"
    )


@dataclass
class IntelligibilityAnalysis:
    """Enhanced error analysis with intelligibility classification."""
    high_impact_errors: List[Tuple[str, str, str]]  # [(ref, hyp, explanation), ...]
    medium_impact_errors: List[Tuple[str, str, str]]
    low_impact_errors: List[Tuple[str, str, str]]
    deletions: List[str]
    insertions: List[str]
    total_critical_errors: int  # HIGH + MEDIUM only


def analyze_intelligibility(error_analysis: ErrorAnalysis) -> IntelligibilityAnalysis:
    """
    Analyze errors from an intelligibility perspective.

    Classifies substitutions by their impact on communication:
    - HIGH: Minimal pairs that change meaning
    - MEDIUM: Noticeable but usually clear from context
    - LOW: Accent features that don't impede understanding

    Args:
        error_analysis: ErrorAnalysis from detect_errors()

    Returns:
        IntelligibilityAnalysis with impact classifications

    Example:
        >>> result = run_text_pipeline("ship on water", "chip on water")
        >>> intel = analyze_intelligibility(result.error_analysis)
        >>> len(intel.high_impact_errors)
        1
    """
    high_impact = []
    medium_impact = []
    low_impact = []

    for ref_word, hyp_word in error_analysis.substitutions:
        impact = assess_intelligibility_impact(ref_word, hyp_word)

        entry = (ref_word, hyp_word, impact.explanation)

        if impact.level == 'HIGH':
            high_impact.append(entry)
        elif impact.level == 'MEDIUM':
            medium_impact.append(entry)
        else:
            low_impact.append(entry)

    # Deletions and insertions are generally MEDIUM impact
    # (they're noticeable but context often helps)
    total_critical = len(high_impact) + len(medium_impact) + \
                    error_analysis.deletion_count + error_analysis.insertion_count

    return IntelligibilityAnalysis(
        high_impact_errors=high_impact,
        medium_impact_errors=medium_impact,
        low_impact_errors=low_impact,
        deletions=error_analysis.deletions,
        insertions=error_analysis.insertions,
        total_critical_errors=total_critical
    )


def format_intelligibility_feedback(intel_analysis: IntelligibilityAnalysis) -> str:
    """
    Format intelligibility analysis as learner-friendly feedback.

    Args:
        intel_analysis: IntelligibilityAnalysis object

    Returns:
        Formatted feedback string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("INTELLIGIBILITY FEEDBACK")
    lines.append("=" * 70)

    if intel_analysis.high_impact_errors:
        lines.append("\n🔴 CRITICAL (May cause confusion):")
        for ref, hyp, explanation in intel_analysis.high_impact_errors:
            lines.append(f"   '{ref}' → '{hyp}'")
            lines.append(f"      {explanation}")

    if intel_analysis.medium_impact_errors:
        lines.append("\n🟡 NOTICEABLE (Reduces clarity):")
        for ref, hyp, explanation in intel_analysis.medium_impact_errors:
            lines.append(f"   '{ref}' → '{hyp}'")
            lines.append(f"      {explanation}")

    if intel_analysis.deletions:
        lines.append("\n🟡 DELETIONS (Words omitted):")
        for word in intel_analysis.deletions:
            lines.append(f"   '{word}' was skipped")

    if intel_analysis.insertions:
        lines.append("\n🟡 INSERTIONS (Extra words):")
        for word in intel_analysis.insertions:
            lines.append(f"   '{word}' was added")

    if intel_analysis.low_impact_errors:
        lines.append("\n🟢 ACCENT FEATURES (Perfectly fine!):")
        for ref, hyp, explanation in intel_analysis.low_impact_errors:
            lines.append(f"   '{ref}' → '{hyp}'")
            lines.append(f"      {explanation}")

    if not intel_analysis.high_impact_errors and \
       not intel_analysis.medium_impact_errors and \
       not intel_analysis.deletions and \
       not intel_analysis.insertions:
        lines.append("\n✅ Excellent! All variations are natural accent features.")

    lines.append("\n" + "=" * 70)
    lines.append(f"Total critical errors (need attention): {intel_analysis.total_critical_errors}")
    lines.append(f"Accent features (no action needed): {len(intel_analysis.low_impact_errors)}")
    lines.append("=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("NLP Analysis Utilities - Example Usage")
    print("=" * 60)

    # Text-based example with intelligibility analysis
    ref = "The ship is on the right side"
    hyp = "The chip is on the light side"

    result = run_text_pipeline(ref, hyp)

    print(f"\nReference:  {result.ref}")
    print(f"Hypothesis: {result.hyp}")
    print(f"\nWER: {result.wer:.2%}")
    print(f"\n{format_error_table(result.error_analysis)}")

    # Intelligibility analysis
    intel = analyze_intelligibility(result.error_analysis)
    print(f"\n{format_intelligibility_feedback(intel)}")
