"""
Unified Phoneme Sources Module
================================

This module provides a single, clean API for extracting phonemes from all 5 sources:

1. **Dictionary** - Expected IPA from CMU Dictionary / G2P
2. **MFA** - Actual phonemes from audio via Montreal Forced Aligner
3. **Wav2Vec2** - Direct audio → IPA recognition
4. **Whisper + MFA** - Words from Whisper ASR, phonemes from MFA
5. **Whisper + Wav2Vec2** - Words from Whisper ASR, phonemes from Wav2Vec2

This consolidates code previously scattered across:
- forced_alignment.py (MFA logic)
- wav2vec_phoneme.py (Wav2Vec2 logic)
- analysis_utils.py (Dictionary logic)

Key principle: Separation of concerns - this module handles ONLY phoneme extraction.

Author: Camilo Martinez
Course: Natural Language Processing
"""

import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# ============================================================================
# ARPABET → IPA MAPPING (Single Source of Truth)
# ============================================================================

ARPABET_TO_IPA = {
    # Consonants - Stops
    'P': 'p', 'B': 'b', 'T': 't', 'D': 'd', 'K': 'k', 'G': 'ɡ',

    # Consonants - Fricatives
    'F': 'f', 'V': 'v', 'TH': 'θ', 'DH': 'ð', 'S': 's', 'Z': 'z',
    'SH': 'ʃ', 'ZH': 'ʒ', 'HH': 'h',

    # Consonants - Affricates
    'CH': 'tʃ', 'JH': 'dʒ',

    # Consonants - Nasals
    'M': 'm', 'N': 'n', 'NG': 'ŋ',

    # Consonants - Liquids
    'L': 'l', 'R': 'ɹ',

    # Consonants - Glides
    'W': 'w', 'Y': 'j',

    # Vowels - Monophthongs (Tense)
    'IY': 'i', 'UW': 'u',

    # Vowels - Monophthongs (Lax)
    'IH': 'ɪ', 'UH': 'ʊ', 'EH': 'ɛ', 'AH': 'ʌ',
    'AE': 'æ', 'AA': 'ɑ', 'AO': 'ɔ',

    # Vowels - R-colored (using dictionary-style broad IPA)
    'ER': 'ər', 'AXR': 'ər',

    # Vowels - Reduced
    'AX': 'ə', 'IX': 'ɨ', 'AXH': 'ə̥',

    # Diphthongs
    'EY': 'eɪ', 'AY': 'aɪ', 'OW': 'oʊ', 'AW': 'aʊ', 'OY': 'ɔɪ',
}


def arpabet_to_ipa(arpabet: str) -> str:
    """
    Convert ARPABET phoneme to IPA.

    Removes stress markers (0, 1, 2) automatically.
    """
    # Remove stress markers
    clean = re.sub(r'[012]$', '', arpabet.upper())
    return ARPABET_TO_IPA.get(clean, clean.lower())


def format_ipa(phonemes: List[str]) -> str:
    """
    Format phonemes as IPA string with slashes.

    Args:
        phonemes: List of phonemes (ARPABET or IPA)

    Returns:
        Formatted IPA string like "/ʃ ɪ p/"
    """
    if not phonemes:
        return '/?/'

    # Convert to IPA if needed and join with spaces
    ipa_phonemes = [arpabet_to_ipa(p) if p.isupper() else p for p in phonemes]
    return f"/{' '.join(ipa_phonemes)}/"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class PhonemeWord:
    """Represents a word with its phoneme transcription."""
    word: str
    phonemes: List[str]  # IPA phonemes
    ipa: str  # Formatted IPA string (e.g., "/ʃ ɪ p/")


@dataclass
class PhonemeSource:
    """Result from a single phoneme extraction source."""
    name: str
    description: str
    success: bool
    words: Optional[List[PhonemeWord]] = None
    continuous_ipa: Optional[str] = None  # For Wav2Vec2 continuous output
    error: Optional[str] = None
    metadata: Optional[Dict] = None


# ============================================================================
# SOURCE 1: DICTIONARY (CMU Dictionary / G2P)
# ============================================================================

# Lazy initialization
_g2p_model = None
_cmudict = None


def _get_g2p_model():
    """Lazy load G2P model."""
    global _g2p_model
    if _g2p_model is None:
        try:
            from g2p_en import G2p
            _g2p_model = G2p()
        except ImportError:
            pass
    return _g2p_model


def _get_cmudict():
    """Lazy load CMU Dictionary."""
    global _cmudict
    if _cmudict is None:
        try:
            import nltk
            from nltk.corpus import cmudict
            try:
                _cmudict = cmudict.dict()
            except LookupError:
                nltk.download('cmudict', quiet=True)
                _cmudict = cmudict.dict()
        except ImportError:
            pass
    return _cmudict


def get_dictionary_phonemes(text: str) -> PhonemeSource:
    """
    Get expected phonemes from CMU Dictionary / G2P.

    This is the "ground truth" - what the pronunciation SHOULD be.

    Args:
        text: Input text (words)

    Returns:
        PhonemeSource with dictionary phonemes
    """
    words = text.lower().split()
    phoneme_words = []

    for word in words:
        # Clean word
        clean_word = re.sub(r'[^a-z\']', '', word)
        if not clean_word:
            continue

        # Try CMU Dictionary first
        cmu = _get_cmudict()
        phonemes_arpabet = []

        if cmu and clean_word in cmu:
            phonemes_arpabet = cmu[clean_word][0]
            phonemes_arpabet = [p.rstrip('012') for p in phonemes_arpabet]
        else:
            # Fall back to G2P
            g2p = _get_g2p_model()
            if g2p:
                try:
                    phonemes_arpabet = g2p(clean_word)
                    phonemes_arpabet = [p.rstrip('012') for p in phonemes_arpabet]
                except:
                    phonemes_arpabet = []

        if phonemes_arpabet:
            # Convert ARPABET to IPA
            phonemes_ipa = [arpabet_to_ipa(p) for p in phonemes_arpabet]
            phoneme_words.append(PhonemeWord(
                word=clean_word,
                phonemes=phonemes_ipa,
                ipa=format_ipa(phonemes_arpabet)
            ))

    return PhonemeSource(
        name='Dictionary (Expected)',
        description='IPA from CMU dictionary / G2P',
        success=True,
        words=phoneme_words
    )


# ============================================================================
# SOURCE 2: MFA (Montreal Forced Aligner)
# ============================================================================

# MFA binary path
MFA_BIN = "/opt/anaconda3/envs/mfa/bin/mfa"


@dataclass
class PhonemeInterval:
    """Phoneme with timing information from MFA."""
    phoneme: str
    start_time: float
    end_time: float
    duration: float


def _find_l2arctic_textgrid(audio_path: str) -> Optional[str]:
    """
    Find pre-existing TextGrid file for L2-ARCTIC dataset.

    L2-ARCTIC structure:
    - Audio: l2arctic_release_v5/SPEAKER/wav/filename.wav
    - TextGrid: l2arctic_release_v5/SPEAKER/textgrid/filename.TextGrid

    Args:
        audio_path: Path to audio file

    Returns:
        Path to TextGrid file if found, None otherwise
    """
    path_obj = Path(audio_path)

    # Check if this is L2-ARCTIC structure
    if 'l2arctic' in str(path_obj).lower() and 'wav' in path_obj.parts:
        # Replace 'wav' with 'textgrid' in path
        textgrid_path = Path(str(path_obj).replace('/wav/', '/textgrid/').replace('.wav', '.TextGrid'))

        if textgrid_path.exists():
            return str(textgrid_path)

    return None


def _check_mfa_available() -> bool:
    """Check if MFA is installed."""
    try:
        result = subprocess.run(
            [MFA_BIN, 'version'],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except:
        return False


def _align_audio_mfa(audio_path: str, transcript: str) -> Optional[str]:
    """
    Run MFA alignment on audio file.

    Returns:
        Path to output TextGrid file, or None if failed
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        return None

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # MFA expects: corpus_dir/speaker/audio.wav + audio.txt
        speaker_dir = temp_path / "speaker"
        speaker_dir.mkdir()

        # Link audio file
        audio_dest = speaker_dir / audio_path.name
        audio_dest.symlink_to(audio_path.absolute())

        # Create transcript file
        transcript_path = speaker_dir / (audio_path.stem + ".txt")
        transcript_path.write_text(transcript)

        # Output directory
        output_path = temp_path / "aligned"
        output_path.mkdir()

        try:
            result = subprocess.run(
                [
                    'conda', 'run', '-n', 'mfa',
                    'mfa', 'align',
                    str(temp_path),
                    'english_us_arpa',
                    'english_us_arpa',
                    str(output_path),
                    '--clean',
                    '--single_speaker',
                    '-j', '1'
                ],
                capture_output=True,
                timeout=300
            )

            if result.returncode != 0:
                return None

            textgrid_path = output_path / "speaker" / (audio_path.stem + ".TextGrid")
            return str(textgrid_path) if textgrid_path.exists() else None

        except:
            return None


def _parse_textgrid(textgrid_path: str) -> List[PhonemeWord]:
    """
    Parse MFA TextGrid to extract word-phoneme alignments.

    Returns:
        List of PhonemeWord objects
    """
    try:
        from praatio import textgrid
    except ImportError:
        return []

    try:
        tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
    except:
        return []

    # Extract word and phone tiers
    word_tier = None
    phone_tier = None

    for tier_name in tg.tierNames:
        tier_lower = tier_name.lower()
        if 'word' in tier_lower:
            word_tier = tg.getTier(tier_name)
        elif 'phone' in tier_lower:
            phone_tier = tg.getTier(tier_name)

    if not phone_tier:
        return []

    # Extract phoneme intervals
    phoneme_intervals = []
    for interval in phone_tier.entries:
        if interval.label and interval.label not in ['', 'sil', 'sp', 'spn']:
            phoneme_intervals.append(PhonemeInterval(
                phoneme=interval.label,
                start_time=interval.start,
                end_time=interval.end,
                duration=interval.end - interval.start
            ))

    # Group phonemes by words
    phoneme_words = []
    if word_tier:
        for word_interval in word_tier.entries:
            if word_interval.label and word_interval.label not in ['', 'sil', 'sp', 'spn']:
                # Find phonemes within word's time range
                word_phonemes = [
                    p for p in phoneme_intervals
                    if p.start_time >= word_interval.start - 0.01 and
                       p.end_time <= word_interval.end + 0.01
                ]

                # Convert to IPA
                ipa_phonemes = [arpabet_to_ipa(p.phoneme) for p in word_phonemes]

                phoneme_words.append(PhonemeWord(
                    word=word_interval.label,
                    phonemes=ipa_phonemes,
                    ipa='/' + ' '.join(ipa_phonemes) + '/'
                ))

    return phoneme_words


def get_mfa_phonemes(audio_path: str, transcript: str) -> PhonemeSource:
    """
    Get actual phonemes from audio using Montreal Forced Aligner.

    For L2-ARCTIC dataset, uses pre-computed TextGrid files.
    Otherwise, attempts to run MFA alignment.

    Args:
        audio_path: Path to audio file (WAV)
        transcript: Expected transcript

    Returns:
        PhonemeSource with MFA phonemes
    """
    # Check for pre-existing TextGrid (L2-ARCTIC dataset)
    textgrid_path = _find_l2arctic_textgrid(audio_path)

    if not textgrid_path:
        # Fall back to running MFA if available
        if not _check_mfa_available():
            return PhonemeSource(
                name='MFA (Forced Alignment)',
                description='Actual phonemes from audio via Montreal Forced Aligner',
                success=False,
                error='MFA not installed and no pre-computed TextGrid found'
            )

        # Run alignment
        textgrid_path = _align_audio_mfa(audio_path, transcript)

        if not textgrid_path:
            return PhonemeSource(
                name='MFA (Forced Alignment)',
                description='Actual phonemes from audio via Montreal Forced Aligner',
                success=False,
                error='MFA alignment failed'
            )

    # Parse TextGrid
    phoneme_words = _parse_textgrid(textgrid_path)

    return PhonemeSource(
        name='MFA (Forced Alignment)',
        description='Actual phonemes from audio via Montreal Forced Aligner',
        success=True,
        words=phoneme_words
    )


# ============================================================================
# SOURCE 3: WAV2VEC2 (Direct Audio Recognition)
# ============================================================================

# Global model cache
_wav2vec_model = None
_wav2vec_tokenizer = None
_wav2vec_extractor = None


def _load_wav2vec_model():
    """Lazy load Wav2Vec2 model."""
    global _wav2vec_model, _wav2vec_tokenizer, _wav2vec_extractor

    if _wav2vec_model is None:
        try:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

            model_name = "facebook/wav2vec2-lv-60-espeak-cv-ft"
            _wav2vec_tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_name)
            _wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            _wav2vec_model = Wav2Vec2ForCTC.from_pretrained(model_name)
            _wav2vec_model.eval()
        except ImportError:
            pass

    return _wav2vec_tokenizer, _wav2vec_extractor, _wav2vec_model


def _load_audio(audio_path: str) -> Optional[Tuple]:
    """Load audio file and return waveform at 16kHz."""
    import numpy as np

    # Try soundfile first
    try:
        import soundfile as sf
        waveform, sample_rate = sf.read(audio_path)
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)
        waveform = waveform.astype(np.float32)
    except ImportError:
        # Try torchaudio
        try:
            import torch
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0)
            waveform = waveform.squeeze().numpy()
        except ImportError:
            return None

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        duration = len(waveform) / sample_rate
        new_length = int(duration * 16000)
        indices = np.linspace(0, len(waveform) - 1, new_length).astype(int)
        waveform = waveform[indices]

    return waveform, 16000


def _normalize_to_dictionary_ipa(phonemes: List[str]) -> List[str]:
    """
    Normalize narrow IPA phonemes to dictionary-style broad IPA for display consistency.

    Converts:
    - /ɚ/ (narrow IPA r-colored schwa) → /ə/ + /r/ (dictionary style)
    - /ɝ/ (narrow IPA stressed r-colored vowel) → /ə/ + /r/

    This is applied to Wav2Vec2 output which natively produces narrow IPA,
    to match the dictionary-style notation used by other sources.

    Args:
        phonemes: List of phoneme strings

    Returns:
        List of normalized phonemes
    """
    normalized = []
    for phoneme in phonemes:
        if phoneme == 'ɚ' or phoneme == 'ɝ':
            # Split r-colored vowels into schwa + r
            normalized.extend(['ə', 'r'])
        else:
            normalized.append(phoneme)
    return normalized


def _split_ipa_string(ipa_string: str) -> List[str]:
    """
    Split continuous IPA string into individual phonemes.

    Handles multi-character phonemes like 'tʃ', 'dʒ', 'eɪ', etc.
    """
    # Multi-character phonemes (order matters - longer first)
    multi_char = [
        'eɪ', 'aɪ', 'oʊ', 'aʊ', 'ɔɪ', 'iː', 'uː', 'ɑː', 'ɔː', 'ɜː', 'ɛː',
        'tʃ', 'dʒ',
        'ɚ', 'ɝ', 'ɑɹ', 'ɔɹ', 'ɛɹ', 'ɪɹ', 'ʊɹ',
    ]

    phonemes = []
    i = 0

    while i < len(ipa_string):
        matched = False

        # Check multi-character first
        for mc in multi_char:
            if ipa_string[i:].startswith(mc):
                phonemes.append(mc)
                i += len(mc)
                matched = True
                break

        if not matched:
            # Check for length marker
            if i + 1 < len(ipa_string) and ipa_string[i + 1] == 'ː':
                phonemes.append(ipa_string[i:i+2])
                i += 2
            else:
                # Single character
                char = ipa_string[i]
                if char not in ['ː', '̩', '̃', '̥', 'ˈ', 'ˌ', '|']:
                    phonemes.append(char)
                i += 1

    return phonemes


def get_wav2vec_phonemes(audio_path: str) -> PhonemeSource:
    """
    Get phonemes directly from audio using Wav2Vec2.

    This model listens to the audio and outputs IPA phonemes directly,
    without needing text or word boundaries.

    Args:
        audio_path: Path to audio file (WAV)

    Returns:
        PhonemeSource with continuous Wav2Vec2 phonemes
    """
    tokenizer, extractor, model = _load_wav2vec_model()

    if model is None:
        return PhonemeSource(
            name='Wav2Vec2 (Direct)',
            description='Direct audio → IPA (no word boundaries)',
            success=False,
            error='Wav2Vec2 model not available'
        )

    # Load audio
    audio_data = _load_audio(audio_path)
    if audio_data is None:
        return PhonemeSource(
            name='Wav2Vec2 (Direct)',
            description='Direct audio → IPA (no word boundaries)',
            success=False,
            error='Could not load audio'
        )

    waveform, sample_rate = audio_data

    # Prepare input
    import torch
    input_values = extractor(
        waveform,
        sampling_rate=sample_rate,
        return_tensors="pt"
    ).input_values

    # Run inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode
    predicted_ids = torch.argmax(logits, dim=-1)
    phoneme_string = tokenizer.batch_decode(predicted_ids)[0]

    # Clean and split
    cleaned = phoneme_string.strip().replace('ˈ', '').replace('ˌ', '').replace('|', '')
    phonemes = _split_ipa_string(cleaned)

    # Normalize to dictionary-style IPA (ɚ → ə r)
    phonemes = _normalize_to_dictionary_ipa(phonemes)

    return PhonemeSource(
        name='Wav2Vec2 (Direct)',
        description='Direct audio → IPA (no word boundaries)',
        success=True,
        continuous_ipa='/' + ''.join(phonemes) + '/',
        metadata={'phonemes': phonemes}
    )


# ============================================================================
# SOURCE 4 & 5: HYBRID METHODS (Whisper + MFA / Wav2Vec2)
# ============================================================================

def _get_whisper_words(audio_path: str, model_size: str = "base") -> List[str]:
    """
    Get word transcription from Whisper ASR.

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium', 'large-v3')

    Returns:
        List of words from transcription
    """
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_path, language="en")
        text = ' '.join(segment.text for segment in segments)

        # Clean text
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text.split()
    except Exception as e:
        print(f"Whisper {model_size} error: {e}")
        return []


def get_whisper_mfa_phonemes(audio_path: str, model_size: str = "base") -> PhonemeSource:
    """
    Get phonemes using Whisper for words, MFA for phonemes.

    Strategy:
    1. Use Whisper ASR to get what words were spoken
    2. Use MFA to align those words with audio and get actual phonemes

    Args:
        audio_path: Path to audio file
        model_size: Whisper model size ('base', 'small', 'medium', 'large-v3')

    Returns:
        PhonemeSource with Whisper+MFA phonemes
    """
    # Get words from Whisper
    words = _get_whisper_words(audio_path, model_size=model_size)

    if not words:
        return PhonemeSource(
            name='Whisper + MFA',
            description='Words from Whisper → Phonemes from MFA',
            success=False,
            error='Whisper transcription failed'
        )

    whisper_text = ' '.join(words)

    # Get phonemes from MFA using Whisper text
    mfa_result = get_mfa_phonemes(audio_path, whisper_text)

    return PhonemeSource(
        name='Whisper + MFA',
        description='Words from Whisper → Phonemes from MFA',
        success=mfa_result.success,
        words=mfa_result.words,
        error=mfa_result.error,
        metadata={'whisper_text': whisper_text}
    )


def get_whisper_large_mfa_phonemes(audio_path: str) -> PhonemeSource:
    """
    Get phonemes using Whisper Large-v3 for words, MFA for phonemes.

    Same as get_whisper_mfa_phonemes but uses the large-v3 model
    for potentially better transcription accuracy.

    Args:
        audio_path: Path to audio file

    Returns:
        PhonemeSource with Whisper Large-v3 + MFA phonemes
    """
    # Get words from Whisper Large-v3
    words = _get_whisper_words(audio_path, model_size="large-v3")

    if not words:
        return PhonemeSource(
            name='Whisper Large-v3 + MFA',
            description='Words from Whisper Large-v3 → Phonemes from MFA',
            success=False,
            error='Whisper Large-v3 transcription failed'
        )

    whisper_text = ' '.join(words)

    # Get phonemes from MFA using Whisper text
    mfa_result = get_mfa_phonemes(audio_path, whisper_text)

    return PhonemeSource(
        name='Whisper Large-v3 + MFA',
        description='Words from Whisper Large-v3 → Phonemes from MFA',
        success=mfa_result.success,
        words=mfa_result.words,
        error=mfa_result.error,
        metadata={'whisper_text': whisper_text, 'model': 'large-v3'}
    )


def get_whisper_wav2vec_phonemes(audio_path: str) -> PhonemeSource:
    """
    Get phonemes using Whisper Base for words, Wav2Vec2 for phonemes.

    Strategy:
    1. Use Whisper ASR to get what words were spoken
    2. Get expected phoneme count per word from dictionary
    3. Use Wav2Vec2 to get continuous phoneme stream
    4. Align Wav2Vec2 phonemes with words using expected counts

    Args:
        audio_path: Path to audio file

    Returns:
        PhonemeSource with Whisper+Wav2Vec2 phonemes
    """
    # Get words from Whisper Base
    words = _get_whisper_words(audio_path, model_size="base")

    if not words:
        return PhonemeSource(
            name='Whisper + Wav2Vec2',
            description='Words from Whisper → Phonemes aligned from Wav2Vec2',
            success=False,
            error='Whisper transcription failed'
        )

    # Get expected phoneme counts
    expected_counts = []
    for word in words:
        cmu = _get_cmudict()
        if cmu and word in cmu:
            expected_counts.append(len(cmu[word][0]))
        else:
            # Estimate based on word length
            expected_counts.append(max(2, len(word) // 2))

    # Get continuous phonemes from Wav2Vec2
    wav2vec_result = get_wav2vec_phonemes(audio_path)

    if not wav2vec_result.success:
        return PhonemeSource(
            name='Whisper + Wav2Vec2',
            description='Words from Whisper → Phonemes aligned from Wav2Vec2',
            success=False,
            error='Wav2Vec2 recognition failed'
        )

    phonemes = wav2vec_result.metadata['phonemes']

    # Align phonemes with words using expected counts
    total_expected = sum(expected_counts)
    total_actual = len(phonemes)

    if total_expected == 0:
        return PhonemeSource(
            name='Whisper + Wav2Vec2',
            description='Words from Whisper → Phonemes aligned from Wav2Vec2',
            success=False,
            error='Could not estimate phoneme counts'
        )

    scale = total_actual / total_expected

    phoneme_words = []
    current_pos = 0

    for word, expected_count in zip(words, expected_counts):
        # Allocate phonemes proportionally
        num_phonemes = max(1, int(expected_count * scale))
        num_phonemes = min(num_phonemes, len(phonemes) - current_pos)

        word_phonemes = phonemes[current_pos:current_pos + num_phonemes]

        phoneme_words.append(PhonemeWord(
            word=word,
            phonemes=word_phonemes,
            ipa='/' + ' '.join(word_phonemes) + '/'
        ))

        current_pos += num_phonemes

    # Add remaining phonemes to last word
    if current_pos < len(phonemes) and phoneme_words:
        phoneme_words[-1].phonemes.extend(phonemes[current_pos:])
        phoneme_words[-1].ipa = '/' + ' '.join(phoneme_words[-1].phonemes) + '/'

    return PhonemeSource(
        name='Whisper + Wav2Vec2',
        description='Words from Whisper → Phonemes aligned from Wav2Vec2',
        success=True,
        words=phoneme_words,
        metadata={'whisper_text': ' '.join(words)}
    )


def get_whisper_large_wav2vec_phonemes(audio_path: str) -> PhonemeSource:
    """
    Get phonemes using Whisper Large-v3 for words, Wav2Vec2 for phonemes.

    Same as get_whisper_wav2vec_phonemes but uses the large-v3 model
    for potentially better transcription accuracy.

    Strategy:
    1. Use Whisper Large-v3 ASR to get what words were spoken
    2. Get expected phoneme count per word from dictionary
    3. Use Wav2Vec2 to get continuous phoneme stream
    4. Align Wav2Vec2 phonemes with words using expected counts

    Args:
        audio_path: Path to audio file

    Returns:
        PhonemeSource with Whisper Large-v3 + Wav2Vec2 phonemes
    """
    # Get words from Whisper Large-v3
    words = _get_whisper_words(audio_path, model_size="large-v3")

    if not words:
        return PhonemeSource(
            name='Whisper Large-v3 + Wav2Vec2',
            description='Words from Whisper Large-v3 → Phonemes aligned from Wav2Vec2',
            success=False,
            error='Whisper Large-v3 transcription failed'
        )

    # Get expected phoneme counts
    expected_counts = []
    for word in words:
        cmu = _get_cmudict()
        if cmu and word in cmu:
            expected_counts.append(len(cmu[word][0]))
        else:
            # Estimate based on word length
            expected_counts.append(max(2, len(word) // 2))

    # Get continuous phonemes from Wav2Vec2
    wav2vec_result = get_wav2vec_phonemes(audio_path)

    if not wav2vec_result.success:
        return PhonemeSource(
            name='Whisper Large-v3 + Wav2Vec2',
            description='Words from Whisper Large-v3 → Phonemes aligned from Wav2Vec2',
            success=False,
            error='Wav2Vec2 recognition failed'
        )

    phonemes = wav2vec_result.metadata['phonemes']

    # Align phonemes with words using expected counts
    total_expected = sum(expected_counts)
    total_actual = len(phonemes)

    if total_expected == 0:
        return PhonemeSource(
            name='Whisper Large-v3 + Wav2Vec2',
            description='Words from Whisper Large-v3 → Phonemes aligned from Wav2Vec2',
            success=False,
            error='Could not estimate phoneme counts'
        )

    scale = total_actual / total_expected

    phoneme_words = []
    current_pos = 0

    for word, expected_count in zip(words, expected_counts):
        # Allocate phonemes proportionally
        num_phonemes = max(1, int(expected_count * scale))
        num_phonemes = min(num_phonemes, len(phonemes) - current_pos)

        word_phonemes = phonemes[current_pos:current_pos + num_phonemes]

        phoneme_words.append(PhonemeWord(
            word=word,
            phonemes=word_phonemes,
            ipa='/' + ' '.join(word_phonemes) + '/'
        ))

        current_pos += num_phonemes

    # Add remaining phonemes to last word
    if current_pos < len(phonemes) and phoneme_words:
        phoneme_words[-1].phonemes.extend(phonemes[current_pos:])
        phoneme_words[-1].ipa = '/' + ' '.join(phoneme_words[-1].phonemes) + '/'

    return PhonemeSource(
        name='Whisper Large-v3 + Wav2Vec2',
        description='Words from Whisper Large-v3 → Phonemes aligned from Wav2Vec2',
        success=True,
        words=phoneme_words,
        metadata={'whisper_text': ' '.join(words), 'model': 'large-v3'}
    )


# ============================================================================
# MAIN API: GET ALL 5 SOURCES
# ============================================================================

def get_all_phoneme_sources(
    audio_path: str,
    expected_text: Optional[str] = None
) -> Dict[str, PhonemeSource]:
    """
    Get phoneme analysis from 3 sources for comparison.

    This is the main API function - it returns a focused comparison of
    phoneme extraction from 3 different methods.

    Sources:
    1. Dictionary (CMU/G2P) - Expected pronunciation from reference text
    2. Whisper Large-v3 + MFA - Words from Whisper Large-v3, phonemes from MFA
    3. Whisper Large-v3 + Wav2Vec2 - Words from Whisper Large-v3, phonemes from Wav2Vec2

    Args:
        audio_path: Path to audio file (WAV)
        expected_text: Optional expected text (if None, uses Whisper)

    Returns:
        Dictionary mapping source names to PhonemeSource objects:
        {
            'dictionary': PhonemeSource(...),
            'whisper_large_mfa': PhonemeSource(...),
            'whisper_large_wav2vec': PhonemeSource(...)
        }
    """
    # Get Whisper Large-v3 transcript if no expected text provided
    if expected_text is None:
        words = _get_whisper_words(audio_path, model_size="large-v3")
        expected_text = ' '.join(words) if words else ''

    results = {}

    # Source 1: Dictionary (CMU/G2P) - Expected pronunciation
    results['dictionary'] = get_dictionary_phonemes(expected_text)

    # Source 2: Whisper Large-v3 + MFA (for accuracy comparison)
    results['whisper_large_mfa'] = get_whisper_large_mfa_phonemes(audio_path)

    # Source 3: Whisper Large-v3 + Wav2Vec2
    results['whisper_large_wav2vec'] = get_whisper_large_wav2vec_phonemes(audio_path)

    return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def format_comparison_summary(sources: Dict[str, PhonemeSource]) -> str:
    """
    Format a text summary of phoneme sources with difference highlighting.

    Args:
        sources: Dictionary from get_all_phoneme_sources()

    Returns:
        Formatted text summary with phonetic differences highlighted
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PHONEME SOURCE COMPARISON")
    lines.append("=" * 70)

    # Display each source
    for source in sources.values():
        lines.append(f"\n{source.name}")
        lines.append(f"  {source.description}")
        lines.append("-" * 70)

        if not source.success:
            lines.append(f"  ❌ Error: {source.error}")
            continue

        if source.words:
            for pw in source.words:
                lines.append(f"  {pw.word:12} → {pw.ipa}")
        elif source.continuous_ipa:
            lines.append(f"  {source.continuous_ipa}")

    # Add phonetic difference analysis
    lines.append("\n" + "=" * 70)
    lines.append("PHONETIC DIFFERENCES")
    lines.append("=" * 70)

    dict_src = sources.get('dictionary')
    mfa_src = sources.get('whisper_large_mfa')
    wav2vec_src = sources.get('whisper_large_wav2vec')

    if dict_src and dict_src.success and dict_src.words:
        # Compare MFA vs Dictionary
        if mfa_src and mfa_src.success and mfa_src.words:
            lines.append("\n🔬 Whisper Large-v3 + MFA vs Dictionary:")
            lines.append("-" * 70)
            differences = _compare_phoneme_sources(dict_src, mfa_src)
            if differences:
                for word, dict_ipa, mfa_ipa in differences:
                    lines.append(f"  {word:12} | Expected: {dict_ipa:20} | Actual: {mfa_ipa}")
            else:
                lines.append("  ✅ No differences found")

        # Compare Wav2Vec vs Dictionary
        if wav2vec_src and wav2vec_src.success and wav2vec_src.words:
            lines.append("\n🎧 Whisper Large-v3 + Wav2Vec2 vs Dictionary:")
            lines.append("-" * 70)
            differences = _compare_phoneme_sources(dict_src, wav2vec_src)
            if differences:
                for word, dict_ipa, w2v_ipa in differences:
                    lines.append(f"  {word:12} | Expected: {dict_ipa:20} | Actual: {w2v_ipa}")
            else:
                lines.append("  ✅ No differences found")

    lines.append("=" * 70)
    return '\n'.join(lines)


def _compare_phoneme_sources(source1: PhonemeSource, source2: PhonemeSource) -> List[tuple]:
    """
    Compare two phoneme sources and return differences.

    Returns:
        List of tuples: (word, source1_ipa, source2_ipa) for words with different phonemes
    """
    if not (source1.words and source2.words):
        return []

    differences = []

    # Create word-to-ipa mappings
    s1_map = {pw.word.lower(): pw.ipa for pw in source1.words}
    s2_map = {pw.word.lower(): pw.ipa for pw in source2.words}

    # Find differences
    for word in s1_map:
        if word in s2_map:
            ipa1 = s1_map[word].strip('/')
            ipa2 = s2_map[word].strip('/')

            if ipa1 != ipa2:
                differences.append((word, s1_map[word], s2_map[word]))

    return differences


# ============================================================================
# CLI / TESTING
# ============================================================================

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python phoneme_sources.py <audio_file.wav> [expected_text]")
        sys.exit(1)

    audio_file = sys.argv[1]
    expected = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\n🎯 Extracting phonemes from: {audio_file}")
    if expected:
        print(f"📝 Expected text: {expected}")
    print()

    # Get all sources
    sources = get_all_phoneme_sources(audio_file, expected)

    # Print comparison
    print(format_comparison_summary(sources))
