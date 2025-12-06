"""
Montreal Forced Aligner (MFA) Integration for Pronunciation Error Detection
============================================================================

This module provides functions to:
1. Run Montreal Forced Aligner on audio + text
2. Extract actual phoneme timings from aligned TextGrid output
3. Compare expected vs actual phonemes for error detection

Phase 2 Implementation: Getting REAL phonemes from audio

Author: Camilo Martinez
Course: Natural Language Processing

Installation Requirements:
--------------------------
MFA must be installed via conda (not pip):
    conda install -c conda-forge montreal-forced-aligner

Download English acoustic model and dictionary:
    mfa model download acoustic english_us_arpa
    mfa model download dictionary english_us_arpa
"""

import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re

# Try to import praatio for TextGrid parsing
try:
    from praatio import textgrid
    PRAATIO_AVAILABLE = True
except ImportError:
    PRAATIO_AVAILABLE = False
    print("Warning: praatio not installed. Run: pip install praatio")


@dataclass
class PhonemeInterval:
    """Represents a phoneme with its timing from audio."""
    phoneme: str       # The phoneme (in ARPABET or IPA)
    start_time: float  # Start time in seconds
    end_time: float    # End time in seconds
    duration: float    # Duration in seconds

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration * 1000


@dataclass
class WordAlignment:
    """Word-level alignment with phoneme details."""
    word: str
    start_time: float
    end_time: float
    phonemes: List[PhonemeInterval]


@dataclass
class UtteranceAlignment:
    """Complete alignment for an utterance."""
    audio_path: str
    transcript: str
    words: List[WordAlignment]
    all_phonemes: List[PhonemeInterval]


# MFA binary path (installed via conda)
MFA_BIN = "/opt/anaconda3/envs/mfa/bin/mfa"


def check_mfa_installed() -> bool:
    """Check if Montreal Forced Aligner is installed and accessible."""
    try:
        result = subprocess.run(
            [MFA_BIN, 'version'],
            capture_output=True,
            text=True,
            timeout=30
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_mfa_models() -> Dict[str, bool]:
    """Check which MFA models are downloaded."""
    models = {
        'acoustic_english': False,
        'dictionary_english': False
    }

    try:
        # Check acoustic models
        result = subprocess.run(
            [MFA_BIN, 'model', 'list', 'acoustic'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if 'english' in result.stdout.lower():
            models['acoustic_english'] = True

        # Check dictionaries
        result = subprocess.run(
            [MFA_BIN, 'model', 'list', 'dictionary'],
            capture_output=True,
            text=True,
            timeout=30
        )
        if 'english' in result.stdout.lower():
            models['dictionary_english'] = True

    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return models


def download_mfa_models():
    """Download required MFA models for English."""
    print("Downloading MFA English acoustic model...")
    subprocess.run([MFA_BIN, 'model', 'download', 'acoustic', 'english_us_arpa'], check=True)

    print("Downloading MFA English dictionary...")
    subprocess.run([MFA_BIN, 'model', 'download', 'dictionary', 'english_us_arpa'], check=True)

    print("✓ MFA models downloaded successfully")


def align_single_file(
    audio_path: str,
    transcript: str,
    output_dir: Optional[str] = None,
    acoustic_model: str = 'english_us_arpa',
    dictionary: str = 'english_us_arpa'
) -> Optional[str]:
    """
    Run MFA alignment on a single audio file.

    Args:
        audio_path: Path to audio file (WAV)
        transcript: Text transcript of the audio
        output_dir: Directory for output TextGrid (temp if None)
        acoustic_model: MFA acoustic model name
        dictionary: MFA dictionary name

    Returns:
        Path to output TextGrid file, or None if alignment failed
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        return None

    # Create temp directory for alignment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # MFA expects: corpus_dir/speaker_name/audio.wav + audio.txt
        speaker_dir = temp_path / "speaker"
        speaker_dir.mkdir()

        # Copy/link audio file
        audio_dest = speaker_dir / audio_path.name
        os.symlink(audio_path.absolute(), audio_dest)

        # Create transcript file (same name as audio, .txt extension)
        transcript_path = speaker_dir / (audio_path.stem + ".txt")
        transcript_path.write_text(transcript)

        # Output directory
        if output_dir is None:
            output_path = temp_path / "aligned"
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Run MFA align using conda run to ensure proper environment
        try:
            result = subprocess.run(
                [
                    'conda', 'run', '-n', 'mfa',
                    'mfa', 'align',
                    str(temp_path),           # corpus directory
                    dictionary,                # dictionary
                    acoustic_model,            # acoustic model
                    str(output_path),          # output directory
                    '--clean',
                    '--single_speaker',
                    '-j', '1'                  # single job for single file
                ],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per file
            )

            if result.returncode != 0:
                print(f"MFA alignment failed: {result.stderr}")
                return None

            # Find output TextGrid
            textgrid_path = output_path / "speaker" / (audio_path.stem + ".TextGrid")

            if textgrid_path.exists():
                # If we need to keep the file, copy it
                if output_dir:
                    return str(textgrid_path)
                else:
                    # Read and return the content for immediate processing
                    return str(textgrid_path)
            else:
                print(f"TextGrid not found at expected path: {textgrid_path}")
                return None

        except subprocess.TimeoutExpired:
            print("MFA alignment timed out")
            return None
        except Exception as e:
            print(f"MFA alignment error: {e}")
            return None


def parse_textgrid(textgrid_path: str) -> Optional[UtteranceAlignment]:
    """
    Parse MFA output TextGrid to extract phoneme alignments.

    Args:
        textgrid_path: Path to TextGrid file from MFA

    Returns:
        UtteranceAlignment with word and phoneme timings
    """
    if not PRAATIO_AVAILABLE:
        print("Error: praatio library required. Install with: pip install praatio")
        return None

    try:
        tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
    except Exception as e:
        print(f"Error reading TextGrid: {e}")
        return None

    words = []
    all_phonemes = []

    # MFA creates tiers: "words" and "phones"
    word_tier = None
    phone_tier = None

    for tier_name in tg.tierNames:
        tier_lower = tier_name.lower()
        if 'word' in tier_lower:
            word_tier = tg.getTier(tier_name)
        elif 'phone' in tier_lower:
            phone_tier = tg.getTier(tier_name)

    if phone_tier is None:
        print("No phone tier found in TextGrid")
        return None

    # Extract all phonemes
    for interval in phone_tier.entries:
        if interval.label and interval.label not in ['', 'sil', 'sp', 'spn']:
            phoneme = PhonemeInterval(
                phoneme=interval.label,
                start_time=interval.start,
                end_time=interval.end,
                duration=interval.end - interval.start
            )
            all_phonemes.append(phoneme)

    # Extract words with their phonemes
    if word_tier:
        for word_interval in word_tier.entries:
            if word_interval.label and word_interval.label not in ['', 'sil', 'sp', 'spn']:
                # Find phonemes within this word's time range
                word_phonemes = [
                    p for p in all_phonemes
                    if p.start_time >= word_interval.start - 0.01 and
                       p.end_time <= word_interval.end + 0.01
                ]

                word_align = WordAlignment(
                    word=word_interval.label,
                    start_time=word_interval.start,
                    end_time=word_interval.end,
                    phonemes=word_phonemes
                )
                words.append(word_align)

    return UtteranceAlignment(
        audio_path=textgrid_path.replace('.TextGrid', '.wav'),
        transcript=' '.join(w.word for w in words),
        words=words,
        all_phonemes=all_phonemes
    )


def get_phonemes_from_audio(
    audio_path: str,
    transcript: str
) -> Optional[UtteranceAlignment]:
    """
    Main function: Get actual phonemes from audio using forced alignment.

    This is the key function for Phase 2 - it extracts what phonemes
    were ACTUALLY pronounced in the audio, with timings.

    Args:
        audio_path: Path to audio file (WAV format preferred)
        transcript: Reference text that was read

    Returns:
        UtteranceAlignment with actual phonemes from audio
    """
    if not check_mfa_installed():
        print("Error: MFA not installed. Install with:")
        print("  conda install -c conda-forge montreal-forced-aligner")
        return None

    # Create temp directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        textgrid_path = align_single_file(
            audio_path=audio_path,
            transcript=transcript,
            output_dir=temp_dir
        )

        if textgrid_path and Path(textgrid_path).exists():
            return parse_textgrid(textgrid_path)

    return None


def mfa_phone_to_ipa(mfa_phone: str) -> str:
    """
    Convert MFA/ARPABET phone symbol to IPA.

    MFA uses ARPABET-like symbols. This maps them to IPA for display.
    """
    # Remove stress markers (0, 1, 2)
    phone = re.sub(r'[012]$', '', mfa_phone.upper())

    # ARPABET to IPA mapping (same as in analysis_utils.py)
    arpabet_to_ipa_map = {
        # Consonants
        'P': 'p', 'B': 'b', 'T': 't', 'D': 'd', 'K': 'k', 'G': 'ɡ',
        'F': 'f', 'V': 'v', 'TH': 'θ', 'DH': 'ð', 'S': 's', 'Z': 'z',
        'SH': 'ʃ', 'ZH': 'ʒ', 'HH': 'h', 'M': 'm', 'N': 'n', 'NG': 'ŋ',
        'L': 'l', 'R': 'ɹ', 'W': 'w', 'Y': 'j', 'CH': 'tʃ', 'JH': 'dʒ',
        # Vowels
        'IY': 'i', 'IH': 'ɪ', 'EH': 'ɛ', 'AE': 'æ', 'AA': 'ɑ', 'AO': 'ɔ',
        'UH': 'ʊ', 'UW': 'u', 'AH': 'ʌ', 'ER': 'ɚ', 'AX': 'ə',
        # Diphthongs
        'EY': 'eɪ', 'AY': 'aɪ', 'OW': 'oʊ', 'AW': 'aʊ', 'OY': 'ɔɪ',
    }

    return arpabet_to_ipa_map.get(phone, phone.lower())


def compare_expected_vs_actual(
    expected_text: str,
    actual_alignment: UtteranceAlignment
) -> Dict:
    """
    Compare expected phonemes (from dictionary) vs actual (from audio).

    This is where we can identify pronunciation errors by comparing
    what the speaker SHOULD have said vs what they ACTUALLY said.

    Args:
        expected_text: The reference/expected text
        actual_alignment: Alignment from audio with actual phonemes

    Returns:
        Dictionary with comparison results
    """
    from analysis_utils import word_to_phonemes, format_phonemes_ipa

    results = {
        'words': [],
        'total_expected_phonemes': 0,
        'total_actual_phonemes': 0,
        'matching_phonemes': 0,
        'mismatched_phonemes': 0
    }

    expected_words = expected_text.lower().split()
    actual_words = actual_alignment.words

    # Compare word by word
    for i, expected_word in enumerate(expected_words):
        word_result = {
            'expected_word': expected_word,
            'actual_word': None,
            'expected_phonemes': [],
            'actual_phonemes': [],
            'expected_ipa': '',
            'actual_ipa': '',
            'match': False
        }

        # Get expected phonemes from dictionary
        expected_phonemes = word_to_phonemes(expected_word)
        word_result['expected_phonemes'] = expected_phonemes
        word_result['expected_ipa'] = format_phonemes_ipa(expected_phonemes)
        results['total_expected_phonemes'] += len(expected_phonemes)

        # Get actual phonemes if we have alignment for this word position
        if i < len(actual_words):
            actual_word = actual_words[i]
            word_result['actual_word'] = actual_word.word

            # Get phonemes from the alignment
            actual_phonemes = [p.phoneme for p in actual_word.phonemes]
            word_result['actual_phonemes'] = actual_phonemes
            word_result['actual_ipa'] = '/' + ''.join(
                mfa_phone_to_ipa(p) for p in actual_phonemes
            ) + '/'
            results['total_actual_phonemes'] += len(actual_phonemes)

            # Check if phonemes match
            if expected_phonemes == actual_phonemes:
                word_result['match'] = True
                results['matching_phonemes'] += len(expected_phonemes)
            else:
                results['mismatched_phonemes'] += max(
                    len(expected_phonemes), len(actual_phonemes)
                )

        results['words'].append(word_result)

    return results


# ============================================================================
# ALTERNATIVE: Use existing L2-Arctic TextGrid annotations
# ============================================================================

def parse_l2arctic_textgrid(textgrid_path: str) -> Optional[UtteranceAlignment]:
    """
    Parse L2-ARCTIC dataset TextGrid files which already have annotations.

    The L2-ARCTIC corpus comes with manually annotated TextGrid files
    that contain phoneme-level transcriptions. This function parses those
    existing annotations instead of running MFA.

    Args:
        textgrid_path: Path to L2-ARCTIC annotation TextGrid

    Returns:
        UtteranceAlignment with phoneme data from annotations
    """
    if not PRAATIO_AVAILABLE:
        print("Error: praatio library required. Install with: pip install praatio")
        return None

    try:
        tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)
    except Exception as e:
        print(f"Error reading TextGrid: {e}")
        return None

    # L2-ARCTIC TextGrids have these tiers:
    # - words: word-level transcription
    # - phones: phone-level transcription (what they actually said)
    # - phones-canonical: what they should have said

    words = []
    all_phonemes = []
    canonical_phonemes = []

    word_tier = None
    phone_tier = None
    canonical_tier = None

    for tier_name in tg.tierNames:
        tier_lower = tier_name.lower()
        if tier_lower == 'words':
            word_tier = tg.getTier(tier_name)
        elif tier_lower == 'phones':
            phone_tier = tg.getTier(tier_name)
        elif 'canonical' in tier_lower:
            canonical_tier = tg.getTier(tier_name)

    if phone_tier:
        for interval in phone_tier.entries:
            if interval.label and interval.label.strip():
                label = interval.label.strip()
                # Skip silence markers
                if label not in ['', 'sil', 'sp', 'spn', 'SIL', 'SP']:
                    phoneme = PhonemeInterval(
                        phoneme=label,
                        start_time=interval.start,
                        end_time=interval.end,
                        duration=interval.end - interval.start
                    )
                    all_phonemes.append(phoneme)

    if word_tier:
        for word_interval in word_tier.entries:
            if word_interval.label and word_interval.label.strip():
                label = word_interval.label.strip()
                if label not in ['', 'sil', 'sp', 'spn', 'SIL', 'SP']:
                    # Find phonemes within this word's time range
                    word_phonemes = [
                        p for p in all_phonemes
                        if p.start_time >= word_interval.start - 0.01 and
                           p.end_time <= word_interval.end + 0.01
                    ]

                    word_align = WordAlignment(
                        word=label,
                        start_time=word_interval.start,
                        end_time=word_interval.end,
                        phonemes=word_phonemes
                    )
                    words.append(word_align)

    return UtteranceAlignment(
        audio_path=textgrid_path.replace('.TextGrid', '.wav'),
        transcript=' '.join(w.word for w in words),
        words=words,
        all_phonemes=all_phonemes
    )


# ============================================================================
# CLI / Testing
# ============================================================================

if __name__ == '__main__':
    print("Montreal Forced Aligner Integration")
    print("=" * 50)

    # Check installation
    print("\n1. Checking MFA installation...")
    if check_mfa_installed():
        print("   ✓ MFA is installed")

        print("\n2. Checking MFA models...")
        models = check_mfa_models()
        for model, available in models.items():
            status = "✓" if available else "✗"
            print(f"   {status} {model}")

        if not all(models.values()):
            print("\n   To download missing models, run:")
            print("   mfa model download acoustic english_us_arpa")
            print("   mfa model download dictionary english_us_arpa")
    else:
        print("   ✗ MFA not installed")
        print("\n   To install MFA:")
        print("   conda install -c conda-forge montreal-forced-aligner")

    # Test with L2-ARCTIC TextGrid parsing
    print("\n3. Testing L2-ARCTIC TextGrid parsing...")
    test_tg = Path("l2arctic_release_v5/ABA/annotation/arctic_a0003.TextGrid")
    if test_tg.exists():
        alignment = parse_l2arctic_textgrid(str(test_tg))
        if alignment:
            print(f"   ✓ Parsed TextGrid successfully")
            print(f"   Transcript: {alignment.transcript[:50]}...")
            print(f"   Words: {len(alignment.words)}")
            print(f"   Phonemes: {len(alignment.all_phonemes)}")

            # Show first few phonemes
            if alignment.all_phonemes:
                print("\n   First 10 phonemes from audio:")
                for p in alignment.all_phonemes[:10]:
                    ipa = mfa_phone_to_ipa(p.phoneme)
                    print(f"     {p.phoneme} → {ipa} ({p.start_time:.3f}s - {p.end_time:.3f}s)")
    else:
        print(f"   Test file not found: {test_tg}")

    print("\n" + "=" * 50)
    print("Phase 2 module ready for integration!")
