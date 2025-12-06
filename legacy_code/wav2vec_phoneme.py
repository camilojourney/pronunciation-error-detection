"""
Wav2Vec2 Phoneme Recognition Module
====================================

This module uses Facebook's wav2vec2-lv-60-espeak-cv-ft model to extract
IPA phonemes directly from audio - no text/dictionary needed!

The model listens to the audio and outputs what phonemes it hears.

Author: Camilo Martinez
Course: Natural Language Processing
"""

import torch
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor
from typing import List, Optional, Tuple
from pathlib import Path
import warnings

# Try different audio loading backends
try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False

warnings.filterwarnings("ignore")

# Model configuration
MODEL_NAME = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# Global model cache (load once, reuse)
_tokenizer = None
_feature_extractor = None
_model = None


def load_model():
    """Load the wav2vec2 phoneme recognition model."""
    global _tokenizer, _feature_extractor, _model

    if _model is None:
        print("Loading wav2vec2-espeak model (first time may take a moment)...")
        _tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_NAME)
        _feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
        _model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)
        _model.eval()
        print("✓ Model loaded successfully")

    return _tokenizer, _feature_extractor, _model


def transcribe_audio_to_phonemes(audio_path: str) -> Tuple[str, List[str]]:
    """
    Transcribe audio directly to IPA phonemes using wav2vec2.

    Args:
        audio_path: Path to audio file (WAV format)

    Returns:
        Tuple of (raw_phoneme_string, list_of_phonemes)
    """
    tokenizer, feature_extractor, model = load_model()

    # Load audio
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Load audio using soundfile (more reliable)
    if SOUNDFILE_AVAILABLE:
        waveform, sample_rate = sf.read(str(audio_file))
        # Convert to numpy array and handle mono/stereo
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)  # Convert stereo to mono
        waveform = waveform.astype(np.float32)
    elif TORCHAUDIO_AVAILABLE:
        waveform, sample_rate = torchaudio.load(str(audio_file))
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0)
        waveform = waveform.squeeze().numpy()
    else:
        raise ImportError("No audio loading library available. Install soundfile or torchaudio.")

    # Resample to 16kHz if needed
    if sample_rate != 16000:
        # Simple resampling using numpy
        duration = len(waveform) / sample_rate
        new_length = int(duration * 16000)
        indices = np.linspace(0, len(waveform) - 1, new_length).astype(int)
        waveform = waveform[indices]

    # Prepare input using feature extractor
    input_values = feature_extractor(
        waveform,
        sampling_rate=16000,
        return_tensors="pt"
    ).input_values

    # Run inference
    with torch.no_grad():
        logits = model(input_values).logits

    # Decode to phonemes
    predicted_ids = torch.argmax(logits, dim=-1)
    phoneme_string = tokenizer.batch_decode(predicted_ids)[0]

    # Clean up and split into individual phonemes
    phonemes = clean_phoneme_output(phoneme_string)

    return phoneme_string, phonemes
def clean_phoneme_output(phoneme_string: str) -> List[str]:
    """
    Clean the raw phoneme output from wav2vec2.

    The model outputs espeak-style IPA as a continuous string.
    We need to split it into individual phoneme units.
    """
    # Remove special tokens and extra spaces
    cleaned = phoneme_string.strip()

    # Remove stress markers and other non-phoneme characters
    cleaned = cleaned.replace('ˈ', '').replace('ˌ', '').replace('|', '')

    # If the output has spaces, split by spaces first
    if ' ' in cleaned:
        parts = [p.strip() for p in cleaned.split() if p.strip()]
        # Then split each part into individual IPA characters
        phonemes = []
        for part in parts:
            phonemes.extend(split_ipa_string(part))
        return phonemes

    # Otherwise, split the continuous string into IPA phonemes
    return split_ipa_string(cleaned)


def split_ipa_string(ipa_string: str) -> List[str]:
    """
    Split a continuous IPA string into individual phoneme units.

    Handles multi-character IPA symbols like diphthongs (eɪ, aɪ, oʊ, etc.)
    and affricates (tʃ, dʒ).
    """
    # Multi-character IPA phonemes (order matters - longer first)
    multi_char_phonemes = [
        # Diphthongs
        'eɪ', 'aɪ', 'oʊ', 'aʊ', 'ɔɪ', 'iː', 'uː', 'ɑː', 'ɔː', 'ɜː', 'ɛː',
        # Affricates
        'tʃ', 'dʒ',
        # R-colored vowels
        'ɚ', 'ɝ', 'ɑɹ', 'ɔɹ', 'ɛɹ', 'ɪɹ', 'ʊɹ',
        # Long vowels with ː
        'iː', 'uː', 'ɑː', 'ɔː', 'ɜː',
    ]

    phonemes = []
    i = 0

    while i < len(ipa_string):
        # Check for multi-character phonemes first
        matched = False
        for mc in multi_char_phonemes:
            if ipa_string[i:].startswith(mc):
                phonemes.append(mc)
                i += len(mc)
                matched = True
                break

        if not matched:
            # Check for length marker attached to single char
            if i + 1 < len(ipa_string) and ipa_string[i + 1] == 'ː':
                phonemes.append(ipa_string[i:i+2])
                i += 2
            else:
                # Single character phoneme
                char = ipa_string[i]
                # Skip combining characters (they should have been captured with their base)
                if char not in ['ː', '̩', '̃', '̥']:
                    phonemes.append(char)
                i += 1

    return phonemes


def align_wav2vec_with_words(
    wav2vec_phonemes: List[str],
    words: List[str],
    expected_phonemes_per_word: Optional[List[List[str]]] = None
) -> List[dict]:
    """
    Align Wav2Vec2 continuous phoneme output with words from Whisper.

    Strategy: Use expected phoneme counts per word to divide the stream.
    If no expected phonemes provided, divide proportionally by word length.

    Args:
        wav2vec_phonemes: Continuous list of phonemes from Wav2Vec2
        words: List of words from Whisper
        expected_phonemes_per_word: Optional list of expected phoneme lists per word

    Returns:
        List of dicts with word and aligned phonemes
    """
    if not words or not wav2vec_phonemes:
        return []

    result = []

    if expected_phonemes_per_word:
        # Use expected phoneme counts as guide
        total_expected = sum(len(p) for p in expected_phonemes_per_word)
        total_actual = len(wav2vec_phonemes)

        if total_expected == 0:
            return []

        # Scale factor: how many actual phonemes per expected phoneme
        scale = total_actual / total_expected

        current_pos = 0
        for word, expected in zip(words, expected_phonemes_per_word):
            # Allocate proportional phonemes
            num_phonemes = max(1, int(len(expected) * scale))

            # Don't exceed remaining phonemes
            num_phonemes = min(num_phonemes, len(wav2vec_phonemes) - current_pos)

            word_phonemes = wav2vec_phonemes[current_pos:current_pos + num_phonemes]

            result.append({
                'word': word,
                'wav2vec_phonemes': word_phonemes,
                'wav2vec_ipa': ''.join(word_phonemes),
                'expected_phonemes': expected,
                'expected_ipa': ''.join(expected)
            })

            current_pos += num_phonemes

        # Handle any remaining phonemes - add to last word
        if current_pos < len(wav2vec_phonemes) and result:
            result[-1]['wav2vec_phonemes'].extend(wav2vec_phonemes[current_pos:])
            result[-1]['wav2vec_ipa'] = ''.join(result[-1]['wav2vec_phonemes'])

    else:
        # Divide proportionally by character length of words
        total_chars = sum(len(w) for w in words)
        total_phonemes = len(wav2vec_phonemes)

        if total_chars == 0:
            return []

        current_pos = 0
        for word in words:
            # Allocate proportional phonemes based on word length
            proportion = len(word) / total_chars
            num_phonemes = max(1, int(proportion * total_phonemes))

            # Don't exceed remaining phonemes
            num_phonemes = min(num_phonemes, len(wav2vec_phonemes) - current_pos)

            word_phonemes = wav2vec_phonemes[current_pos:current_pos + num_phonemes]

            result.append({
                'word': word,
                'wav2vec_phonemes': word_phonemes,
                'wav2vec_ipa': ''.join(word_phonemes)
            })

            current_pos += num_phonemes

        # Handle any remaining phonemes
        if current_pos < len(wav2vec_phonemes) and result:
            result[-1]['wav2vec_phonemes'].extend(wav2vec_phonemes[current_pos:])
            result[-1]['wav2vec_ipa'] = ''.join(result[-1]['wav2vec_phonemes'])

    return result


def get_wav2vec_with_whisper_alignment(audio_path: str) -> dict:
    """
    Combine Whisper (for words) + Wav2Vec2 (for phonemes).

    This gives us:
    - What words the person meant (Whisper)
    - What phonemes they actually produced (Wav2Vec2)

    Args:
        audio_path: Path to audio file

    Returns:
        Dictionary with word-aligned phoneme analysis
    """
    from analysis_utils import transcribe_audio, word_to_phonemes, arpabet_to_ipa, clean_text

    # Step 1: Get words from Whisper
    whisper_text = transcribe_audio(audio_path)
    whisper_text = clean_text(whisper_text)
    words = whisper_text.split()

    # Step 2: Get expected phonemes for each word (from dictionary)
    expected_per_word = []
    for word in words:
        phonemes = word_to_phonemes(word)
        ipa_phonemes = [arpabet_to_ipa(p) for p in phonemes]
        expected_per_word.append(ipa_phonemes)

    # Step 3: Get phonemes from Wav2Vec2
    raw_output, wav2vec_phonemes = transcribe_audio_to_phonemes(audio_path)

    # Step 4: Align Wav2Vec2 phonemes with words
    aligned = align_wav2vec_with_words(wav2vec_phonemes, words, expected_per_word)

    return {
        'success': True,
        'whisper_text': whisper_text,
        'words': words,
        'wav2vec_raw': raw_output,
        'word_alignments': aligned,
        'method': 'whisper+wav2vec2'
    }


def get_phonemes_for_display(audio_path: str) -> dict:
    """
    Get phonemes from audio in a format suitable for dashboard display.

    Returns:
        Dictionary with phoneme information
    """
    try:
        raw_output, phonemes = transcribe_audio_to_phonemes(audio_path)

        return {
            'success': True,
            'raw_output': raw_output,
            'phonemes': phonemes,
            'ipa_string': ' '.join(phonemes),
            'model': 'wav2vec2-espeak'
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'model': 'wav2vec2-espeak'
        }


def compare_phoneme_sources(audio_path: str, reference_text: str) -> dict:
    """
    Compare phonemes from different sources:
    1. Dictionary (expected)
    2. MFA (forced alignment)
    3. Wav2Vec2 (direct audio recognition)

    Args:
        audio_path: Path to audio file
        reference_text: The expected text

    Returns:
        Dictionary comparing all three sources
    """
    from analysis_utils import word_to_phonemes, format_phonemes_ipa

    results = {
        'audio_path': audio_path,
        'reference_text': reference_text,
        'sources': {}
    }

    # 1. Dictionary phonemes (expected)
    words = reference_text.split()
    dict_phonemes = []
    for word in words:
        clean_word = ''.join(c for c in word if c.isalpha())
        if clean_word:
            phonemes = word_to_phonemes(clean_word)
            if phonemes:
                dict_phonemes.append({
                    'word': clean_word,
                    'phonemes': format_phonemes_ipa(phonemes)
                })
    results['sources']['dictionary'] = dict_phonemes

    # 2. Wav2Vec2 (direct from audio)
    try:
        wav2vec_result = get_phonemes_for_display(audio_path)
        results['sources']['wav2vec2'] = wav2vec_result
    except Exception as e:
        results['sources']['wav2vec2'] = {'success': False, 'error': str(e)}

    # 3. MFA (if available) - handled separately in dashboard

    return results


def get_all_five_sources(audio_path: str, expected_text: Optional[str] = None) -> dict:
    """
    Get phoneme analysis from all 5 sources for comparison:

    1. Dictionary - Expected IPA from text
    2. MFA - Actual phonemes from audio (forced alignment)
    3. Wav2Vec2 - Direct audio → IPA (continuous stream)
    4. Whisper + MFA - Words from Whisper, phonemes from MFA
    5. Whisper + Wav2Vec2 - Words from Whisper, phonemes from Wav2Vec2

    Args:
        audio_path: Path to audio file
        expected_text: Optional expected text (if None, uses Whisper)

    Returns:
        Dictionary with all 5 source analyses
    """
    from analysis_utils import (
        transcribe_audio, word_to_phonemes, arpabet_to_ipa,
        format_phonemes_ipa, clean_text
    )
    from forced_alignment import get_phonemes_from_audio, mfa_phone_to_ipa

    results = {
        'audio_path': audio_path,
        'sources': {}
    }

    # Get Whisper transcription first (used by multiple sources)
    whisper_text = transcribe_audio(audio_path)
    whisper_text_clean = clean_text(whisper_text)
    whisper_words = whisper_text_clean.split()
    results['whisper_transcript'] = whisper_text_clean

    # Working text (use expected if provided, otherwise Whisper)
    working_text = expected_text if expected_text else whisper_text_clean
    working_text_clean = clean_text(working_text)
    working_words = working_text_clean.split()
    results['expected_text'] = expected_text

    # ========================================
    # SOURCE 1: Dictionary (Expected IPA)
    # ========================================
    dict_words = []
    for word in working_words:
        phonemes = word_to_phonemes(word)
        ipa = [arpabet_to_ipa(p) for p in phonemes]
        dict_words.append({
            'word': word,
            'phonemes': ipa,
            'ipa': '/' + ' '.join(ipa) + '/'
        })
    results['sources']['dictionary'] = {
        'name': 'Dictionary (Expected)',
        'description': 'IPA from CMU dictionary / G2P',
        'words': dict_words
    }

    # ========================================
    # SOURCE 2: MFA (Forced Alignment)
    # ========================================
    try:
        mfa_result = get_phonemes_from_audio(audio_path, working_text_clean)
        if mfa_result:
            mfa_words = []
            for word_align in mfa_result.words:
                ipa = [mfa_phone_to_ipa(p.phoneme) for p in word_align.phonemes]
                mfa_words.append({
                    'word': word_align.word,
                    'phonemes': ipa,
                    'ipa': '/' + ' '.join(ipa) + '/'
                })
            results['sources']['mfa'] = {
                'name': 'MFA (Forced Alignment)',
                'description': 'Actual phonemes from audio via Montreal Forced Aligner',
                'words': mfa_words,
                'success': True
            }
        else:
            results['sources']['mfa'] = {
                'name': 'MFA (Forced Alignment)',
                'success': False,
                'error': 'MFA alignment failed'
            }
    except Exception as e:
        results['sources']['mfa'] = {
            'name': 'MFA (Forced Alignment)',
            'success': False,
            'error': str(e)
        }

    # ========================================
    # SOURCE 3: Wav2Vec2 (Direct Recognition)
    # ========================================
    try:
        raw_output, phonemes = transcribe_audio_to_phonemes(audio_path)
        results['sources']['wav2vec2'] = {
            'name': 'Wav2Vec2 (Direct)',
            'description': 'Direct audio → IPA (no word boundaries)',
            'raw_output': raw_output,
            'phonemes': phonemes,
            'ipa': '/' + ''.join(phonemes) + '/',
            'success': True
        }
    except Exception as e:
        results['sources']['wav2vec2'] = {
            'name': 'Wav2Vec2 (Direct)',
            'success': False,
            'error': str(e)
        }

    # ========================================
    # SOURCE 4: Whisper + MFA
    # ========================================
    try:
        # Use Whisper text for MFA alignment
        mfa_whisper = get_phonemes_from_audio(audio_path, whisper_text_clean)
        if mfa_whisper:
            wm_words = []
            for word_align in mfa_whisper.words:
                ipa = [mfa_phone_to_ipa(p.phoneme) for p in word_align.phonemes]
                wm_words.append({
                    'word': word_align.word,
                    'phonemes': ipa,
                    'ipa': '/' + ' '.join(ipa) + '/'
                })
            results['sources']['whisper_mfa'] = {
                'name': 'Whisper + MFA',
                'description': 'Words from Whisper → Phonemes from MFA',
                'whisper_text': whisper_text_clean,
                'words': wm_words,
                'success': True
            }
        else:
            results['sources']['whisper_mfa'] = {
                'name': 'Whisper + MFA',
                'success': False,
                'error': 'MFA alignment failed'
            }
    except Exception as e:
        results['sources']['whisper_mfa'] = {
            'name': 'Whisper + MFA',
            'success': False,
            'error': str(e)
        }

    # ========================================
    # SOURCE 5: Whisper + Wav2Vec2
    # ========================================
    try:
        ww_result = get_wav2vec_with_whisper_alignment(audio_path)
        if ww_result['success']:
            ww_words = []
            for align in ww_result['word_alignments']:
                ww_words.append({
                    'word': align['word'],
                    'phonemes': align['wav2vec_phonemes'],
                    'ipa': '/' + align['wav2vec_ipa'] + '/',
                    'expected_ipa': '/' + align.get('expected_ipa', '') + '/'
                })
            results['sources']['whisper_wav2vec'] = {
                'name': 'Whisper + Wav2Vec2',
                'description': 'Words from Whisper → Phonemes aligned from Wav2Vec2',
                'whisper_text': ww_result['whisper_text'],
                'words': ww_words,
                'success': True
            }
        else:
            results['sources']['whisper_wav2vec'] = {
                'name': 'Whisper + Wav2Vec2',
                'success': False,
                'error': 'Alignment failed'
            }
    except Exception as e:
        results['sources']['whisper_wav2vec'] = {
            'name': 'Whisper + Wav2Vec2',
            'success': False,
            'error': str(e)
        }

    return results


# Quick test function
def test_on_file(audio_path: str):
    """Test the model on a single audio file."""
    print(f"\n🎤 Testing wav2vec2-espeak on: {audio_path}")
    print("=" * 60)

    result = get_phonemes_for_display(audio_path)

    if result['success']:
        print(f"Raw output: {result['raw_output']}")
        print(f"Phonemes: {result['phonemes']}")
        print(f"IPA string: /{result['ipa_string']}/")
    else:
        print(f"Error: {result['error']}")

    return result


def test_all_sources(audio_path: str):
    """Test all 5 phoneme sources on a single file."""
    print(f"\n🎯 TESTING ALL 5 PHONEME SOURCES")
    print(f"Audio: {audio_path}")
    print("=" * 70)

    results = get_all_five_sources(audio_path)

    print(f"\n📝 Whisper Transcript: {results['whisper_transcript']}")
    print()

    for source_key, source_data in results['sources'].items():
        print(f"\n{'='*60}")
        print(f"📊 {source_data['name']}")
        print(f"   {source_data.get('description', '')}")
        print("-" * 60)

        if source_data.get('success', True):
            if 'words' in source_data:
                for w in source_data['words']:
                    print(f"   {w['word']:12} → {w['ipa']}")
            elif 'ipa' in source_data:
                print(f"   {source_data['ipa']}")
        else:
            print(f"   ❌ Error: {source_data.get('error', 'Unknown')}")

    return results


if __name__ == "__main__":
    import sys

    # Test on a sample file
    if len(sys.argv) > 1:
        test_audio = sys.argv[1]
    else:
        test_audio = "l2arctic_release_v5/ABA/wav/arctic_a0005.wav"

    # Run full comparison
    test_all_sources(test_audio)
