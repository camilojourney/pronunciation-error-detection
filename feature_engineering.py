"""
Feature extraction for phoneme error classification.

Converts phoneme errors into feature dictionaries for machine learning.
"""

from typing import Dict, Any
from phoneme_properties import (
    get_phoneme_properties,
    is_minimal_pair,
    is_noticeable_error,
    is_l1_specific_error
)


def extract_features(error: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract features from a phoneme error.

    Args:
        error: Error dictionary with keys:
            - expected: Expected phoneme (CPL)
            - actual: Actual phoneme (PPL)
            - error_type: 's' (substitution), 'd' (deletion), 'a' (addition)
            - prev_phoneme: Previous phoneme (context)
            - next_phoneme: Next phoneme (context)
            - l1_language: Speaker's native language
            - speaker: Speaker ID

    Returns:
        Feature dictionary for classification
    """
    features = {}

    # Error type
    features['error_type'] = error['error_type']

    # Handle different error types
    if error['error_type'] == 's':  # Substitution
        features.update(_extract_substitution_features(error))
    elif error['error_type'] == 'd':  # Deletion
        features.update(_extract_deletion_features(error))
    elif error['error_type'] == 'a':  # Addition
        features.update(_extract_addition_features(error))

    # Native language
    features['native_language'] = error.get('native_language', 'Unknown')

    # Context features (if available)
    if error.get('prev_phoneme'):
        prev_props = get_phoneme_properties(error['prev_phoneme'])
        features['prev_type'] = prev_props.get('type', 'unknown')
    else:
        features['prev_type'] = 'none'

    if error.get('next_phoneme'):
        next_props = get_phoneme_properties(error['next_phoneme'])
        features['next_type'] = next_props.get('type', 'unknown')
    else:
        features['next_type'] = 'none'

    return features


def _extract_substitution_features(error: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features specific to substitution errors."""
    features = {}

    expected = error['expected']
    actual = error['actual']

    # Get phoneme properties
    exp_props = get_phoneme_properties(expected)
    act_props = get_phoneme_properties(actual)

    # Expected phoneme features
    features['exp_type'] = exp_props.get('type', 'unknown')
    features['exp_place'] = exp_props.get('place', 'unknown')
    features['exp_voicing'] = exp_props.get('voicing', 'unknown')

    # Actual phoneme features
    features['act_type'] = act_props.get('type', 'unknown')
    features['act_place'] = act_props.get('place', 'unknown')
    features['act_voicing'] = act_props.get('voicing', 'unknown')

    # Difference features
    features['same_type'] = exp_props.get('type') == act_props.get('type')
    features['same_place'] = exp_props.get('place') == act_props.get('place')
    features['same_voicing'] = exp_props.get('voicing') == act_props.get('voicing')

    # Linguistic impact features
    features['is_minimal_pair'] = is_minimal_pair(expected, actual)
    features['is_noticeable'] = is_noticeable_error(expected, actual)

    # L1-specific pattern
    l1_lang = error.get('native_language', 'Unknown')
    features['is_l1_pattern'] = is_l1_specific_error(expected, actual, l1_lang)

    # Common error patterns
    features.update(_detect_common_patterns(expected, actual))

    return features


def _extract_deletion_features(error: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features specific to deletion errors."""
    features = {}

    deleted = error['expected']
    del_props = get_phoneme_properties(deleted)

    # Deleted phoneme features
    features['deleted_type'] = del_props.get('type', 'unknown')
    features['deleted_place'] = del_props.get('place', 'unknown')
    features['deleted_voicing'] = del_props.get('voicing', 'unknown')

    # Deletions are often severe if they're consonants
    features['deleted_consonant'] = del_props.get('type') != 'vowel'

    # Check if it's a final consonant deletion (common L2 error)
    features['final_consonant'] = error.get('next_phoneme') == 'sil' or error.get('next_phoneme') is None

    return features


def _extract_addition_features(error: Dict[str, Any]) -> Dict[str, Any]:
    """Extract features specific to addition errors."""
    features = {}

    added = error['actual']
    add_props = get_phoneme_properties(added)

    # Added phoneme features
    features['added_type'] = add_props.get('type', 'unknown')
    features['added_place'] = add_props.get('place', 'unknown')
    features['added_voicing'] = add_props.get('voicing', 'unknown')

    # Additions are usually less severe than substitutions/deletions
    features['added_vowel'] = add_props.get('type') == 'vowel'

    return features


def _detect_common_patterns(expected: str, actual: str) -> Dict[str, bool]:
    """Detect common error patterns."""
    patterns = {}

    # Clean phonemes (remove stress markers)
    exp_clean = expected.split('*')[0].rstrip('0123456789')
    act_clean = actual.split('*')[0].rstrip('0123456789')

    # TH-sound errors (very common)
    patterns['th_to_s'] = (exp_clean == 'TH' and act_clean == 'S')
    patterns['th_to_f'] = (exp_clean == 'TH' and act_clean == 'F')
    patterns['th_to_t'] = (exp_clean == 'TH' and act_clean == 'T')
    patterns['dh_to_d'] = (exp_clean == 'DH' and act_clean == 'D')
    patterns['dh_to_z'] = (exp_clean == 'DH' and act_clean == 'Z')

    # R/L confusion
    patterns['r_to_l'] = (exp_clean == 'R' and act_clean == 'L')
    patterns['l_to_r'] = (exp_clean == 'L' and act_clean == 'R')

    # Voicing errors
    patterns['devoicing'] = _is_devoicing(exp_clean, act_clean)
    patterns['voicing'] = _is_voicing(exp_clean, act_clean)

    # V/W confusion
    patterns['v_to_w'] = (exp_clean == 'V' and act_clean == 'W')
    patterns['w_to_v'] = (exp_clean == 'W' and act_clean == 'V')

    # B/V confusion (Spanish speakers)
    patterns['b_to_v'] = (exp_clean == 'B' and act_clean == 'V')
    patterns['v_to_b'] = (exp_clean == 'V' and act_clean == 'B')

    return patterns


def _is_devoicing(expected: str, actual: str) -> bool:
    """Check if error is a devoicing substitution."""
    devoicing_pairs = [
        ('B', 'P'), ('D', 'T'), ('G', 'K'),
        ('V', 'F'), ('Z', 'S'), ('ZH', 'SH'),
        ('JH', 'CH'), ('DH', 'TH')
    ]
    return (expected, actual) in devoicing_pairs


def _is_voicing(expected: str, actual: str) -> bool:
    """Check if error is a voicing substitution."""
    voicing_pairs = [
        ('P', 'B'), ('T', 'D'), ('K', 'G'),
        ('F', 'V'), ('S', 'Z'), ('SH', 'ZH'),
        ('CH', 'JH'), ('TH', 'DH')
    ]
    return (expected, actual) in voicing_pairs


def extract_features_batch(errors: list) -> list:
    """
    Extract features for a batch of errors.

    Args:
        errors: List of error dictionaries

    Returns:
        List of feature dictionaries
    """
    return [extract_features(error) for error in errors]
