"""
Phoneme properties and linguistic knowledge for feature extraction.

This module contains the linguistic knowledge needed to classify
phoneme errors by severity.
"""

from typing import Dict, Tuple, Set

# Phoneme properties based on ARPAbet phoneme set
PHONEME_PROPERTIES = {
    # Vowels
    'AA': {'type': 'vowel', 'place': 'back', 'height': 'low', 'voicing': 'voiced'},
    'AE': {'type': 'vowel', 'place': 'front', 'height': 'low', 'voicing': 'voiced'},
    'AH': {'type': 'vowel', 'place': 'central', 'height': 'mid', 'voicing': 'voiced'},
    'AO': {'type': 'vowel', 'place': 'back', 'height': 'mid', 'voicing': 'voiced'},
    'AW': {'type': 'vowel', 'place': 'back', 'height': 'low', 'voicing': 'voiced'},
    'AX': {'type': 'vowel', 'place': 'central', 'height': 'mid', 'voicing': 'voiced'},
    'AY': {'type': 'vowel', 'place': 'central', 'height': 'low', 'voicing': 'voiced'},
    'EH': {'type': 'vowel', 'place': 'front', 'height': 'mid', 'voicing': 'voiced'},
    'ER': {'type': 'vowel', 'place': 'central', 'height': 'mid', 'voicing': 'voiced'},
    'EY': {'type': 'vowel', 'place': 'front', 'height': 'mid', 'voicing': 'voiced'},
    'IH': {'type': 'vowel', 'place': 'front', 'height': 'high', 'voicing': 'voiced'},
    'IY': {'type': 'vowel', 'place': 'front', 'height': 'high', 'voicing': 'voiced'},
    'OW': {'type': 'vowel', 'place': 'back', 'height': 'mid', 'voicing': 'voiced'},
    'OY': {'type': 'vowel', 'place': 'back', 'height': 'mid', 'voicing': 'voiced'},
    'UH': {'type': 'vowel', 'place': 'back', 'height': 'high', 'voicing': 'voiced'},
    'UW': {'type': 'vowel', 'place': 'back', 'height': 'high', 'voicing': 'voiced'},

    # Stops
    'B': {'type': 'stop', 'place': 'bilabial', 'voicing': 'voiced'},
    'D': {'type': 'stop', 'place': 'alveolar', 'voicing': 'voiced'},
    'G': {'type': 'stop', 'place': 'velar', 'voicing': 'voiced'},
    'P': {'type': 'stop', 'place': 'bilabial', 'voicing': 'voiceless'},
    'T': {'type': 'stop', 'place': 'alveolar', 'voicing': 'voiceless'},
    'K': {'type': 'stop', 'place': 'velar', 'voicing': 'voiceless'},

    # Fricatives
    'DH': {'type': 'fricative', 'place': 'dental', 'voicing': 'voiced'},
    'F': {'type': 'fricative', 'place': 'labiodental', 'voicing': 'voiceless'},
    'S': {'type': 'fricative', 'place': 'alveolar', 'voicing': 'voiceless'},
    'SH': {'type': 'fricative', 'place': 'postalveolar', 'voicing': 'voiceless'},
    'TH': {'type': 'fricative', 'place': 'dental', 'voicing': 'voiceless'},
    'V': {'type': 'fricative', 'place': 'labiodental', 'voicing': 'voiced'},
    'Z': {'type': 'fricative', 'place': 'alveolar', 'voicing': 'voiced'},
    'ZH': {'type': 'fricative', 'place': 'postalveolar', 'voicing': 'voiced'},
    'HH': {'type': 'aspirate', 'place': 'glottal', 'voicing': 'voiceless'},

    # Affricates
    'CH': {'type': 'affricate', 'place': 'postalveolar', 'voicing': 'voiceless'},
    'JH': {'type': 'affricate', 'place': 'postalveolar', 'voicing': 'voiced'},

    # Nasals
    'M': {'type': 'nasal', 'place': 'bilabial', 'voicing': 'voiced'},
    'N': {'type': 'nasal', 'place': 'alveolar', 'voicing': 'voiced'},
    'NG': {'type': 'nasal', 'place': 'velar', 'voicing': 'voiced'},

    # Liquids
    'L': {'type': 'liquid', 'place': 'alveolar', 'voicing': 'voiced'},
    'R': {'type': 'liquid', 'place': 'postalveolar', 'voicing': 'voiced'},

    # Semivowels
    'W': {'type': 'semivowel', 'place': 'bilabial', 'voicing': 'voiced'},
    'Y': {'type': 'semivowel', 'place': 'palatal', 'voicing': 'voiced'},
}

# Minimal pairs: substitutions that create different words
# Format: (expected, actual) -> examples
MINIMAL_PAIRS: Set[Tuple[str, str]] = {
    # TH sounds (very common L2 errors)
    ('TH', 'S'),   # think → sink, thing → sing
    ('TH', 'F'),   # thin → fin, thank → frank
    ('TH', 'T'),   # thick → tick, math → mat
    ('DH', 'D'),   # this → dis, breathe → breed
    ('DH', 'Z'),   # this → zis, breathe → breeze

    # R/L confusion (common for Asian L1 speakers)
    ('R', 'L'),    # right → light, river → liver
    ('L', 'R'),    # light → right, alive → arrive

    # Voicing contrasts
    ('P', 'B'),    # cap → cab, rip → rib
    ('T', 'D'),    # tip → dip, bat → bad
    ('K', 'G'),    # back → bag, lock → log
    ('S', 'Z'),    # bus → buzz, rice → rise
    ('F', 'V'),    # fan → van, safe → save
    ('SH', 'ZH'),  # wash → garage (rare)

    # Affricates
    ('SH', 'CH'),  # ship → chip, wash → watch
    ('CH', 'SH'),  # chip → ship, match → mash

    # V/W confusion
    ('V', 'W'),    # vest → west, vine → wine
    ('W', 'V'),    # west → vest, wine → vine

    # B/V confusion (Spanish speakers)
    ('B', 'V'),    # berry → very, bat → vat
    ('V', 'B'),    # very → berry, vat → bat
}

# Noticeable errors: affect intelligibility but usually clear from context
NOTICEABLE_ERRORS: Set[Tuple[str, str]] = {
    ('T', 'CH'),   # tip → chip (not minimal but noticeable)
    ('D', 'JH'),   # dip → jip (noticeable)
    ('N', 'NG'),   # thin → thing (context-dependent)
    ('M', 'N'),    # sum → sun (usually clear from context)
    ('L', 'W'),    # light → wight (rare word)
}


def get_phoneme_properties(phoneme: str) -> Dict:
    """
    Get properties for a phoneme.

    Args:
        phoneme: ARPAbet phoneme symbol

    Returns:
        Dictionary of phoneme properties
    """
    # Handle phonemes with stress markers or modifications
    clean_phoneme = phoneme.split('*')[0].rstrip('0123456789')

    return PHONEME_PROPERTIES.get(clean_phoneme, {
        'type': 'unknown',
        'place': 'unknown',
        'voicing': 'unknown'
    })


def is_minimal_pair(expected: str, actual: str) -> bool:
    """
    Check if substitution creates a minimal pair.

    Args:
        expected: Expected phoneme
        actual: Actual phoneme produced

    Returns:
        True if this substitution creates minimal pairs
    """
    # Clean phonemes
    exp_clean = expected.split('*')[0].rstrip('0123456789')
    act_clean = actual.split('*')[0].rstrip('0123456789')

    return (exp_clean, act_clean) in MINIMAL_PAIRS


def is_noticeable_error(expected: str, actual: str) -> bool:
    """
    Check if error is noticeable but not critical.

    Args:
        expected: Expected phoneme
        actual: Actual phoneme produced

    Returns:
        True if this is a noticeable (but not critical) error
    """
    exp_clean = expected.split('*')[0].rstrip('0123456789')
    act_clean = actual.split('*')[0].rstrip('0123456789')

    return (exp_clean, act_clean) in NOTICEABLE_ERRORS


# Common L1-specific error patterns
L1_ERROR_PATTERNS = {
    'Arabic': {
        'common': [('P', 'B'), ('V', 'F'), ('TH', 'S'), ('TH', 'T')],
        'reason': 'Arabic lacks /p/, /v/, and interdental fricatives'
    },
    'Chinese': {
        'common': [('R', 'L'), ('TH', 'S'), ('V', 'W'), ('N', 'L')],
        'reason': 'Chinese lacks /r/, /v/, and interdental fricatives'
    },
    'Hindi': {
        'common': [('TH', 'T'), ('DH', 'D'), ('V', 'W')],
        'reason': 'Retroflexion common in Hindi, v/w confusion'
    },
    'Korean': {
        'common': [('R', 'L'), ('F', 'P'), ('V', 'B'), ('Z', 'S')],
        'reason': 'Korean lacks /f/, /v/, /z/, and r/l distinction'
    },
    'Spanish': {
        'common': [('V', 'B'), ('Y', 'J'), ('SH', 'CH'), ('Z', 'S')],
        'reason': 'Spanish lacks /v/, /sh/, /zh/ distinctions'
    },
    'Vietnamese': {
        'common': [('TH', 'T'), ('S', 'SH'), ('CH', 'TR')],
        'reason': 'Vietnamese has different consonant system'
    },
}


def is_l1_specific_error(expected: str, actual: str, l1_language: str) -> bool:
    """
    Check if error matches known L1-specific pattern.

    Args:
        expected: Expected phoneme
        actual: Actual phoneme produced
        l1_language: Speaker's native language

    Returns:
        True if this matches a known L1-specific error pattern
    """
    if l1_language not in L1_ERROR_PATTERNS:
        return False

    exp_clean = expected.split('*')[0].rstrip('0123456789')
    act_clean = actual.split('*')[0].rstrip('0123456789')

    common_errors = L1_ERROR_PATTERNS[l1_language]['common']
    return (exp_clean, act_clean) in common_errors
