"""
Parse L2-ARCTIC TextGrid annotations and extract phoneme errors.

This module handles reading TextGrid files and extracting the three types
of phoneme errors: substitutions, deletions, and additions.
"""

import textgrid
import glob
import re
from pathlib import Path
from typing import List, Dict, Optional
import json

# Speaker to L1 language mapping
SPEAKER_L1 = {
    'ABA': 'Arabic', 'SKA': 'Arabic', 'YBAA': 'Arabic', 'ZHAA': 'Arabic',
    'BWC': 'Chinese', 'LXC': 'Chinese', 'NCC': 'Chinese', 'TXHC': 'Chinese',
    'ASI': 'Hindi', 'RRBI': 'Hindi', 'SVBI': 'Hindi', 'TNI': 'Hindi',
    'HJK': 'Korean', 'HKK': 'Korean', 'YDCK': 'Korean', 'YKWK': 'Korean',
    'EBVS': 'Spanish', 'ERMS': 'Spanish', 'MBMPS': 'Spanish', 'NJS': 'Spanish',
    'HQTV': 'Vietnamese', 'PNV': 'Vietnamese', 'THV': 'Vietnamese', 'TLV': 'Vietnamese',
    # Duplicate directories (likely backup/alternative versions)
    'PNV 2': 'Vietnamese',
    'RRBI 2': 'Hindi',
}


def parse_textgrid_errors(textgrid_file: str) -> List[Dict]:
    """
    Parse a TextGrid file and extract all phoneme errors.

    Args:
        textgrid_file: Path to TextGrid annotation file

    Returns:
        List of error dictionaries containing expected, actual, type, etc.
    """
    try:
        tg = textgrid.TextGrid.fromFile(textgrid_file)
    except Exception as e:
        print(f"Error reading {textgrid_file}: {e}")
        return []

    errors = []

    # Find the phones tier
    phones_tier = None
    for tier in tg.tiers:
        if tier.name == "phones":
            phones_tier = tier
            break

    if not phones_tier:
        return errors

    # Extract errors from intervals
    for i, interval in enumerate(phones_tier.intervals):
        text = interval.mark.strip()

        # Remove whitespace (as per README tip)
        text = re.sub(r'\s+', '', text)

        # Check if it's an error annotation (contains comma)
        if ',' in text:
            parts = text.split(',')

            if len(parts) >= 3:
                expected = parts[0]
                actual = parts[1]
                error_type = parts[2]

                # Only process s (substitution), d (deletion), a (addition)
                if error_type not in ['s', 'd', 'a']:
                    continue

                # Get context (surrounding phonemes)
                prev_phoneme = 'START'
                next_phoneme = 'END'

                if i > 0:
                    prev_text = phones_tier.intervals[i-1].mark.strip()
                    # Get just the phoneme, not the error annotation
                    if ',' in prev_text:
                        prev_phoneme = prev_text.split(',')[0]
                    else:
                        prev_phoneme = prev_text

                if i < len(phones_tier.intervals) - 1:
                    next_text = phones_tier.intervals[i+1].mark.strip()
                    if ',' in next_text:
                        next_phoneme = next_text.split(',')[0]
                    else:
                        next_phoneme = next_text

                # Determine word position (simplified)
                position = 'medial'
                if prev_phoneme in ['START', 'sil', 'sp']:
                    position = 'initial'
                elif next_phoneme in ['END', 'sil', 'sp']:
                    position = 'final'

                errors.append({
                    'expected': expected,
                    'actual': actual,
                    'error_type': error_type,
                    'time_start': interval.minTime,
                    'time_end': interval.maxTime,
                    'prev_phoneme': prev_phoneme,
                    'next_phoneme': next_phoneme,
                    'position': position,
                })

    return errors


def process_all_annotations(l2arctic_path: str = "l2arctic_release_v5") -> List[Dict]:
    """
    Process all 3,599 annotated TextGrid files and extract errors.

    Args:
        l2arctic_path: Path to L2-ARCTIC corpus root directory

    Returns:
        List of all errors with metadata
    """

    all_errors = []

    # Get all annotation files
    annotation_pattern = f"{l2arctic_path}/*/annotation/*.TextGrid"
    annotation_files = glob.glob(annotation_pattern)

    print(f"Found {len(annotation_files)} annotation files")

    for tg_file in annotation_files:
        # Extract speaker info from path
        # e.g., l2arctic_release_v5/ABA/annotation/arctic_a0001.TextGrid
        path_parts = Path(tg_file).parts
        filename = path_parts[-1]

        # Skip suitcase_corpus (separate dataset, not part of L2-ARCTIC speakers)
        if 'suitcase_corpus' in path_parts:
            continue

        # Extract speaker code from directory name
        speaker_code = path_parts[-3]

        # Get speaker's native language
        native_language = SPEAKER_L1.get(speaker_code, 'Unknown')

        # Parse errors from this file
        errors = parse_textgrid_errors(tg_file)

        # Add metadata to each error
        for error in errors:
            error['speaker'] = speaker_code
            error['native_language'] = native_language
            error['filename'] = filename
            all_errors.append(error)

    print(f"\nExtracted {len(all_errors)} total phoneme errors:")

    # Count error types
    substitutions = sum(1 for e in all_errors if e['error_type'] == 's')
    deletions = sum(1 for e in all_errors if e['error_type'] == 'd')
    additions = sum(1 for e in all_errors if e['error_type'] == 'a')

    print(f"  - Substitutions: {substitutions}")
    print(f"  - Deletions: {deletions}")
    print(f"  - Additions: {additions}")

    return all_errors


def save_errors_to_json(errors: List[Dict], output_file: str = "data/processed/all_errors.json"):
    """Save extracted errors to JSON file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(errors, f, indent=2)

    print(f"\nSaved {len(errors)} errors to {output_file}")


if __name__ == "__main__":
    # Process all annotations
    errors = process_all_annotations()

    # Save to JSON
    save_errors_to_json(errors)

    # Show some examples
    print("\nExample errors:")
    for i, error in enumerate(errors[:5]):
        print(f"\n{i+1}. {error['speaker']} ({error['native_language']})")
        print(f"   Expected: {error['expected']} → Actual: {error['actual']}")
        print(f"   Type: {error['error_type']}, Position: {error['position']}")
