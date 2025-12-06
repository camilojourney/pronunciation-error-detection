"""
Ground Truth Parser for L2-ARCTIC TextGrid Annotations
=======================================================

This module parses human-annotated TextGrid files from the L2-ARCTIC corpus
to extract phoneme-level pronunciation errors for evaluation.

TextGrid Format:
- 3 tiers: "words", "phones" (ARPABET), "IPA"
- Error notation: CPL,PPL,error_type
  - CPL = Correct Phoneme Label (expected)
  - PPL = Perceived Phoneme Label (actual)
  - error_type = 's' (substitution), 'd' (deletion), 'a' (addition)

Example: "TH,S,s" means expected /θ/ but heard /s/ (substitution)

Author: Camilo Martinez
Course: Natural Language Processing
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
import re

try:
    from praatio import textgrid
    PRAATIO_AVAILABLE = True
except ImportError:
    PRAATIO_AVAILABLE = False
    print("Warning: praatio not available. Install with: pip install praatio")


@dataclass
class GroundTruthError:
    """Single phoneme error from ground truth annotation"""
    time_start: float
    time_end: float
    expected_phoneme: str  # CPL (ARPABET or IPA)
    actual_phoneme: str    # PPL (ARPABET or IPA)
    error_type: str        # 's', 'd', 'a'
    word_context: Optional[str] = None

    def __str__(self) -> str:
        type_name = {'s': 'Substitution', 'd': 'Deletion', 'a': 'Addition'}.get(self.error_type, 'Unknown')
        return f"{type_name}: {self.expected_phoneme} → {self.actual_phoneme} [{self.time_start:.2f}-{self.time_end:.2f}s]"


@dataclass
class GroundTruthAnnotation:
    """Complete annotation for one utterance"""
    file_id: str
    speaker_id: str
    total_phonemes: int
    correct_phonemes: int
    errors: List[GroundTruthError] = field(default_factory=list)

    @property
    def phoneme_error_rate(self) -> float:
        """Phoneme Error Rate (PER): percentage of phonemes with errors"""
        return len(self.errors) / self.total_phonemes if self.total_phonemes > 0 else 0.0

    @property
    def substitution_count(self) -> int:
        return sum(1 for e in self.errors if e.error_type == 's')

    @property
    def deletion_count(self) -> int:
        return sum(1 for e in self.errors if e.error_type == 'd')

    @property
    def insertion_count(self) -> int:
        return sum(1 for e in self.errors if e.error_type == 'a')

    def __str__(self) -> str:
        return (f"GroundTruthAnnotation({self.speaker_id}/{self.file_id}): "
                f"PER={self.phoneme_error_rate:.1%}, "
                f"{self.substitution_count} subs, "
                f"{self.deletion_count} dels, "
                f"{self.insertion_count} adds")


def parse_error_tag(text: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse error tag in format: CPL,PPL,error_type

    Args:
        text: Text like "TH,S,s" or "T,sil,d"

    Returns:
        Tuple of (expected, actual, error_type) or None if not an error

    Examples:
        >>> parse_error_tag("TH,S,s")
        ('TH', 'S', 's')
        >>> parse_error_tag("DH,D,s")
        ('DH', 'D', 's')
        >>> parse_error_tag("AE1,EH,s")
        ('AE1', 'EH', 's')
        >>> parse_error_tag("T,sil,d")
        ('T', 'sil', 'd')
        >>> parse_error_tag("K")  # No error
        None
    """
    if ',' not in text:
        return None

    parts = text.split(',')
    if len(parts) < 3:
        return None

    expected = parts[0].strip()
    actual = parts[1].strip()
    error_type = parts[2].strip()

    # Validate error type
    if error_type not in ['s', 'd', 'a']:
        return None

    return (expected, actual, error_type)


def parse_textgrid_annotation(textgrid_path: str, use_ipa: bool = False) -> Optional[GroundTruthAnnotation]:
    """
    Parse TextGrid file and extract ground truth errors.

    Args:
        textgrid_path: Path to .TextGrid file
        use_ipa: If True, use IPA tier; if False, use ARPABET phones tier

    Returns:
        GroundTruthAnnotation object or None if parsing fails
    """
    if not PRAATIO_AVAILABLE:
        print("Error: praatio library not available")
        return None

    path = Path(textgrid_path)
    if not path.exists():
        print(f"Error: TextGrid file not found: {textgrid_path}")
        return None

    try:
        # Load TextGrid
        tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=False)

        # Choose tier based on preference
        tier_name = "IPA" if use_ipa else "phones"

        # Get phoneme tier
        if tier_name not in tg.tierNames:
            print(f"Warning: '{tier_name}' tier not found in {path.name}. Available tiers: {tg.tierNames}")
            # Try alternative tier names
            if "IPA" in tg.tierNames and not use_ipa:
                tier_name = "IPA"
            elif "phones" in tg.tierNames and use_ipa:
                tier_name = "phones"
            else:
                return None

        phone_tier = tg.getTier(tier_name)

        # Extract file_id and speaker_id from path
        # Expected: .../l2arctic_release_v5/SPEAKER/annotation/arctic_a0001.TextGrid
        file_id = path.stem  # e.g., "arctic_a0001"
        speaker_id = path.parent.parent.name  # e.g., "ABA"

        # Parse phonemes and errors
        errors = []
        total_phonemes = 0
        correct_phonemes = 0

        # Try to get word tier for context
        word_tier = None
        if "words" in tg.tierNames:
            word_tier = tg.getTier("words")

        for interval in phone_tier.entries:
            label = interval.label.strip()
            if not label or label == 'sp' or label == '':
                # Skip silence markers and empty intervals
                continue

            total_phonemes += 1

            # Check if this is an error annotation
            error_info = parse_error_tag(label)

            if error_info:
                expected, actual, error_type = error_info

                # Find word context if available
                word_context = None
                if word_tier:
                    # Find word that overlaps with this phoneme
                    mid_time = (interval.start + interval.end) / 2
                    for word_interval in word_tier.entries:
                        if word_interval.start <= mid_time <= word_interval.end:
                            word_context = word_interval.label
                            break

                error = GroundTruthError(
                    time_start=interval.start,
                    time_end=interval.end,
                    expected_phoneme=expected,
                    actual_phoneme=actual,
                    error_type=error_type,
                    word_context=word_context
                )
                errors.append(error)
            else:
                # No error tag, this phoneme is correct
                correct_phonemes += 1

        annotation = GroundTruthAnnotation(
            file_id=file_id,
            speaker_id=speaker_id,
            total_phonemes=total_phonemes,
            correct_phonemes=correct_phonemes,
            errors=errors
        )

        return annotation

    except Exception as e:
        print(f"Error parsing {textgrid_path}: {e}")
        return None


def find_annotation_for_audio(audio_path: str) -> Optional[str]:
    """
    Find corresponding TextGrid annotation for audio file.

    Converts path from:
        .../l2arctic_release_v5/SPEAKER/wav/arctic_a0001.wav
    to:
        .../l2arctic_release_v5/SPEAKER/annotation/arctic_a0001.TextGrid

    Args:
        audio_path: Path to audio WAV file

    Returns:
        Path to TextGrid annotation file if it exists, None otherwise
    """
    audio_path = Path(audio_path)

    # Check if this is an L2-ARCTIC file structure
    if 'wav' not in str(audio_path):
        return None

    # Replace /wav/ with /annotation/ and .wav with .TextGrid
    annotation_path = Path(str(audio_path).replace('/wav/', '/annotation/').replace('.wav', '.TextGrid'))

    if annotation_path.exists():
        return str(annotation_path)

    return None


def load_all_annotations(base_dir: str, speaker_id: Optional[str] = None) -> List[GroundTruthAnnotation]:
    """
    Load all available ground truth annotations.

    Args:
        base_dir: Base directory (e.g., "l2arctic_release_v5")
        speaker_id: Optional speaker ID to filter (e.g., "ABA")

    Returns:
        List of GroundTruthAnnotation objects
    """
    annotations = []
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Base directory not found: {base_dir}")
        return annotations

    # Find all annotation directories
    if speaker_id:
        annotation_dirs = [base_path / speaker_id / "annotation"]
    else:
        annotation_dirs = list(base_path.glob("*/annotation"))

    for annotation_dir in annotation_dirs:
        if not annotation_dir.is_dir():
            continue

        # Load all TextGrid files in this directory
        for textgrid_file in annotation_dir.glob("*.TextGrid"):
            annotation = parse_textgrid_annotation(str(textgrid_file))
            if annotation:
                annotations.append(annotation)

    return annotations


def print_annotation_summary(annotations: List[GroundTruthAnnotation]):
    """Print summary statistics for a list of annotations"""
    if not annotations:
        print("No annotations found")
        return

    total_files = len(annotations)
    total_phonemes = sum(a.total_phonemes for a in annotations)
    total_errors = sum(len(a.errors) for a in annotations)
    total_subs = sum(a.substitution_count for a in annotations)
    total_dels = sum(a.deletion_count for a in annotations)
    total_adds = sum(a.insertion_count for a in annotations)

    avg_per = total_errors / total_phonemes if total_phonemes > 0 else 0

    print(f"\n{'='*60}")
    print(f"Ground Truth Annotation Summary")
    print(f"{'='*60}")
    print(f"Total files:       {total_files}")
    print(f"Total phonemes:    {total_phonemes:,}")
    print(f"Total errors:      {total_errors:,}")
    print(f"Average PER:       {avg_per:.1%}")
    print(f"\nError breakdown:")
    print(f"  Substitutions:   {total_subs:,} ({total_subs/total_errors*100:.1f}%)")
    print(f"  Deletions:       {total_dels:,} ({total_dels/total_errors*100:.1f}%)")
    print(f"  Additions:       {total_adds:,} ({total_adds/total_errors*100:.1f}%)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Test the parser
    import sys

    if len(sys.argv) > 1:
        textgrid_path = sys.argv[1]
        print(f"Parsing: {textgrid_path}")
        annotation = parse_textgrid_annotation(textgrid_path)
        if annotation:
            print(annotation)
            print(f"\nErrors found:")
            for error in annotation.errors[:10]:  # Show first 10 errors
                print(f"  {error}")
            if len(annotation.errors) > 10:
                print(f"  ... and {len(annotation.errors) - 10} more errors")
    else:
        print("Usage: python ground_truth_parser.py <path_to_textgrid_file>")
        print("\nOr load all annotations from a directory:")
        print("  annotations = load_all_annotations('l2arctic_release_v5', 'ABA')")
