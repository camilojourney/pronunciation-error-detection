"""
Run Evaluation: Word-Level and Phoneme-Level Error Detection
============================================================

This script evaluates the pronunciation error detection system using:
1. Word-Level: Did we correctly identify which WORDS have errors?
2. Phoneme-Level: Did we correctly identify which PHONEMES have errors?

Author: Camilo Martinez
Course: Natural Language Processing
"""

from pathlib import Path
from typing import Dict, List, Set, Tuple
from dataclasses import dataclass
import pandas as pd

from ground_truth_parser import (
    parse_textgrid_annotation,
    find_annotation_for_audio,
    GroundTruthAnnotation
)
from evaluation_metrics import ConfusionMetrics


@dataclass
class WordLevelResult:
    """Result for one utterance at word level"""
    file_id: str
    speaker_id: str

    # Ground truth
    gt_error_words: Set[str]  # Words that have errors
    gt_correct_words: Set[str]  # Words with no errors

    # System predictions
    pred_error_words: Set[str]  # Words system flagged as errors
    pred_correct_words: Set[str]  # Words system said are correct

    # Metrics
    true_positives: int = 0  # Correctly detected error words
    false_positives: int = 0  # Incorrectly flagged as error
    true_negatives: int = 0  # Correctly identified as correct
    false_negatives: int = 0  # Missed error words

    def calculate_metrics(self) -> ConfusionMetrics:
        """Calculate TP/FP/TN/FN from word sets"""
        self.true_positives = len(self.gt_error_words & self.pred_error_words)
        self.false_positives = len(self.pred_error_words - self.gt_error_words)
        self.false_negatives = len(self.gt_error_words - self.pred_error_words)
        self.true_negatives = len(self.gt_correct_words & self.pred_correct_words)

        return ConfusionMetrics(
            true_positives=self.true_positives,
            false_positives=self.false_positives,
            true_negatives=self.true_negatives,
            false_negatives=self.false_negatives
        )


@dataclass
class PhonemeLevelResult:
    """Result for one utterance at phoneme level"""
    file_id: str
    speaker_id: str

    # Ground truth phoneme errors
    gt_phoneme_errors: Set[Tuple[str, str]]  # (expected, actual) pairs

    # System detected phoneme errors
    pred_phoneme_errors: Set[Tuple[str, str]]  # (expected, actual) pairs

    # Metrics
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def calculate_metrics(self) -> ConfusionMetrics:
        """Calculate TP/FP/FN from phoneme error sets"""
        self.true_positives = len(self.gt_phoneme_errors & self.pred_phoneme_errors)
        self.false_positives = len(self.pred_phoneme_errors - self.gt_phoneme_errors)
        self.false_negatives = len(self.gt_phoneme_errors - self.pred_phoneme_errors)

        # Note: TN is harder to define for phonemes (all correct phonemes)
        # For simplicity, we focus on error detection (TP/FP/FN)

        return ConfusionMetrics(
            true_positives=self.true_positives,
            false_positives=self.false_positives,
            true_negatives=0,  # Not easily defined for phonemes
            false_negatives=self.false_negatives
        )


def extract_error_words_from_ground_truth(gt: GroundTruthAnnotation, reference_text: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract which words have errors from ground truth annotation.

    Args:
        gt: Ground truth annotation with phoneme-level errors
        reference_text: Reference text sentence

    Returns:
        (error_words, correct_words) sets
    """
    # Get words from reference
    reference_words = set(reference_text.lower().split())

    # Words with errors (from ground truth word contexts)
    error_words = set()
    for error in gt.errors:
        if error.word_context:
            error_words.add(error.word_context.lower())

    # If we don't have word contexts, we can't do word-level evaluation
    if not error_words and gt.errors:
        print(f"Warning: {gt.file_id} has errors but no word contexts")
        return set(), reference_words

    # Correct words = all words minus error words
    correct_words = reference_words - error_words

    return error_words, correct_words


def extract_error_words_from_system(whisper_output: str, reference_text: str) -> Tuple[Set[str], Set[str]]:
    """
    Extract which words the system thinks have errors.

    This uses Levenshtein alignment to identify substitutions/deletions.

    Args:
        whisper_output: What Whisper transcribed
        reference_text: Expected text

    Returns:
        (error_words, correct_words) sets
    """
    from analysis_utils import align_tokens

    ref_words = reference_text.lower().split()
    hyp_words = whisper_output.lower().split()

    # Align words
    alignment = align_tokens(ref_words, hyp_words)

    error_words = set()
    correct_words = set()

    for op in alignment:
        if op.operation == 'equal':
            # Correct word
            correct_words.add(op.ref_word.lower())
        elif op.operation in ['replace', 'delete']:
            # Error word (substituted or deleted)
            if op.ref_word:
                error_words.add(op.ref_word.lower())
        # Note: insertions are extra words, not errors in reference

    return error_words, correct_words


def evaluate_word_level(
    ground_truth: GroundTruthAnnotation,
    whisper_output: str,
    reference_text: str
) -> WordLevelResult:
    """
    Evaluate word-level error detection.

    Question: Did we correctly identify which WORDS have errors?
    """
    # Extract ground truth error words
    gt_error_words, gt_correct_words = extract_error_words_from_ground_truth(
        ground_truth, reference_text
    )

    # Extract system predicted error words
    pred_error_words, pred_correct_words = extract_error_words_from_system(
        whisper_output, reference_text
    )

    # Create result
    result = WordLevelResult(
        file_id=ground_truth.file_id,
        speaker_id=ground_truth.speaker_id,
        gt_error_words=gt_error_words,
        gt_correct_words=gt_correct_words,
        pred_error_words=pred_error_words,
        pred_correct_words=pred_correct_words
    )

    # Calculate metrics
    result.calculate_metrics()

    return result


def evaluate_phoneme_level(
    ground_truth: GroundTruthAnnotation,
    system_phoneme_errors: List[Tuple[str, str]] = None
) -> PhonemeLevelResult:
    """
    Evaluate phoneme-level error detection.

    Question: Did we correctly identify which PHONEMES have errors?

    Note: This requires phoneme extraction from your system (MFA or Wav2Vec2).
    For now, we'll just extract ground truth errors.
    """
    # Ground truth phoneme errors
    gt_errors = set()
    for error in ground_truth.errors:
        gt_errors.add((error.expected_phoneme, error.actual_phoneme))

    # System detected phoneme errors
    pred_errors = set(system_phoneme_errors) if system_phoneme_errors else set()

    result = PhonemeLevelResult(
        file_id=ground_truth.file_id,
        speaker_id=ground_truth.speaker_id,
        gt_phoneme_errors=gt_errors,
        pred_phoneme_errors=pred_errors
    )

    result.calculate_metrics()

    return result


def run_evaluation_on_dataset(
    annotation_dir: str,
    results_csv: str
) -> Dict[str, ConfusionMetrics]:
    """
    Run evaluation on all annotated files.

    Args:
        annotation_dir: Directory with TextGrid annotations
        results_csv: CSV file with Whisper results (from process_l2arctic.py)

    Returns:
        Dictionary with word-level and phoneme-level metrics
    """
    # Load system results
    results_df = pd.read_csv(results_csv)

    word_level_results = []
    phoneme_level_results = []

    # Overall metrics
    overall_word = ConfusionMetrics()
    overall_phoneme = ConfusionMetrics()

    for idx, row in results_df.iterrows():
        # Find corresponding annotation
        audio_path = row.get('audio_path', '')
        if not audio_path:
            continue

        annotation_path = find_annotation_for_audio(audio_path)
        if not annotation_path:
            continue

        # Parse ground truth
        gt = parse_textgrid_annotation(annotation_path)
        if not gt:
            continue

        # Extract data from results
        reference = row['reference']
        hypothesis = row['hypothesis']

        # Evaluate word-level
        word_result = evaluate_word_level(gt, hypothesis, reference)
        word_metrics = word_result.calculate_metrics()

        # Accumulate overall metrics
        overall_word.true_positives += word_metrics.true_positives
        overall_word.false_positives += word_metrics.false_positives
        overall_word.true_negatives += word_metrics.true_negatives
        overall_word.false_negatives += word_metrics.false_negatives

        word_level_results.append({
            'file_id': gt.file_id,
            'speaker_id': gt.speaker_id,
            'gt_errors': len(word_result.gt_error_words),
            'pred_errors': len(word_result.pred_error_words),
            'tp': word_metrics.true_positives,
            'fp': word_metrics.false_positives,
            'fn': word_metrics.false_negatives,
            'precision': word_metrics.precision,
            'recall': word_metrics.recall,
            'f1': word_metrics.f1_score
        })

        # Evaluate phoneme-level (just ground truth for now)
        phoneme_result = evaluate_phoneme_level(gt)
        phoneme_level_results.append({
            'file_id': gt.file_id,
            'speaker_id': gt.speaker_id,
            'gt_phoneme_errors': len(phoneme_result.gt_phoneme_errors)
        })

    print(f"\n{'='*70}")
    print(f"Evaluation Results")
    print(f"{'='*70}")
    print(f"Files evaluated: {len(word_level_results)}")
    print(f"\nWord-Level Error Detection:")
    print(f"  Accuracy:  {overall_word.accuracy:.3f}")
    print(f"  Precision: {overall_word.precision:.3f}")
    print(f"  Recall:    {overall_word.recall:.3f}")
    print(f"  F1 Score:  {overall_word.f1_score:.3f}")
    print(f"\n  TP: {overall_word.true_positives} (correctly detected errors)")
    print(f"  FP: {overall_word.false_positives} (false alarms)")
    print(f"  TN: {overall_word.true_negatives} (correctly identified as correct)")
    print(f"  FN: {overall_word.false_negatives} (missed errors)")
    print(f"{'='*70}\n")

    # Save results
    word_df = pd.DataFrame(word_level_results)
    word_df.to_csv('data/results/word_level_evaluation.csv', index=False)
    print(f"✓ Saved: data/results/word_level_evaluation.csv")

    phoneme_df = pd.DataFrame(phoneme_level_results)
    phoneme_df.to_csv('data/results/phoneme_level_evaluation.csv', index=False)
    print(f"✓ Saved: data/results/phoneme_level_evaluation.csv")

    return {
        'word_level': overall_word,
        'phoneme_level': overall_phoneme
    }


if __name__ == '__main__':
    import sys

    # Check if results CSV exists
    results_csv = 'data/results/speaker_results.csv'

    if not Path(results_csv).exists():
        print(f"Error: {results_csv} not found!")
        print("Run process_l2arctic.py first to generate results.")
        sys.exit(1)

    print("Running evaluation...")
    print(f"Using results from: {results_csv}")

    metrics = run_evaluation_on_dataset(
        annotation_dir='l2arctic_release_v5',
        results_csv=results_csv
    )

    print("\n✓ Evaluation complete!")
    print("Check data/results/word_level_evaluation.csv for details")
