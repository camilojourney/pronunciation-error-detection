"""
Evaluation Metrics for Pronunciation Error Detection
====================================================

This module calculates precision, recall, F1 score, and accuracy for both
word-level and phoneme-level pronunciation error detection.

Compares system predictions against ground truth annotations from L2-ARCTIC.

Author: Camilo Martinez
Course: Natural Language Processing
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict, Counter
import pandas as pd

from ground_truth_parser import GroundTruthAnnotation


@dataclass
class ConfusionMetrics:
    """Basic confusion matrix metrics"""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def accuracy(self) -> float:
        """Accuracy: (TP + TN) / (TP + TN + FP + FN)"""
        total = self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
        return (self.true_positives + self.true_negatives) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        """Precision: TP / (TP + FP) - How many detected errors are real?"""
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall: TP / (TP + FN) - How many real errors did we catch?"""
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 Score: Harmonic mean of precision and recall"""
        p, r = self.precision, self.recall
        return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        """Specificity: TN / (TN + FP) - How many negatives are truly negative?"""
        denominator = self.true_negatives + self.false_positives
        return self.true_negatives / denominator if denominator > 0 else 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for easy export"""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'specificity': self.specificity,
            'true_positives': self.true_positives,
            'false_positives': self.false_positives,
            'true_negatives': self.true_negatives,
            'false_negatives': self.false_negatives
        }

    def __str__(self) -> str:
        return (f"Metrics(Acc={self.accuracy:.3f}, P={self.precision:.3f}, "
                f"R={self.recall:.3f}, F1={self.f1_score:.3f})")


@dataclass
class PhonemeLevelMetrics:
    """Metrics for phoneme-level error detection"""
    # By error type
    substitution_metrics: ConfusionMetrics = field(default_factory=ConfusionMetrics)
    deletion_metrics: ConfusionMetrics = field(default_factory=ConfusionMetrics)
    insertion_metrics: ConfusionMetrics = field(default_factory=ConfusionMetrics)

    # Overall
    overall_metrics: ConfusionMetrics = field(default_factory=ConfusionMetrics)

    # Confusion matrix: (expected, actual) → count
    confusion_matrix: Dict[Tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export"""
        return {
            'overall': self.overall_metrics.to_dict(),
            'substitution': self.substitution_metrics.to_dict(),
            'deletion': self.deletion_metrics.to_dict(),
            'insertion': self.insertion_metrics.to_dict(),
            'confusion_matrix': dict(self.confusion_matrix)
        }


@dataclass
class EvaluationResult:
    """Complete evaluation result for one utterance"""
    file_id: str
    speaker_id: str
    native_language: Optional[str] = None

    # Ground truth stats
    gt_phoneme_error_rate: float = 0.0
    gt_substitutions: int = 0
    gt_deletions: int = 0
    gt_insertions: int = 0
    gt_total_errors: int = 0

    # System prediction stats (from current error detection)
    pred_wer: float = 0.0
    pred_total_errors: int = 0

    # Evaluation metrics
    word_level_metrics: Optional[ConfusionMetrics] = None
    phoneme_level_metrics: Optional[PhonemeLevelMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for DataFrame"""
        result = {
            'file_id': self.file_id,
            'speaker_id': self.speaker_id,
            'native_language': self.native_language,
            'gt_per': self.gt_phoneme_error_rate,
            'gt_substitutions': self.gt_substitutions,
            'gt_deletions': self.gt_deletions,
            'gt_insertions': self.gt_insertions,
            'gt_total_errors': self.gt_total_errors,
            'pred_wer': self.pred_wer,
            'pred_total_errors': self.pred_total_errors
        }

        if self.word_level_metrics:
            for key, value in self.word_level_metrics.to_dict().items():
                result[f'word_{key}'] = value

        if self.phoneme_level_metrics:
            for key, value in self.phoneme_level_metrics.overall_metrics.to_dict().items():
                result[f'phoneme_{key}'] = value

        return result


def evaluate_word_level(
    predicted_errors: List[str],
    ground_truth_errors: List[str],
    total_words: int
) -> ConfusionMetrics:
    """
    Evaluate word-level error detection.

    Args:
        predicted_errors: List of words flagged as errors by system
        ground_truth_errors: List of words with errors in ground truth
        total_words: Total number of words in utterance

    Returns:
        ConfusionMetrics object
    """
    metrics = ConfusionMetrics()

    predicted_set = set(predicted_errors)
    ground_truth_set = set(ground_truth_errors)

    # True Positives: Words correctly identified as errors
    metrics.true_positives = len(predicted_set & ground_truth_set)

    # False Positives: Words incorrectly flagged as errors
    metrics.false_positives = len(predicted_set - ground_truth_set)

    # False Negatives: Errors missed by system
    metrics.false_negatives = len(ground_truth_set - predicted_set)

    # True Negatives: Words correctly identified as correct
    total_errors_union = len(predicted_set | ground_truth_set)
    metrics.true_negatives = total_words - total_errors_union

    return metrics


def evaluate_phoneme_level(
    ground_truth: GroundTruthAnnotation,
    predicted_phoneme_errors: Optional[List[Dict]] = None
) -> PhonemeLevelMetrics:
    """
    Evaluate phoneme-level error detection.

    Args:
        ground_truth: Ground truth annotation
        predicted_phoneme_errors: List of predicted phoneme errors
            Each dict should have: {'phoneme': str, 'error_type': str, 'time': float}

    Returns:
        PhonemeLevelMetrics object
    """
    metrics = PhonemeLevelMetrics()

    # For now, if we don't have phoneme-level predictions, we can only
    # count ground truth errors
    if predicted_phoneme_errors is None:
        # Just collect confusion matrix from ground truth
        for error in ground_truth.errors:
            if error.error_type == 's':  # Substitution
                metrics.confusion_matrix[(error.expected_phoneme, error.actual_phoneme)] += 1
        return metrics

    # Convert predictions to sets for comparison
    predicted_subs = {(e['phoneme'], e.get('actual', '')) for e in predicted_phoneme_errors if e['error_type'] == 's'}
    predicted_dels = {e['phoneme'] for e in predicted_phoneme_errors if e['error_type'] == 'd'}
    predicted_ins = {e['phoneme'] for e in predicted_phoneme_errors if e['error_type'] == 'a'}

    gt_subs = {(e.expected_phoneme, e.actual_phoneme) for e in ground_truth.errors if e.error_type == 's'}
    gt_dels = {e.expected_phoneme for e in ground_truth.errors if e.error_type == 'd'}
    gt_ins = {e.actual_phoneme for e in ground_truth.errors if e.error_type == 'a'}

    # Evaluate substitutions
    metrics.substitution_metrics.true_positives = len(predicted_subs & gt_subs)
    metrics.substitution_metrics.false_positives = len(predicted_subs - gt_subs)
    metrics.substitution_metrics.false_negatives = len(gt_subs - predicted_subs)

    # Evaluate deletions
    metrics.deletion_metrics.true_positives = len(predicted_dels & gt_dels)
    metrics.deletion_metrics.false_positives = len(predicted_dels - gt_dels)
    metrics.deletion_metrics.false_negatives = len(gt_dels - predicted_dels)

    # Evaluate insertions
    metrics.insertion_metrics.true_positives = len(predicted_ins & gt_ins)
    metrics.insertion_metrics.false_positives = len(predicted_ins - gt_ins)
    metrics.insertion_metrics.false_negatives = len(gt_ins - predicted_ins)

    # Overall metrics
    metrics.overall_metrics.true_positives = (
        metrics.substitution_metrics.true_positives +
        metrics.deletion_metrics.true_positives +
        metrics.insertion_metrics.true_positives
    )
    metrics.overall_metrics.false_positives = (
        metrics.substitution_metrics.false_positives +
        metrics.deletion_metrics.false_positives +
        metrics.insertion_metrics.false_positives
    )
    metrics.overall_metrics.false_negatives = (
        metrics.substitution_metrics.false_negatives +
        metrics.deletion_metrics.false_negatives +
        metrics.insertion_metrics.false_negatives
    )

    # Build confusion matrix
    for error in ground_truth.errors:
        if error.error_type == 's':
            metrics.confusion_matrix[(error.expected_phoneme, error.actual_phoneme)] += 1

    return metrics


def aggregate_results_by_language(results: List[EvaluationResult]) -> pd.DataFrame:
    """
    Aggregate evaluation results by native language.

    Args:
        results: List of EvaluationResult objects

    Returns:
        DataFrame with per-language statistics
    """
    language_data = defaultdict(list)

    for result in results:
        if result.native_language:
            language_data[result.native_language].append(result)

    aggregated = []
    for language, lang_results in language_data.items():
        # Aggregate ground truth stats
        total_gt_errors = sum(r.gt_total_errors for r in lang_results)
        total_gt_subs = sum(r.gt_substitutions for r in lang_results)
        total_gt_dels = sum(r.gt_deletions for r in lang_results)
        total_gt_ins = sum(r.gt_insertions for r in lang_results)
        avg_per = sum(r.gt_phoneme_error_rate for r in lang_results) / len(lang_results)

        # Aggregate word-level metrics
        word_metrics = ConfusionMetrics()
        for r in lang_results:
            if r.word_level_metrics:
                word_metrics.true_positives += r.word_level_metrics.true_positives
                word_metrics.false_positives += r.word_level_metrics.false_positives
                word_metrics.true_negatives += r.word_level_metrics.true_negatives
                word_metrics.false_negatives += r.word_level_metrics.false_negatives

        # Aggregate phoneme-level metrics
        phoneme_metrics = ConfusionMetrics()
        for r in lang_results:
            if r.phoneme_level_metrics:
                overall = r.phoneme_level_metrics.overall_metrics
                phoneme_metrics.true_positives += overall.true_positives
                phoneme_metrics.false_positives += overall.false_positives
                phoneme_metrics.true_negatives += overall.true_negatives
                phoneme_metrics.false_negatives += overall.false_negatives

        aggregated.append({
            'language': language,
            'sample_size': len(lang_results),
            'avg_per': avg_per,
            'total_errors': total_gt_errors,
            'substitutions': total_gt_subs,
            'deletions': total_gt_dels,
            'insertions': total_gt_ins,
            'word_accuracy': word_metrics.accuracy,
            'word_precision': word_metrics.precision,
            'word_recall': word_metrics.recall,
            'word_f1': word_metrics.f1_score,
            'phoneme_accuracy': phoneme_metrics.accuracy,
            'phoneme_precision': phoneme_metrics.precision,
            'phoneme_recall': phoneme_metrics.recall,
            'phoneme_f1': phoneme_metrics.f1_score
        })

    return pd.DataFrame(aggregated)


def get_top_confusions(
    results: List[EvaluationResult],
    top_n: int = 20,
    language: Optional[str] = None
) -> pd.DataFrame:
    """
    Get most common phoneme confusions.

    Args:
        results: List of EvaluationResult objects
        top_n: Number of top confusions to return
        language: Optional filter by native language

    Returns:
        DataFrame with top confusions
    """
    confusion_counts = Counter()

    for result in results:
        if language and result.native_language != language:
            continue

        if result.phoneme_level_metrics:
            for (expected, actual), count in result.phoneme_level_metrics.confusion_matrix.items():
                confusion_counts[(expected, actual)] += count

    # Convert to DataFrame
    confusions = []
    for (expected, actual), count in confusion_counts.most_common(top_n):
        confusions.append({
            'expected_phoneme': expected,
            'actual_phoneme': actual,
            'count': count,
            'language': language if language else 'All'
        })

    return pd.DataFrame(confusions)


def print_evaluation_summary(results: List[EvaluationResult]):
    """Print summary of evaluation results"""
    if not results:
        print("No evaluation results")
        return

    print(f"\n{'='*70}")
    print(f"Evaluation Summary")
    print(f"{'='*70}")
    print(f"Total utterances evaluated: {len(results)}")

    # Word-level summary
    word_metrics = ConfusionMetrics()
    for r in results:
        if r.word_level_metrics:
            word_metrics.true_positives += r.word_level_metrics.true_positives
            word_metrics.false_positives += r.word_level_metrics.false_positives
            word_metrics.true_negatives += r.word_level_metrics.true_negatives
            word_metrics.false_negatives += r.word_level_metrics.false_negatives

    print(f"\nWord-Level Detection:")
    print(f"  Accuracy:  {word_metrics.accuracy:.3f}")
    print(f"  Precision: {word_metrics.precision:.3f}")
    print(f"  Recall:    {word_metrics.recall:.3f}")
    print(f"  F1 Score:  {word_metrics.f1_score:.3f}")

    # Phoneme-level summary
    phoneme_metrics = ConfusionMetrics()
    for r in results:
        if r.phoneme_level_metrics:
            overall = r.phoneme_level_metrics.overall_metrics
            phoneme_metrics.true_positives += overall.true_positives
            phoneme_metrics.false_positives += overall.false_positives
            phoneme_metrics.true_negatives += overall.true_negatives
            phoneme_metrics.false_negatives += overall.false_negatives

    print(f"\nPhoneme-Level Detection:")
    print(f"  Accuracy:  {phoneme_metrics.accuracy:.3f}")
    print(f"  Precision: {phoneme_metrics.precision:.3f}")
    print(f"  Recall:    {phoneme_metrics.recall:.3f}")
    print(f"  F1 Score:  {phoneme_metrics.f1_score:.3f}")

    print(f"{'='*70}\n")


# Alias for backward compatibility
aggregate_metrics_by_language = aggregate_results_by_language


if __name__ == '__main__':
    # Example usage
    print("Evaluation Metrics Module")
    print("Import this module to calculate precision, recall, F1, and accuracy")
    print("\nExample:")
    print("  from evaluation_metrics import evaluate_word_level, ConfusionMetrics")
    print("  metrics = evaluate_word_level(predicted=['the', 'cat'], ground_truth=['the'], total_words=3)")
    print("  print(f'F1 Score: {metrics.f1_score:.3f}')")
