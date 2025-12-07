"""
Model evaluation and hyperparameter tuning.

This module evaluates the trained classifier using standard metrics
and performs hyperparameter tuning by testing different feature combinations.
"""

import random
from typing import List, Tuple, Dict, Any
from collections import defaultdict
import nltk
from nltk.classify import NaiveBayesClassifier, accuracy

from train_classifier import prepare_training_data, label_error_severity
from feature_engineering import extract_features


def evaluate_classifier(
    classifier: NaiveBayesClassifier,
    test_data: List[Tuple[Dict, str]]
) -> Dict[str, Any]:
    """
    Evaluate classifier with comprehensive metrics.

    Args:
        classifier: Trained classifier
        test_data: List of (features, label) tuples for testing

    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    # Overall accuracy
    acc = accuracy(classifier, test_data)
    print(f"\nOverall Accuracy: {acc:.3f}")

    # Compute predictions for all test examples
    predictions = []
    true_labels = []

    for features, true_label in test_data:
        pred_label = classifier.classify(features)
        predictions.append(pred_label)
        true_labels.append(true_label)

    # Precision, Recall, F1 per class
    print("\nPer-Class Metrics:")
    print("-" * 60)

    classes = ['HIGH', 'MEDIUM', 'LOW']
    class_metrics = {}

    for cls in classes:
        tp = sum(1 for p, t in zip(predictions, true_labels) if p == cls and t == cls)
        fp = sum(1 for p, t in zip(predictions, true_labels) if p == cls and t != cls)
        fn = sum(1 for p, t in zip(predictions, true_labels) if p != cls and t == cls)
        tn = sum(1 for p, t in zip(predictions, true_labels) if p != cls and t != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': sum(1 for t in true_labels if t == cls)
        }

        print(f"{cls:8s}  Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}  Support: {class_metrics[cls]['support']}")

    # Macro-averaged F1
    macro_f1 = sum(m['f1'] for m in class_metrics.values()) / len(classes)
    print(f"\nMacro-averaged F1: {macro_f1:.3f}")

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 60)
    confusion = defaultdict(lambda: defaultdict(int))
    for pred, true in zip(predictions, true_labels):
        confusion[true][pred] += 1

    print(f"{'':8s}  " + "  ".join(f"{cls:8s}" for cls in classes))
    for true_cls in classes:
        row = [confusion[true_cls][pred_cls] for pred_cls in classes]
        print(f"{true_cls:8s}  " + "  ".join(f"{count:8d}" for count in row))

    return {
        'accuracy': acc,
        'macro_f1': macro_f1,
        'class_metrics': class_metrics,
        'confusion_matrix': dict(confusion),
        'predictions': predictions,
        'true_labels': true_labels
    }


def cross_validate(
    training_data: List[Tuple[Dict, str]],
    n_folds: int = 10,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation.

    Args:
        training_data: List of (features, label) tuples
        n_folds: Number of folds
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with cross-validation results
    """
    print("\n" + "=" * 60)
    print(f"{n_folds}-FOLD CROSS-VALIDATION")
    print("=" * 60)

    # Shuffle data
    random.seed(random_seed)
    shuffled_data = training_data[:]
    random.shuffle(shuffled_data)

    # Split into folds
    fold_size = len(shuffled_data) // n_folds
    folds = [shuffled_data[i*fold_size:(i+1)*fold_size] for i in range(n_folds)]

    # Store results
    fold_accuracies = []
    fold_f1_scores = []

    for i in range(n_folds):
        # Create train/test split
        test_fold = folds[i]
        train_folds = [folds[j] for j in range(n_folds) if j != i]
        train_data = [item for fold in train_folds for item in fold]

        # Train classifier
        classifier = NaiveBayesClassifier.train(train_data)

        # Evaluate
        acc = accuracy(classifier, test_fold)
        fold_accuracies.append(acc)

        # Calculate macro F1
        predictions = [classifier.classify(features) for features, _ in test_fold]
        true_labels = [label for _, label in test_fold]

        classes = ['HIGH', 'MEDIUM', 'LOW']
        f1_scores = []

        for cls in classes:
            tp = sum(1 for p, t in zip(predictions, true_labels) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(predictions, true_labels) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(predictions, true_labels) if p != cls and t == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores)
        fold_f1_scores.append(macro_f1)

        print(f"Fold {i+1:2d}: Accuracy = {acc:.3f}, Macro-F1 = {macro_f1:.3f}")

    # Summary statistics
    mean_acc = sum(fold_accuracies) / n_folds
    std_acc = (sum((x - mean_acc) ** 2 for x in fold_accuracies) / n_folds) ** 0.5

    mean_f1 = sum(fold_f1_scores) / n_folds
    std_f1 = (sum((x - mean_f1) ** 2 for x in fold_f1_scores) / n_folds) ** 0.5

    print(f"\nMean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print(f"Mean Macro-F1: {mean_f1:.3f} ± {std_f1:.3f}")

    return {
        'fold_accuracies': fold_accuracies,
        'fold_f1_scores': fold_f1_scores,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'mean_f1': mean_f1,
        'std_f1': std_f1
    }


def test_feature_combinations(
    training_data: List[Tuple[Dict, str]],
    test_split: float = 0.2,
    random_seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Test different feature combinations (hyperparameter tuning).

    Args:
        training_data: List of (features, label) tuples
        test_split: Fraction for test set
        random_seed: Random seed

    Returns:
        List of results for each feature combination
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING: FEATURE COMBINATIONS")
    print("=" * 60)

    # Split data
    random.seed(random_seed)
    random.shuffle(training_data)
    split_point = int(len(training_data) * (1 - test_split))
    train_set = training_data[:split_point]
    test_set = training_data[split_point:]

    # Feature combinations to test
    feature_sets = {
        'baseline': ['error_type', 'exp_type', 'act_type'],
        'with_place': ['error_type', 'exp_type', 'act_type', 'exp_place', 'act_place'],
        'with_voicing': ['error_type', 'exp_type', 'act_type', 'exp_voicing', 'act_voicing'],
        'with_similarity': ['error_type', 'same_type', 'same_place', 'same_voicing'],
        'with_patterns': ['error_type', 'is_minimal_pair', 'is_noticeable', 'is_l1_pattern'],
        'with_context': ['error_type', 'exp_type', 'act_type', 'prev_type', 'next_type'],
        'with_l1': ['error_type', 'exp_type', 'act_type', 'l1_language'],
        'all_features': None  # Use all features
    }

    results = []

    for name, feature_subset in feature_sets.items():
        print(f"\nTesting feature set: {name}")

        # Filter features if needed
        if feature_subset is not None:
            filtered_train = [(filter_features(f, feature_subset), label)
                             for f, label in train_set]
            filtered_test = [(filter_features(f, feature_subset), label)
                            for f, label in test_set]
        else:
            filtered_train = train_set
            filtered_test = test_set

        # Train classifier
        classifier = NaiveBayesClassifier.train(filtered_train)

        # Evaluate
        acc = accuracy(classifier, filtered_test)

        # Calculate macro F1
        predictions = [classifier.classify(features) for features, _ in filtered_test]
        true_labels = [label for _, label in filtered_test]

        classes = ['HIGH', 'MEDIUM', 'LOW']
        f1_scores = []

        for cls in classes:
            tp = sum(1 for p, t in zip(predictions, true_labels) if p == cls and t == cls)
            fp = sum(1 for p, t in zip(predictions, true_labels) if p == cls and t != cls)
            fn = sum(1 for p, t in zip(predictions, true_labels) if p != cls and t == cls)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            f1_scores.append(f1)

        macro_f1 = sum(f1_scores) / len(f1_scores)

        print(f"  Accuracy: {acc:.3f}, Macro-F1: {macro_f1:.3f}")

        results.append({
            'name': name,
            'features': feature_subset,
            'accuracy': acc,
            'macro_f1': macro_f1
        })

    # Print summary
    print("\n" + "-" * 60)
    print("SUMMARY OF FEATURE COMBINATIONS")
    print("-" * 60)
    print(f"{'Feature Set':20s}  {'Accuracy':>10s}  {'Macro-F1':>10s}")
    print("-" * 60)

    for result in sorted(results, key=lambda x: x['macro_f1'], reverse=True):
        print(f"{result['name']:20s}  {result['accuracy']:10.3f}  {result['macro_f1']:10.3f}")

    return results


def filter_features(features: Dict[str, Any], keep_keys: List[str]) -> Dict[str, Any]:
    """
    Filter feature dictionary to keep only specified keys.

    Args:
        features: Original feature dictionary
        keep_keys: List of keys to keep

    Returns:
        Filtered feature dictionary
    """
    return {k: v for k, v in features.items() if k in keep_keys}


def main():
    """Run comprehensive evaluation."""
    print("=" * 60)
    print("PRONUNCIATION ERROR CLASSIFIER EVALUATION")
    print("=" * 60)

    # Load training data
    print("\nLoading training data...")
    training_data = prepare_training_data()

    # Cross-validation
    cv_results = cross_validate(training_data, n_folds=10)

    # Feature combination testing
    feature_results = test_feature_combinations(training_data)

    # Train final model with all features
    from train_classifier import train_classifier
    print("\n" + "=" * 60)
    print("TRAINING FINAL MODEL WITH ALL FEATURES")
    print("=" * 60)
    classifier, test_set = train_classifier(training_data)

    # Final evaluation
    eval_results = evaluate_classifier(classifier, test_set)

    return {
        'cross_validation': cv_results,
        'feature_comparison': feature_results,
        'final_evaluation': eval_results,
        'classifier': classifier
    }


if __name__ == '__main__':
    results = main()
