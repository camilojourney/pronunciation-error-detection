"""
Train Naive Bayes classifier for phoneme error severity.

This module implements the supervised learning approach (Chapter 6)
for classifying phoneme errors as HIGH, MEDIUM, or LOW severity.
"""

import random
from typing import List, Tuple, Dict, Any
import nltk
from nltk.classify import NaiveBayesClassifier

from parse_annotations import process_all_annotations
from feature_engineering import extract_features
from phoneme_properties import is_minimal_pair, is_noticeable_error


def label_error_severity(error: Dict[str, Any]) -> str:
    """
    Manually label error severity based on linguistic knowledge.

    Severity levels:
    - HIGH: Errors that create minimal pairs or significantly impair comprehension
    - MEDIUM: Noticeable errors that affect naturalness but not comprehension
    - LOW: Minor accent features that don't impair communication

    Args:
        error: Error dictionary with expected, actual, error_type, etc.

    Returns:
        'HIGH', 'MEDIUM', or 'LOW'
    """
    error_type = error['error_type']

    # DELETIONS: Usually high severity if consonants
    if error_type == 'd':
        deleted = error['expected']

        # Final consonant deletion is very noticeable
        is_final = error.get('next_phoneme') == 'sil' or error.get('next_phoneme') is None
        if is_final:
            return 'HIGH'

        # Consonant deletions are generally high severity
        from phoneme_properties import get_phoneme_properties
        props = get_phoneme_properties(deleted)
        if props.get('type') != 'vowel':
            return 'HIGH'
        else:
            return 'MEDIUM'

    # ADDITIONS: Usually lower severity
    if error_type == 'a':
        from phoneme_properties import get_phoneme_properties
        added = error['actual']
        props = get_phoneme_properties(added)

        # Vowel additions are less severe
        if props.get('type') == 'vowel':
            return 'LOW'
        else:
            return 'MEDIUM'

    # SUBSTITUTIONS: Vary by minimal pair status
    if error_type == 's':
        expected = error['expected']
        actual = error['actual']

        # Minimal pairs are HIGH severity (change word meaning)
        if is_minimal_pair(expected, actual):
            return 'HIGH'

        # Noticeable errors are MEDIUM severity
        if is_noticeable_error(expected, actual):
            return 'MEDIUM'

        # Check if same type of sound (vowel-to-vowel, consonant-to-consonant)
        from phoneme_properties import get_phoneme_properties
        exp_props = get_phoneme_properties(expected)
        act_props = get_phoneme_properties(actual)

        # Cross-type substitutions (vowel→consonant) are more severe
        if exp_props.get('type') != act_props.get('type'):
            return 'HIGH'

        # Same place but different voicing is usually MEDIUM
        if (exp_props.get('place') == act_props.get('place') and
            exp_props.get('voicing') != act_props.get('voicing')):
            return 'MEDIUM'

        # Same type and similar features is LOW severity
        if (exp_props.get('type') == act_props.get('type') and
            exp_props.get('place') == act_props.get('place')):
            return 'LOW'

        # Default for substitutions
        return 'MEDIUM'

    # Default fallback
    return 'MEDIUM'


def prepare_training_data(
    annotations_path: str = 'l2arctic_release_v5',
    max_errors: int = None,
    random_seed: int = 42
) -> List[Tuple[Dict, str]]:
    """
    Prepare labeled training data from L2-ARCTIC annotations.

    Args:
        annotations_path: Path to L2-ARCTIC corpus
        max_errors: Maximum number of errors to use (for testing)
        random_seed: Random seed for reproducibility

    Returns:
        List of (features, label) tuples
    """
    print("Parsing L2-ARCTIC annotations...")
    errors = process_all_annotations(annotations_path)
    print(f"Found {len(errors)} phoneme errors")

    # Sample if requested
    if max_errors and max_errors < len(errors):
        random.seed(random_seed)
        errors = random.sample(errors, max_errors)
        print(f"Sampled {max_errors} errors for training")

    print("Extracting features and labeling severity...")
    training_data = []

    for error in errors:
        # Extract features
        features = extract_features(error)

        # Label severity
        severity = label_error_severity(error)

        # Add to training data
        training_data.append((features, severity))

    return training_data


def train_classifier(
    training_data: List[Tuple[Dict, str]],
    test_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[NaiveBayesClassifier, List[Tuple[Dict, str]]]:
    """
    Train Naive Bayes classifier on labeled data.

    Args:
        training_data: List of (features, label) tuples
        test_split: Fraction of data to hold out for testing
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (trained_classifier, test_data)
    """
    # Shuffle data
    random.seed(random_seed)
    random.shuffle(training_data)

    # Split into train/test
    split_point = int(len(training_data) * (1 - test_split))
    train_set = training_data[:split_point]
    test_set = training_data[split_point:]

    print(f"Training set: {len(train_set)} examples")
    print(f"Test set: {len(test_set)} examples")

    # Count class distribution
    print("\nClass distribution in training set:")
    class_counts = {}
    for _, label in train_set:
        class_counts[label] = class_counts.get(label, 0) + 1
    for label, count in sorted(class_counts.items()):
        print(f"  {label}: {count} ({count/len(train_set)*100:.1f}%)")

    # Train classifier
    print("\nTraining Naive Bayes classifier...")
    classifier = NaiveBayesClassifier.train(train_set)

    # Show most informative features
    print("\nMost informative features:")
    classifier.show_most_informative_features(20)

    return classifier, test_set


def save_classifier(classifier: NaiveBayesClassifier, filename: str = 'classifier.pkl'):
    """
    Save trained classifier to disk.

    Args:
        classifier: Trained classifier
        filename: Output filename
    """
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"\nClassifier saved to {filename}")


def load_classifier(filename: str = 'classifier.pkl') -> NaiveBayesClassifier:
    """
    Load trained classifier from disk.

    Args:
        filename: Input filename

    Returns:
        Trained classifier
    """
    import pickle
    with open(filename, 'rb') as f:
        classifier = pickle.load(f)
    return classifier


def main():
    """Train classifier on full L2-ARCTIC dataset."""
    print("=" * 60)
    print("PRONUNCIATION ERROR SEVERITY CLASSIFICATION")
    print("=" * 60)

    # Prepare training data
    training_data = prepare_training_data()

    # Train classifier
    classifier, test_set = train_classifier(training_data)

    # Save classifier
    save_classifier(classifier)

    # Quick accuracy check
    from nltk.classify import accuracy
    test_accuracy = accuracy(classifier, test_set)
    print(f"\nTest accuracy: {test_accuracy:.3f}")

    return classifier, test_set


if __name__ == '__main__':
    classifier, test_set = main()
