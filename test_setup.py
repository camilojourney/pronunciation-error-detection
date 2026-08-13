#!/usr/bin/env python3
"""
Quick test script to verify project setup.

This script:
1. Checks if all dependencies are installed
2. Tests parsing a few TextGrid files
3. Extracts features from sample errors
4. Shows manual severity labeling in action
5. Trains a small classifier on subset of data

Run this BEFORE running the full pipeline to ensure everything works.
"""

def check_dependencies():
    """Check if all required packages are installed."""
    print("=" * 60)
    print("CHECKING DEPENDENCIES")
    print("=" * 60)

    required = {
        'nltk': 'Natural Language Toolkit',
        'textgrid': 'TextGrid parser',
        'pandas': 'Data analysis',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization'
    }

    missing = []

    for package, description in required.items():
        try:
            __import__(package)
            print(f"✓ {package:15s} - {description}")
        except ImportError:
            print(f"✗ {package:15s} - {description} (MISSING)")
            missing.append(package)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: uv sync")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True


def test_parsing():
    """Test TextGrid parsing on a few files."""
    print("\n" + "=" * 60)
    print("TESTING TEXTGRID PARSING")
    print("=" * 60)

    try:
        from parse_annotations import process_all_annotations

        print("Parsing first few annotation files...")
        errors = process_all_annotations('l2arctic_release_v5')

        print(f"✓ Successfully parsed {len(errors)} errors from 10 files")

        # Show sample errors
        if errors:
            print("\nSample errors:")
            for i, error in enumerate(errors[:3], 1):
                error_type = error['error_type']
                if error_type == 's':
                    print(f"  {i}. {error['expected']} → {error['actual']} (substitution)")
                elif error_type == 'd':
                    print(f"  {i}. {error['expected']} → ∅ (deletion)")
                else:
                    print(f"  {i}. ∅ → {error['actual']} (addition)")

        return True, errors[:100]  # Return first 100 for testing

    except Exception as e:
        print(f"❌ Error parsing TextGrids: {e}")
        return False, []


def test_feature_extraction(errors):
    """Test feature extraction."""
    print("\n" + "=" * 60)
    print("TESTING FEATURE EXTRACTION")
    print("=" * 60)

    try:
        from feature_engineering import extract_features

        if not errors:
            print("⚠ No errors to extract features from")
            return False

        # Extract features from first error
        features = extract_features(errors[0])

        print(f"✓ Successfully extracted {len(features)} features")
        print("\nSample features:")
        for key, value in list(features.items())[:5]:
            print(f"  {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Error extracting features: {e}")
        return False


def test_severity_labeling(errors):
    """Test manual severity labeling."""
    print("\n" + "=" * 60)
    print("TESTING SEVERITY LABELING")
    print("=" * 60)

    try:
        from train_classifier import label_error_severity

        if not errors:
            print("⚠ No errors to label")
            return False

        # Label first few errors
        print("\nSample severity labels:")
        for error in errors[:5]:
            severity = label_error_severity(error)
            error_type = error['error_type']

            if error_type == 's':
                desc = f"{error['expected']} → {error['actual']}"
            elif error_type == 'd':
                desc = f"{error['expected']} → ∅"
            else:
                desc = f"∅ → {error['actual']}"

            print(f"  {desc:15s}  →  {severity}")

        print(f"\n✓ Successfully labeled {len(errors)} errors")
        return True

    except Exception as e:
        print(f"❌ Error labeling severity: {e}")
        return False


def test_small_classifier(errors):
    """Train a small classifier on subset of data."""
    print("\n" + "=" * 60)
    print("TESTING CLASSIFIER TRAINING (SMALL SAMPLE)")
    print("=" * 60)

    try:
        from feature_engineering import extract_features
        from train_classifier import label_error_severity
        from nltk.classify import NaiveBayesClassifier, accuracy

        if len(errors) < 50:
            print("⚠ Not enough errors for testing classifier")
            return False

        print(f"Using {len(errors)} errors for quick test...")

        # Prepare training data
        training_data = []
        for error in errors:
            features = extract_features(error)
            severity = label_error_severity(error)
            training_data.append((features, severity))

        print(f"✓ Prepared {len(training_data)} training examples")

        # Train classifier
        print("Training Naive Bayes classifier...")
        classifier = NaiveBayesClassifier.train(training_data[:80])

        print("✓ Training complete!")

        # Test on remaining data
        if len(training_data) > 80:
            test_set = training_data[80:]
            acc = accuracy(classifier, test_set)
            print(f"✓ Test accuracy on {len(test_set)} examples: {acc:.3f}")

        # Show most informative features
        print("\nTop 5 most informative features:")
        classifier.show_most_informative_features(5)

        return True

    except Exception as e:
        print(f"❌ Error training classifier: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" PRONUNCIATION ERROR CLASSIFICATION - QUICK TEST")
    print("=" * 70)

    # Test 1: Dependencies
    if not check_dependencies():
        print("\n❌ FAILED: Install missing dependencies first")
        return False

    # Test 2: Parsing
    success, errors = test_parsing()
    if not success:
        print("\n❌ FAILED: Cannot parse TextGrid files")
        print("Make sure l2arctic_release_v5/ directory exists with annotation files")
        return False

    # Test 3: Feature extraction
    if not test_feature_extraction(errors):
        print("\n❌ FAILED: Feature extraction error")
        return False

    # Test 4: Severity labeling
    if not test_severity_labeling(errors):
        print("\n❌ FAILED: Severity labeling error")
        return False

    # Test 5: Classifier training
    if not test_small_classifier(errors):
        print("\n❌ FAILED: Classifier training error")
        return False

    # All tests passed!
    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED!")
    print("=" * 70)
    print("\nYou can now run the full pipeline:")
    print("  python train_classifier.py    # Train on full dataset")
    print("  python evaluate_model.py       # Evaluate with cross-validation")
    print("  quarto render nlp_presentation_final.qmd  # Generate report")
    print()

    return True


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
