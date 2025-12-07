# Pronunciation Error Severity Classification

**NLP Class Term Project**
**Supervised Learning for L2 Speech Assessment**

## Overview

This project applies supervised machine learning (Chapter 6 techniques) to classify the severity of pronunciation errors made by non-native English speakers. Using the L2-ARCTIC corpus with 18,610 annotated phoneme errors across 24 speakers from 6 native languages, we train a Naive Bayes classifier to predict whether each error is HIGH, MEDIUM, or LOW severity.

## Problem Statement

Second language (L2) learners make pronunciation errors that vary in their impact on communication:
- **HIGH severity**: Errors that change word meaning (minimal pairs like "think" → "sink")
- **MEDIUM severity**: Noticeable errors that don't impair comprehension
- **LOW severity**: Accent features that are acceptable

Manual assessment by teachers is time-consuming. This project automates severity classification using NLP techniques.

## Dataset

**L2-ARCTIC Corpus**: Non-native English speech dataset
- 24 speakers from 6 native languages (Arabic, Chinese, Hindi, Korean, Spanish, Vietnamese)
- 3,599 manually annotated audio files
- 18,610 phoneme-level error annotations
  - 14,098 substitutions (e.g., TH → S)
  - 3,420 deletions (e.g., missing final consonants)
  - 1,092 additions (e.g., extra vowels)

Annotation format: `CPL,PPL,type` where:
- CPL = Expected phoneme (canonical pronunciation)
- PPL = Actual phoneme (what speaker produced)
- type ∈ {s, d, a} (substitution, deletion, addition)

## Methods

### 1. Feature Engineering
Extracts linguistic properties from each error:
- **Phoneme properties**: Type (vowel/consonant), place of articulation, voicing
- **Similarity features**: Same type, same place, same voicing
- **Linguistic patterns**: Minimal pair detection, common L1-specific errors
- **Context features**: Previous/next phoneme, position in word
- **Speaker features**: Native language background

### 2. Manual Severity Labeling
Domain expertise applied to create training labels:
- **HIGH**: Minimal pairs (TH→S: "think"→"sink"), cross-type substitutions, deletions
- **MEDIUM**: Noticeable but non-critical errors, voicing changes, additions
- **LOW**: Accent features with minimal impact

### 3. Classification Model
**Naive Bayes Classifier** (NLTK implementation, Chapter 6)
- Probabilistic classifier well-suited for text/linguistic data
- Handles high-dimensional feature spaces
- Provides interpretable feature importances

### 4. Evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-score per class
- **Cross-validation**: 10-fold CV for robust performance estimates
- **Hyperparameter tuning**: Tests 8 different feature combinations

## Project Structure

```
pronunciation-error-detection/
├── parse_annotations.py       # Extract errors from TextGrid files
├── phoneme_properties.py      # Linguistic knowledge base
├── feature_engineering.py     # Feature extraction
├── train_classifier.py        # Train Naive Bayes model
├── evaluate_model.py          # Model evaluation & tuning
├── nlp_presentation.qmd       # Quarto presentation
├── requirements.txt           # Python dependencies
└── l2arctic_release_v5/       # Dataset (not in repo)
    └── */annotation/*.TextGrid
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Download L2-ARCTIC corpus
# https://psi.engr.tamu.edu/l2-arctic-corpus/

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"
```

## Usage

### Train the classifier
```bash
python train_classifier.py
```

This will:
1. Parse all 3,599 annotated files
2. Extract features for 18,610 errors
3. Label severity using linguistic rules
4. Train Naive Bayes classifier
5. Save model to `classifier.pkl`

### Evaluate performance
```bash
python evaluate_model.py
```

This will:
1. Perform 10-fold cross-validation
2. Test 8 feature combinations
3. Generate confusion matrix
4. Report precision/recall/F1 per class

### Generate presentation
```bash
quarto render nlp_presentation.qmd
```

## Expected Results

Based on pilot experiments:
- **Overall accuracy**: ~70-75%
- **HIGH severity**: High recall (catch most critical errors)
- **LOW severity**: High precision (few false alarms)
- **MEDIUM severity**: Most challenging class (boundary cases)

Most informative features:
- `is_minimal_pair`: Strong predictor of HIGH severity
- `error_type='d'`: Deletions are usually HIGH
- `same_type=False`: Cross-type substitutions are HIGH
- `is_l1_pattern`: Common L1 errors tend to be MEDIUM

## NLP Techniques Demonstrated

1. **Supervised Classification** (Chapter 6)
   - Naive Bayes with NLTK
   - Feature engineering from linguistic knowledge
   - Training/test split with cross-validation

2. **Feature Engineering**
   - Domain knowledge representation
   - Pattern detection and extraction
   - Contextual features

3. **Model Evaluation**
   - Precision/Recall/F1 metrics
   - Confusion matrix analysis
   - Hyperparameter tuning (feature selection)

## Future Work

- Test other classifiers (MaxEnt, Decision Trees)
- Add acoustic features from audio
- Per-language models for L1-specific patterns
- Real-time feedback system for learners
- Extend to other L2 languages

## References

- Zhao, G., et al. (2018). L2-ARCTIC: A Non-native English Speech Corpus. INTERSPEECH.
- Jurafsky, D. & Martin, J. H. (2023). Speech and Language Processing (3rd ed.). Chapter 6: Naive Bayes.
- NLTK: Natural Language Toolkit. https://www.nltk.org/

## License

This project uses the L2-ARCTIC corpus (see `l2arctic_release_v5/LICENSE`).
Project code is available for educational use.

## Author

NLP Class Term Project
[Your Name]
[Date]
