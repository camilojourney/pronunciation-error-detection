# Pronunciation Error Severity Classification

**NLP Term Project: Supervised Learning for L2 Speech Assessment**

## 📊 Main Presentation

**Primary file:** `nlp_presentation_final.qmd`

Run this Quarto notebook to generate the complete project report with all code, analysis, and visualizations.

## 🚀 Quick Start

```bash
# 1. Set up environment
uv sync

# 2. Download NLTK data
uv run python -c "import nltk; nltk.download('punkt')"

# 3. Generate presentation
quarto render nlp_presentation_final.qmd

# 4. Open the presentation
open nlp_presentation_final.html
```

## 📁 Project Files

```
Essential Files:
├── nlp_presentation_final.qmd    ⭐ Main presentation
├── parse_annotations.py          → Extract errors from TextGrid files
├── phoneme_properties.py         → Linguistic knowledge base
├── feature_engineering.py        → Feature extraction
├── train_classifier.py           → Train Naive Bayes classifier
└── evaluate_model.py             → Model evaluation

Data:
└── l2arctic_release_v5/          → L2-ARCTIC corpus (already present)
```

## 📖 Project Overview

- **Dataset**: L2-ARCTIC corpus (18,610 phoneme errors, 24 speakers, 6 languages)
- **Task**: Classify error severity (HIGH/MEDIUM/LOW)
- **Method**: Naive Bayes classifier (Chapter 6)
- **Features**: Phoneme properties, minimal pairs, L1 patterns
- **Evaluation**: 10-fold CV, precision/recall/F1, hyperparameter tuning

## 🎯 Expected Results

- Accuracy: ~70-75%
- Macro-F1: ~69-71%
- HIGH severity errors detected with high recall

## 📝 Citation

> Zhao, G., et al. (2018). L2-ARCTIC: A non-native English speech corpus. INTERSPEECH 2018.
