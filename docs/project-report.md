# Project Report: Pronunciation Error Detection Using NLP

## 1. Project Overview
The goal is to build an NLP system that identifies pronunciation errors made by non-native English speakers. By combining speech recognition, text preprocessing, and alignment techniques, the system compares spoken vs. expected text and highlights differences that correspond to pronunciation issues.

## 2. Datasets
### 2.1 L2-ARCTIC (Main Corpus)
- Description: Recordings of non-native English speakers reading CMU ARCTIC passages.
- Data: WAV audio + reference text
- Use: Primary training/analysis dataset
- Source: OpenSLR #96 – L2-ARCTIC
- Notes: Each folder is a speaker; each WAV is a sentence; TXT is the reference.

### 2.2 Common Voice (Accent Diversity)
- Description: Large, crowdsourced dataset with accents metadata.
- Data: Audio + transcripts + accent tags
- Use: Accent diversity and robustness
- Source: Common Voice Dataset

### 2.3 Speech Accent Archive (Validation)
- Description: Recordings from 100+ L1 backgrounds reading the same paragraph; includes IPA transcriptions.
- Use: External validation of model performance
- Source: Speech Accent Archive
- Glossary: IPA is a standardized system to represent sounds independent of spelling.

## 3. Methodology
### 3.1 Data Preprocessing
- ASR: Use Whisper to transcribe audio to text.
- Text Cleaning: lowercase, remove punctuation/noise, normalize contractions/spelling.
- Tokenization & Lemmatization: SpaCy/NTLK; unify forms for alignment.
- Alignment: Compare expected vs. spoken with sequence alignment (Levenshtein) or phoneme distance; mismatches indicate potential pronunciation/fluency errors.
- Optional G2P: Convert to phonemes (IPA/ARPAbet) via g2p-en or phonemizer + analyze sound-level errors.

### Analysis Method
1) Text Classification: Label issue type (grammar vs pronunciation-proxy, etc.)
2) Sequence Labeling: Highlight exact spans with errors
3) Topic Clustering: Discover patterns (verb tense, articles, etc.) for weekly thematic insights
4) Feedback Generation: Turn analytics into personalized recommendations and links to practice

## 4. System Pipeline
Audio → Whisper (ASR) → Clean Text → Tokenize → Lemmatize → Align → Error Detection

## 5. Tools & Libraries
- ASR: OpenAI Whisper / faster-whisper
- Text: SpaCy, NLTK
- Alignment: difflib, python-Levenshtein, phonemizer, jiwer
- G2P: g2p-en, Epitran
- Viz: matplotlib, seaborn, plotly

## 6. Evaluation
- Metrics: Word Error Rate (WER), Phoneme Error Rate (PER), accuracy of pronunciation detection
- Validation: Cross-evaluate on Speech Accent Archive; compare phoneme alignment by accent

## 7. Expected Outcomes
- Cleaned, aligned, and labeled dataset of pronunciation errors
- Visualizations of common error patterns by accent/L1
- A baseline model to detect and categorize pronunciation deviations

## 8. How to explain
Our NLP component isn’t classifying sentiment or topic. It performs phoneme-level classification—detecting if pronunciation patterns align with native English or reflect an accent group. Similar to language ID, but focused on phonetic/acoustic features.

## 9. Future Extensions
- Train deep models for accent classification
- Integrate a feedback module for learners
- Build a web/mobile app for pronunciation practice
