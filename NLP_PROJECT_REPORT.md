# Project Report: Pronunciation Error Detection Using NLP

## 1. Project Overview
The goal of this project is to build an NLP system that identifies pronunciation errors made by non-native English speakers. By combining speech recognition, text preprocessing, and alignment techniques, the system aims to automatically compare spoken vs. expected text and highlight differences that correspond to pronunciation issues.

---

## 2. Datasets

### 2.1 L2-ARCTIC (Main Corpus)
- **Description**: Contains recordings of non-native English speakers reading standard English passages (based on CMU ARCTIC).
- **Data type**: Audio (.wav) + reference text (.txt)
- **Use case**: Primary training and analysis dataset
- **Download**: [OpenSLR #96 – L2-ARCTIC](http://www.openslr.org/96/)
- **Purpose in project**: Provides aligned pairs of spoken and expected text for phoneme-level comparison.
  - Each folder = one speaker
  - Each .wav = a single sentence from a CMU-ARCTIC passage
  - Each .txt = the reference text (what they were reading)

### 2.2 Common Voice (Accent Diversity)
- **Description**: Large-scale, crowdsourced dataset from Mozilla with thousands of speakers across multiple accents.
- **Data type**: Audio + transcripts + accent metadata
- **Use case**: Accent diversity and model robustness
- **Download**: [Common Voice Dataset](https://commonvoice.mozilla.org/en/datasets)

### 2.3 Speech Accent Archive (Validation)
- **Description**: Recordings of speakers from 100+ language backgrounds reading the same paragraph, with IPA transcriptions.
  - **IPA transcription**: Writing down how words are pronounced using the International Phonetic Alphabet (IPA) — a standardized system of symbols that represent the sounds of spoken language, not the spelling.
- **Use case**: External validation of model performance
- **Access**: [Speech Accent Archive](https://accent.gmu.edu/)

---

## 3. Methodology

### 3.1 Data Preprocessing Pipeline

1. **Automatic Speech Recognition (ASR)**:
   - Use **Whisper** (OpenAI's ASR model) to transcribe raw audio into text.

2. **Text Cleaning**:
   - Lowercase text
   - Remove punctuation, noise, and non-verbal markers
   - Normalize contractions and spelling variants

3. **Tokenization & Lemmatization**:
   - Split sentences into tokens using SpaCy or NLTK
   - Lemmatize to obtain base word forms for consistent alignment

4. **Alignment**:
   - Compare expected text vs. spoken text using sequence alignment (Levenshtein or phoneme-based distance)
   - Detect mismatches as pronunciation or fluency errors

5. **(Optional) Grapheme-to-Phoneme (G2P) Conversion**:
   - Convert words to phonetic form (IPA or ARPAbet)
   - Align spoken phonemes to expected phonemes to analyze specific sound-level errors

### 3.2 Analysis Methods

#### Step 1: Text Classification
- **Purpose**: Identify what type of issue the input contains by categorizing and labeling.
- **Categories**: Grammar error, Pronunciation proxy, etc.

#### Step 2: Sequence Labeling
- **Purpose**: Detect and highlight where in the text the error occurs.

#### Step 3: Topic Clustering
- **Purpose**: Discover patterns and recurring mistake types.
- Groups similar errors (e.g., verb tense, article misuse, etc.).
- Enables weekly thematic analysis of learner performance.

#### Step 4: Feedback Generation
- **Purpose**: Convert analytical results into personalized recommendations.
- Summarizes frequent issues and suggests practice tips.
- Links error patterns to learning resources or exercises.
- **Outcome**: Continuous, data-driven learning improvement loop.

---

## 4. System Pipeline

```
Audio Input → Whisper (ASR) → Clean Text → Tokenization → Lemmatization → Alignment → Error Detection
```

This pipeline transforms raw audio data into analyzable linguistic units, enabling both textual and phonetic comparison.

---

## 5. Tools & Libraries

| Library / Tool | Description |
|----------------|-------------|
| **ASR** | OpenAI Whisper | Converts speech to text |
| **Text Processing** | SpaCy, NLTK | Tokenization, lemmatization |
| **Alignment** | difflib, Jiwer, Phonemizer | Word/phoneme-level comparison |
| **G2P Conversion** | g2p-en, Epitran | Text-to-phoneme mapping |
| **Visualization** | matplotlib, seaborn, plotly | Error frequency and trend visualization |

---

## 6. Evaluation

### Metrics:
- **Word Error Rate (WER)**
- **Phoneme Error Rate (PER)**
- **Accuracy of pronunciation detection**

### Validation:
- Use the Speech Accent Archive for cross-evaluation.
- Compare phoneme alignment accuracy across accents.

---

## 7. Expected Outcomes

1. A cleaned, aligned, and labeled dataset of pronunciation errors.
2. Visualizations showing common error patterns by accent or language background.
3. A baseline model capable of detecting and categorizing pronunciation deviations.

---

## 8. How to Explain This Project

**"Our NLP component isn't classifying text sentiment or topic. Instead, it performs a phoneme-level classification — detecting if the pronunciation pattern aligns with native English or reflects a specific accent group. This is similar in logic to language identification tasks, but applied to phonetic and acoustic features instead of words."**

---

## 9. Future Extensions (Not in Current Scope)

- Train a deep learning model for automatic accent classification.
- Integrate with a feedback module that provides corrective suggestions for language learners.
- Build a web dashboard or mobile app for pronunciation practice.
