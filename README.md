# Pronunciation Error Detection in Non-Native English Speakers

An NLP system for detecting and analyzing pronunciation errors using automatic speech recognition (Whisper), sequence alignment, and error classification.

## Quick Start

**View the presentation:**

```bash
# Render the presentation
quarto render nlp_presentation.qmd

# Open the HTML output
open nlp_presentation.html
```

## Project Overview

This project implements an end-to-end pronunciation error detection pipeline:

1. **ASR Transcription** - Converts speech to text using Whisper
2. **Sequence Alignment** - Compares transcribed text with reference using Levenshtein distance
3. **Error Detection** - Classifies errors (substitutions, deletions, insertions)
4. **Statistical Analysis** - Calculates WER and identifies error patterns

## Project Structure

```text
pronunciation-error-detection/
├── nlp_presentation.qmd      # Main presentation and analysis
├── analysis_utils.py          # NLP utility functions
├── data/
│   ├── processed/
│   │   └── l2arctic_manifest.json
│   └── results/               # Generated after processing
├── l2arctic_release_v5/       # L2-ARCTIC dataset (27K audio files)
├── NLP_FOCUS_ROADMAP.md       # Project roadmap and workflow
└── README.md                  # This file
```

## Dataset

**L2-ARCTIC Speech Corpus:**

- 24 non-native English speakers
- 6 language backgrounds (Arabic, Mandarin, Hindi, Korean, Spanish, Vietnamese)
- ~1,130 utterances per speaker (27,120 total audio files)
- CMU ARCTIC phonetically balanced sentences
- Source: OpenSLR.org/96 (University of Edinburgh)

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pronunciation-error-detection

# Install dependencies with uv (recommended)
uv sync

# Or with pip
pip install faster-whisper pandas matplotlib seaborn quarto
```

## Usage

### 1. Text-Based Error Detection

```python
from analysis_utils import run_text_pipeline

# Compare reference and hypothesis texts
result = run_text_pipeline(
    ref_text="The weather is very nice today",
    hyp_text="The weazer is wery nice today"
)

print(f"WER: {result.wer:.2%}")
print(f"Errors: {result.error_analysis.total_errors}")
```

### 2. Audio-Based Error Detection (with Whisper ASR)

```python
from analysis_utils import run_audio_pipeline

# Process audio file
result = run_audio_pipeline(
    audio_path="l2arctic_release_v5/ABA/wav/arctic_a0001.wav",
    ref_text="Author of the danger trail Philip Steels etc",
    model_size="base"
)

print(f"Transcribed: {result.hyp}")
print(f"WER: {result.wer:.2%}")
```

### 3. Render Presentation

```bash
quarto render nlp_presentation.qmd
```

## NLP Techniques Demonstrated

- **Text Preprocessing** - Tokenization, normalization, cleaning
- **Sequence Alignment** - Edit distance, dynamic programming
- **Error Classification** - Linguistic error categorization
- **Evaluation Metrics** - WER computation
- **Pattern Recognition** - Statistical analysis of error patterns
- **ASR Integration** - Speech-to-text with Whisper
- **Data Visualization** - Charts and statistical plots

## Key Files

- **[nlp_presentation.qmd](nlp_presentation.qmd)** - Main presentation (methodology, analysis, results)
- **[analysis_utils.py](analysis_utils.py)** - Reusable NLP functions (ASR, alignment, metrics)
- **[NLP_FOCUS_ROADMAP.md](NLP_FOCUS_ROADMAP.md)** - Project roadmap and step-by-step workflow

## References

- **Whisper ASR**: Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision"
- **Edit Distance**: Levenshtein distance algorithm
- **WER**: Standard speech recognition evaluation metric
- **L2-ARCTIC**: Non-native English speech corpus (OpenSLR.org/96)

## License

MIT
