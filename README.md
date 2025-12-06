# Phoneme-Level Pronunciation Error Analysis

End-to-end pipeline that extracts, compares, and visualizes phonemes from multiple sources to detect pronunciation errors in non-native English speech (L2-ARCTIC).

## Quick Start

```bash
# Install dependencies (uv recommended)
uv sync

# Process L2-ARCTIC sample and build dashboard
python process_l2arctic.py
open data/results/pronunciation_dashboard.html

# Render the presentation (Quarto)
quarto render nlp_presentation.qmd
open nlp_presentation.html
```

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd pronunciation-error-detection

# Core dependencies
uv sync
# or
pip install -e .

# With ML + experiments extras (Whisper, Wav2Vec2, notebooks, viz)
pip install -e ".[ml,experiments]"

# Optional: Montreal Forced Aligner (for MFA-based sources)
conda create -n mfa -c conda-forge montreal-forced-aligner
conda activate mfa
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
```

## Dataset

- **L2-ARCTIC Release v5** (OpenSLR.org/96): 24 non-native English speakers, ~1,130 utterances each (~27k audio files)
- CMU ARCTIC phonetically balanced sentences with transcripts and TextGrids
- Located at `l2arctic_release_v5/` (files discovered directly from dataset directory)

## Architecture

```
phoneme_sources.py   # DATA: unified access to 5 phoneme sources
    ↓
analysis_utils.py    # PROCESSING: alignment, metrics, error detection
    ↓
generate_dashboard.py# VIEW: interactive HTML dashboard
    ↓
process_l2arctic.py  # CONTROLLER: orchestrates pipeline
```

## Project Structure

```
pronunciation-error-detection/
├── phoneme_sources.py         # Unified API for all 5 phoneme sources
├── analysis_utils.py          # Core NLP functions
├── generate_dashboard.py      # Dashboard generation
├── process_l2arctic.py        # Main processing script
├── nlp_presentation.qmd       # Quarto presentation
├── legacy_code/               # Archived redundant files
│   ├── pronunciation_pipeline.py
│   ├── forced_alignment.py
│   └── wav2vec_phoneme.py
├── data/
│   └── results/
│       ├── pronunciation_dashboard.html
│       ├── speaker_results.csv
│       ├── language_statistics.csv
│       └── speaker_statistics.csv
└── pyproject.toml             # Dependencies and extras
```

## The Three Phoneme Sources

Our system compares pronunciation across 3 complementary sources:

1. **📚 Dictionary (CMU/G2P)** - Expected pronunciation from reference text
2. **🔬 Whisper Large-v3 + MFA** - Words from Whisper Large-v3, phonemes from Montreal Forced Aligner
3. **🎧 Whisper Large-v3 + Wav2Vec2** - Words from Whisper Large-v3, phonemes from direct audio recognition

**Why these 3?** Dictionary gives us the "expected" pronunciation, while the two Whisper Large-v3 hybrid methods provide actual pronunciation from different phoneme extraction approaches (alignment-based vs. direct recognition), allowing us to identify pronunciation errors with high confidence.

## Usage Examples

### Text-Based Error Detection

```python
from analysis_utils import run_text_pipeline

result = run_text_pipeline(
    ref_text="The weather is very nice today",
    hyp_text="The weazer is wery nice today"
)

print(f"WER: {result.wer:.2%}")
print(f"Errors: {result.error_analysis.total_errors}")
```

### Audio-Based Error Detection (Whisper ASR)

```python
from analysis_utils import run_audio_pipeline

result = run_audio_pipeline(
    audio_path="l2arctic_release_v5/ABA/wav/arctic_a0001.wav",
    ref_text="Author of the danger trail Philip Steels etc",
    model_size="base"
)

print(f"Transcribed: {result.hyp}")
print(f"WER: {result.wer:.2%}")
```

### Get All Phoneme Sources for One File

```python
from phoneme_sources import get_all_phoneme_sources, format_comparison_summary

sources = get_all_phoneme_sources(
    audio_path="audio.wav",
    expected_text="the ship sails"  # Optional; Whisper used if None
)

print(format_comparison_summary(sources))
```

### Generate a Custom Dashboard

```python
from generate_dashboard import generate_html_dashboard

generate_html_dashboard(
    results_csv="data/results/speaker_results.csv",
    output_html="data/results/pronunciation_dashboard.html"
)
```

## Dashboard Highlights (`data/results/pronunciation_dashboard.html`)

- Side-by-side view of all 5 phoneme sources with IPA formatting
- Built-in audio player per utterance
- Filters: native language, error severity, text search
- Error coloring: green (correct), red (high impact), orange (medium), yellow (low)

## API Reference

```python
# Individual sources
from phoneme_sources import (
    get_dictionary_phonemes,
    get_mfa_phonemes,
    get_wav2vec_phonemes,
    get_whisper_mfa_phonemes,
    get_whisper_wav2vec_phonemes,
    get_whisper_large_mfa_phonemes,  # NEW: Whisper Large-v3 + MFA
    arpabet_to_ipa,
    format_ipa,
    ARPABET_TO_IPA,
)
```

`get_all_phoneme_sources` returns a dict of `PhonemeSource` objects:

```python
@dataclass
class PhonemeSource:
    name: str
    description: str
    success: bool
    words: Optional[List[PhonemeWord]]
    continuous_ipa: Optional[str]
    error: Optional[str]
    metadata: Optional[Dict]
```

`PhonemeWord`:

```python
@dataclass
class PhonemeWord:
    word: str
    phonemes: List[str]
    ipa: str  # e.g., "/ʃ ɪ p/"
```

## Processing Pipeline (Key Functions)

```python
from analysis_utils import (
    clean_text,
    tokenize,
    align_tokens,
    wer,
    run_audio_pipeline,
    analyze_intelligibility,
)

clean = clean_text("The ship sails!")        # → "the ship sails"
tokens = tokenize("the ship sails")          # → ["the", "ship", "sails"]
error_rate = wer(tokenize("the ship sails"), tokenize("the chip sales"))
result = run_audio_pipeline("audio.wav", "the ship sails")
intel = analyze_intelligibility(result.error_analysis)
```

## Performance and Troubleshooting

- Models are cached after first load (Whisper, Wav2Vec2 in `~/.cache`)
- Skip heavy sources by calling `get_dictionary_phonemes` directly when speed matters
- MFA optional; if unavailable, MFA sources are skipped gracefully
- First Wav2Vec2 call downloads the model (~1GB); ensure disk space
- Install `soundfile` or `torchaudio` if audio loading fails

## Roadmap and Current Status

- ✅ Dataset ready (L2-ARCTIC downloaded, direct file discovery)
- ✅ Core NLP utilities (`analysis_utils.py`, alignment, WER/CER, Whisper integration)
- ✅ Evaluation framework (`ground_truth_parser.py`, `evaluation_metrics.py`)
- ✅ Pipeline scripts (`process_l2arctic.py`, `run_evaluation.py`)
- ✅ Interactive dashboards (pronunciation analysis, evaluation results)
- ✅ Updated `nlp_presentation.qmd` with evaluation framework
- 🔜 Run full evaluation on all 3,599 annotated files
- 🔜 Hyperparameter tuning (Whisper model size, thresholds)
- 🔜 Phoneme-level evaluation (integrate MFA/Wav2Vec2)
- 🔜 Optional: add phoneme-level PER metrics and pattern clustering

## Deliverables for the NLP Project

- Code: pipeline and APIs in `phoneme_sources.py`, `analysis_utils.py`, `process_l2arctic.py`
- Data artifacts: `data/results/speaker_results.csv`, `language_statistics.csv`, dashboards
- Metrics: WER (and optional PER), error distributions
- Presentation: `nlp_presentation.qmd` → `nlp_presentation.html`
- Demo: dashboard and sample audio runs

## References

- Whisper ASR: Radford, A., et al. (2022). "Robust Speech Recognition via Large-Scale Weak Supervision"
- Edit Distance: Levenshtein distance algorithm
- L2-ARCTIC Corpus: https://psi.engr.tamu.edu/l2-arctic-corpus/
- Montreal Forced Aligner: https://montreal-forced-aligner.readthedocs.io/
- Wav2Vec2: https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft
- CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict

## License

MIT
