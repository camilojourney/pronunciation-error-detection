# Pronunciation Error Detection (PED)

An NLP system for detecting and analyzing pronunciation errors in non-native English speakers using automatic speech recognition (Whisper), sequence alignment, and error classification.

## 🎓 NLP University Project - Complete & Ready!

**View the complete presentation**: Open `nlp_presentation.html` in your browser

This project demonstrates:
- ✅ Text preprocessing and tokenization
- ✅ Sequence alignment using Levenshtein distance
- ✅ Error detection and classification (insertions, deletions, substitutions)
- ✅ Word Error Rate (WER) computation
- ✅ ASR integration with Whisper
- ✅ Statistical analysis and pattern recognition
- ✅ Data visualization

## 🚀 Quick Start

### View Your Presentation
```bash
# Open the complete HTML presentation
open nlp_presentation.html
```

### Run Interactive Demo
```bash
# Activate environment
source .venv/bin/activate

# Run live demo (great for presentations!)
python scripts/demo_live.py
```

### Regenerate Analysis
```bash
# Generate demo data
python scripts/generate_demo_data.py

# Run complete analysis
python scripts/analyze_demo_data.py

# Re-render presentation
quarto render nlp_presentation.qmd
```

## 📊 Key Results

- **15 speakers** from **10 language backgrounds** analyzed
- **Average WER**: 25.9%
- **23 pronunciation errors** detected and classified
- **Language-specific patterns** identified (e.g., Arabic th→z, Mandarin r→l)

## 📁 Project Structure

```
pronunciation-error-detection/
├── nlp_presentation.html     # 🎯 MAIN PRESENTATION (open this!)
├── nlp_presentation.qmd       # Source for presentation
├── QUICK_START.md             # Detailed guide
├── ped/                       # Core Python package
│   ├── text.py                # Text preprocessing & alignment
│   ├── metrics.py             # WER calculation
│   ├── asr.py                 # Whisper ASR integration
│   ├── errors.py              # Error detection & classification
│   └── pipeline.py            # End-to-end pipeline
├── scripts/
│   ├── generate_demo_data.py  # Create demo dataset
│   ├── analyze_demo_data.py   # Run analysis
│   └── demo_live.py           # Interactive demo
├── data/
│   ├── processed/             # Demo data (15 speakers)
│   └── results/               # Analysis results & statistics
└── pyproject.toml             # UV package configuration
```

## 🛠 Installation

```bash
# Install with UV (already done)
uv sync --all-extras

# Or with pip
pip install -e ".[ml,dev,experiments]"
```

## 💻 Usage Examples

### Text-Based Error Detection
```python
from ped.pipeline import run_text_pipeline

result = run_text_pipeline(
    ref_text="The weather is very nice today",
    hyp_text="The weazer is wery nice today"
)

print(f"WER: {result.wer:.2%}")
print(f"Errors: {result.error_analysis.total_errors}")
```

### Audio-Based (with Whisper ASR)
```python
from ped.pipeline import run_audio_pipeline

result = run_audio_pipeline(
    audio_path="audio.wav",
    reference_text="Expected text",
    model_size="base"
)

print(f"Transcribed: {result.hyp}")
print(f"WER: {result.wer:.2%}")
```

## 📈 What's Included

### 1. Complete Presentation (`nlp_presentation.html`)
- Executive summary
- Methodology with diagrams
- All results and visualizations
- Code examples
- Analysis and conclusions

### 2. Analysis Results (`data/results/`)
- `speaker_results.csv` - Performance metrics per speaker
- `error_details.csv` - Detailed error information
- `aggregate_stats.json` - Overall statistics
- `substitution_patterns.csv` - Common error patterns
- `errors_by_language.csv` - Language-specific analysis

### 3. Working Code (`ped/` package)
- Fully implemented ASR with Whisper
- Complete error detection system
- Evaluation metrics (WER)
- Visualization tools

## 🎯 For Your Presentation

1. Open `nlp_presentation.html`
2. Walk through methodology and results
3. Run `python scripts/demo_live.py` for live demonstration
4. Show the modular code architecture
5. Discuss language-specific error patterns

See `QUICK_START.md` for detailed presentation tips!

## 🔬 NLP Techniques Demonstrated

1. **Text Preprocessing** - Tokenization, normalization, cleaning
2. **Sequence Alignment** - Edit distance, dynamic programming
3. **Error Classification** - Systematic linguistic error categorization
4. **Evaluation Metrics** - WER, error rate computation
5. **Pattern Recognition** - Statistical analysis of error patterns
6. **ASR Integration** - Speech-to-text with Whisper
7. **Data Visualization** - Charts, tables, statistical plots

## 📚 References

- **Whisper**: Robust Speech Recognition via Large-Scale Weak Supervision
- **Edit Distance**: Levenshtein distance algorithm
- **WER**: Standard speech recognition evaluation metric

## 📄 License

MIT
