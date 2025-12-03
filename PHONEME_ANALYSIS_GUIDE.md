# Phoneme-Level Pronunciation Error Analysis Guide

## Overview

This system provides **phoneme-level pronunciation error analysis** with an **interactive web dashboard** for visualizing errors. It combines:

- **Word-level error detection** (substitutions, deletions, insertions)
- **Phoneme-level alignment** using CMU Pronunciation Dictionary
- **Intelligibility classification** (HIGH/MEDIUM/LOW impact)
- **IPA (International Phonetic Alphabet) transcription**
- **Interactive HTML dashboard** with filtering and search

## Features

### 1. Phoneme Extraction
- Converts English words to phoneme sequences using CMU Dictionary (ARPABET)
- Falls back to G2P (grapheme-to-phoneme) for words not in dictionary
- Displays phonemes in both ARPABET and IPA formats

Example:
```
"ship" → ['SH', 'IH', 'P'] → /ʃ ɪ p/
"chip" → ['CH', 'IH', 'P'] → /tʃ ɪ p/
```

### 2. Phoneme-Level Alignment
- Aligns reference and hypothesis phonemes using edit distance
- Identifies which specific phonemes were mispronounced
- Calculates Phoneme Error Rate (PER)

### 3. Intelligibility Impact Classification
- **HIGH Impact**: Minimal pairs that change word meaning (e.g., "ship" → "chip")
- **MEDIUM Impact**: Noticeable errors but context helps (e.g., incomplete words)
- **LOW Impact**: Accent features that don't impede understanding

### 4. Interactive Dashboard
- **Visual error highlighting**: Color-coded words based on severity
- **Phoneme tooltips**: Hover over errors to see IPA transcription
- **Filtering**: By language, error type, or search text
- **Detailed breakdowns**: See phoneme-level differences for each error

## Usage

### Step 1: Process L2-ARCTIC Audio

```bash
# Activate virtual environment
source .venv/bin/activate

# Run processing (generates CSV files + HTML dashboard)
python process_l2arctic.py
```

Or with uv (after fixing pyproject.toml):
```bash
uv run python process_l2arctic.py
```

### Step 2: View Results

The script generates multiple output files in `data/results/`:

1. **speaker_results.csv** - Detailed per-utterance results
2. **language_statistics.csv** - Aggregated by native language
3. **speaker_statistics.csv** - Aggregated by speaker
4. **error_details.csv** - Word-level error patterns
5. **pronunciation_dashboard.html** - ⭐ **Interactive Web Dashboard**

### Step 3: Open the Dashboard

```bash
# Open the HTML file in your browser
open data/results/pronunciation_dashboard.html

# Or navigate to it manually:
# file:///path/to/pronunciation-error-detection/data/results/pronunciation_dashboard.html
```

## Dashboard Features

### Visual Error Highlighting

- 🟢 **Green**: Correct pronunciation
- 🔴 **Red background**: HIGH impact error (minimal pair)
- 🟠 **Orange background**: MEDIUM impact error
- 🟡 **Yellow background**: LOW impact error (accent feature)
- 🔵 **Blue background**: Insertion
- ❌ **Strikethrough**: Deletion

### Phoneme Information

Hover over any highlighted word to see:
- IPA transcription of expected pronunciation
- IPA transcription of actual pronunciation
- Phoneme-by-phoneme comparison

### Filters

1. **Search box**: Filter by speaker ID, language, or text content
2. **Language filter**: Show only specific native languages
3. **Error filter**:
   - All Utterances
   - Only Errors (any error type)
   - Critical Errors Only (HIGH + MEDIUM)

## Example Output

### CSV Format (speaker_results.csv)

| speaker_id | native_language | reference | hypothesis | wer | critical_errors | error_details |
|---|---|---|---|---|---|---|
| ABA | Arabic | the ship is big | the chip is big | 0.25 | 1 | 🔴 HIGH: 'ship' → 'chip' (Minimal pair: 'ship' ↔ 'chip' (different meanings)) |

### Dashboard View

```
[Speaker: ABA] [Language: Arabic] [File: arctic_a0001]

Expected: the [ship] is big
Actual:   the [chip] is big

🔴 HIGH: 'ship' → 'chip'
Expected: /ʃ ɪ p/
Actual:   /tʃ ɪ p/
Minimal pair: 'ship' ↔ 'chip' (different meanings)
```

## API Reference

### Core Functions

```python
from analysis_utils import (
    word_to_phonemes,
    format_phonemes_ipa,
    analyze_phoneme_alignment,
    create_detailed_word_errors
)

# Extract phonemes
phonemes = word_to_phonemes("ship")  # ['SH', 'IH', 'P']
ipa = format_phonemes_ipa(phonemes)  # '/ʃ ɪ p/'

# Align two words at phoneme level
alignment = analyze_phoneme_alignment("ship", "chip")
print(alignment.phoneme_operations)
# [('replace', 'SH', 'CH'), ('equal', 'IH', 'IH'), ('equal', 'P', 'P')]

# Get detailed error analysis
result = run_text_pipeline("the ship sails", "the chip sails")
detailed_errors = create_detailed_word_errors(result.error_analysis)

for error in detailed_errors:
    print(f"{error.ref_word} → {error.hyp_word}")
    print(f"  Expected: {format_phonemes_ipa(error.ref_phonemes)}")
    print(f"  Actual: {format_phonemes_ipa(error.hyp_phonemes)}")
    print(f"  Impact: {error.intelligibility_impact}")
```

### Dashboard Generation

```python
from generate_dashboard import generate_html_dashboard

# Generate dashboard from CSV results
generate_html_dashboard(
    results_csv='data/results/speaker_results.csv',
    output_html='data/results/pronunciation_dashboard.html'
)
```

## Technical Details

### Phoneme Representation

- **ARPABET**: CMU Dictionary format (e.g., 'SH', 'IH', 'P')
- **IPA**: International Phonetic Alphabet (e.g., /ʃ/, /ɪ/, /p/)

### Alignment Algorithm

Uses **Levenshtein distance** (edit distance) at both:
- **Word level**: Identifies which words differ
- **Phoneme level**: Identifies which sounds differ within words

### Intelligibility Classification

Based on:
1. **Minimal pairs database**: 15+ phoneme confusion patterns
2. **Word completion**: Detects partial/incomplete pronunciations
3. **Edit distance**: Measures overall similarity

## Dependencies

Required packages (already installed):
- `nltk` - Natural Language Toolkit (CMU Dictionary)
- `g2p-en` - Grapheme-to-phoneme conversion
- `pandas` - Data manipulation
- `tqdm` - Progress bars
- `faster-whisper` - ASR transcription

## Troubleshooting

### "Whisper not installed"
```bash
uv pip install faster-whisper
```

### "cmudict not found"
```python
import nltk
nltk.download('cmudict')
```

### "g2p_en not available"
```bash
uv pip install g2p-en
```

### Dashboard not opening
Make sure you're opening the HTML file in a web browser:
```bash
# macOS
open data/results/pronunciation_dashboard.html

# Linux
xdg-open data/results/pronunciation_dashboard.html

# Windows
start data/results/pronunciation_dashboard.html
```

## Future Enhancements

Potential additions:
- [ ] Phonetic feature analysis (voicing, place, manner)
- [ ] Language-specific phoneme inventory comparison
- [ ] Audio playback in dashboard
- [ ] Export to PDF reports
- [ ] Real-time pronunciation feedback

## References

- CMU Pronouncing Dictionary: http://www.speech.cs.cmu.edu/cgi-bin/cmudict
- IPA Chart: https://www.internationalphoneticassociation.org/content/ipa-chart
- L2-ARCTIC Corpus: https://psi.engr.tamu.edu/l2-arctic-corpus/
