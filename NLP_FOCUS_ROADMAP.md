# NLP Project: Focused Roadmap

## Project Goal
Build an NLP system to detect pronunciation errors in non-native English speakers by comparing spoken text (from ASR) to expected reference text using the **L2-ARCTIC speech corpus**.

## Current Project Status ✅

**Dataset**: L2-ARCTIC Release v5.0
- ✅ **Downloaded and extracted** in `l2arctic_release_v5/`
- ✅ **24 non-native speakers** from diverse language backgrounds
- ✅ **~1,130 utterances per speaker** (27,120 total audio files)
- ✅ **Manifest created** (`data/processed/l2arctic_manifest.json`) with audio paths and reference texts
- ✅ **Languages**: Arabic, Chinese (Mandarin), Hindi, Korean, Spanish, Vietnamese, and more

**What We Have**:
- ✅ L2-ARCTIC corpus with reference transcripts
- ✅ Data science notebook (`nlp_presentation.qmd`)
- ✅ NLP utility package (`analysis_utils.py`) with all functions
- ⚠️ **Need to do**: Real ASR transcription on L2-ARCTIC audio files
- ⚠️ **Need to do**: Process audio and generate real results

---

## What We Need to Focus On (For Your NLP Class)

### Phase 1: Core NLP Pipeline (PRIORITY)
These are the essential components you need to complete for your class project:

#### 1. Text Preprocessing & Normalization
- **Status**: Basic implementation exists in `ped/text.py`
- **What to do**:
  - Text cleaning (lowercase, punctuation removal)
  - Tokenization using SpaCy or NLTK
  - Lemmatization for consistent word forms
- **Why**: Foundation for all alignment and error detection

#### 2. Sequence Alignment (Text-to-Text)
- **Status**: Basic implementation exists using `difflib`
- **What to do**:
  - Implement Levenshtein distance-based alignment
  - Word-level alignment between reference and hypothesis
  - Identify insertions, deletions, substitutions
- **Why**: Core NLP technique for detecting pronunciation proxies

#### 3. Error Detection & Classification
- **Status**: Not implemented yet
- **What to do**:
  - Classify error types (insertion, deletion, substitution)
  - Create error frequency analysis
  - Identify common mispronunciation patterns
- **Why**: Main deliverable for your NLP project

#### 4. Metrics & Evaluation
- **Status**: Basic WER implemented in `ped/metrics.py`
- **What to do**:
  - Word Error Rate (WER)
  - Character Error Rate (CER)
  - Phoneme Error Rate (PER) - optional advanced
- **Why**: Quantitative evaluation of system performance

#### 5. Visualization & Analysis
- **Status**: Not implemented
- **What to do**:
  - Create charts showing error distributions
  - Visualize common error patterns by accent/speaker
  - Generate summary statistics
- **Why**: Required for project presentation and report

---

## Phase 2: ASR Integration (REQUIRED for Audio Analysis)

#### 6. Whisper ASR Integration
- **Status**: Placeholder in `ped/asr.py`
- **What to do**:
  - Implement `faster-whisper` or `openai-whisper`
  - Transcribe audio files from L2-ARCTIC dataset
  - Generate hypothesis text for alignment
- **Why**: Converts audio to text for comparison

---

## Phase 3: Advanced NLP (OPTIONAL)

#### 7. Phoneme-Level Analysis
- **Status**: Placeholder in `ped/g2p.py`
- **What to do**:
  - Grapheme-to-Phoneme (G2P) conversion using `g2p-en` or `phonemizer`
  - IPA transcription alignment
  - Phoneme Error Rate calculation
- **Why**: More granular pronunciation error detection

#### 8. Pattern Clustering
- **What to do**:
  - Use clustering (K-means, DBSCAN) to group similar errors
  - Topic modeling for error types
  - Accent-based error pattern analysis
- **Why**: Discover systematic pronunciation patterns

---

## Step-by-Step Workflow (What We're Actually Doing)

### ✅ Phase 1: Dataset Preparation (DONE)
1. ✅ Downloaded L2-ARCTIC dataset (27,120 audio files)
2. ✅ Created manifest file mapping audio files to reference texts
3. ✅ Organized dataset by speaker and language background

### ✅ Phase 2: Implementation (DONE)
4. ✅ **Created `analysis_utils.py`** - Complete NLP package with:
   - Text preprocessing, tokenization, alignment
   - WER/CER calculation
   - Whisper ASR transcription
   - Error detection and classification
   - End-to-end pipelines (text and audio)

5. **Run ASR on L2-ARCTIC samples**:
   - Select subset of speakers (e.g., 5-10 speakers, 50 utterances each)
   - Transcribe using Whisper (base or small model)
   - Save transcriptions alongside reference texts

6. **Perform alignment and error detection**:
   - Align ASR output with reference text using Levenshtein distance
   - Classify errors (substitutions, deletions, insertions)
   - Calculate WER per speaker and per language group

### 📊 Phase 3: Analysis & Visualization (NEXT)
7. **Statistical analysis**:
   - Calculate aggregate WER across speakers
   - Identify most common substitution patterns
   - Analyze error distribution by native language

8. **Generate visualizations**:
   - Error type distribution (pie/bar charts)
   - WER by speaker (horizontal bar chart)
   - Error patterns by language background (stacked bar)
   - Common substitution patterns (table)

### 📝 Phase 4: Presentation (FINAL)
9. **Update nlp_presentation.qmd** with real results
10. **Render final HTML presentation**
11. **Prepare live demo** showcasing pronunciation error detection

---

## What Files to Keep vs. Remove

### KEEP (Essential for NLP Project):
- `ped/` - Core Python package with all processing logic
- `ped/text.py` - Text preprocessing
- `ped/metrics.py` - WER and other metrics
- `ped/asr.py` - ASR integration (needs implementation)
- `ped/g2p.py` - Phoneme conversion (optional)
- `ped/pipeline.py` - Orchestration
- `scripts/run_text_alignment.py` - Demo script
- `tests/` - Unit tests
- `notebooks/` - For experimentation and visualization
- `data/` - Dataset storage
- `NLP_PROJECT_REPORT.md` - Your project documentation
- `README.md` - Project overview
- `pyproject.toml` - Dependencies

### REMOVED (Infrastructure files not needed):
- `.github/workflows/` - CI/CD automation
- `.pre-commit-config.yaml` - Git hooks
- `justfile` - Task automation
- `GITHUB_ACCESS.md` - Setup documentation
- `docs/infra-recommendations.md` - Infrastructure guide

---

## Key NLP Concepts You'll Demonstrate

1. **Sequence Alignment**: Comparing two text sequences (core NLP technique)
2. **Tokenization & Lemmatization**: Text preprocessing fundamentals
3. **Edit Distance**: Levenshtein distance for string comparison
4. **Error Analysis**: Systematic categorization of linguistic errors
5. **Evaluation Metrics**: WER, PER for sequence prediction tasks
6. **Phonetic Analysis**: Sound-level linguistic analysis (if you do Phase 3)
7. **Pattern Recognition**: Identifying recurring error types
8. **Visualization**: Communicating NLP results effectively

---

## Deliverables for Your NLP Class

1. **Code**: Working Python pipeline in `ped/` package
2. **Dataset**: Processed L2-ARCTIC data with alignments
3. **Metrics**: WER, error distribution statistics
4. **Visualizations**: Charts showing error patterns
5. **Report**: `NLP_PROJECT_REPORT.md` with methodology and results
6. **Demo**: Script showing alignment and error detection on sample audio
7. **Presentation**: Slides explaining your approach and findings

---

## L2-ARCTIC Dataset Details

**Dataset Information**:
- **Source**: OpenSLR.org/96 (University of Edinburgh)
- **Speakers**: 24 non-native English speakers
- **Native Languages**: Arabic, Mandarin, Hindi, Korean, Spanish, Vietnamese, and others
- **Content**: CMU ARCTIC prompts (phonetically balanced English sentences)
- **Format**: 16kHz WAV audio files
- **Annotations**: Word-level transcripts, TextGrid alignments

**Speakers in Our Dataset** (Examples):
- `ABA` - Arabic
- `ASI` - ? (need to check)
- `BWC` - ? (need to check)
- Each speaker has ~1,130 utterances

**Directory Structure**:
```
l2arctic_release_v5/
├── ABA/              # Arabic speaker
│   ├── wav/          # Audio files (1,130 files)
│   ├── transcript/   # Reference texts
│   └── textgrid/     # Phonetic alignments
├── ASI/
├── BWC/
└── ...
```

## Next Steps (What to Do Right Now)

### Immediate Actions:

1. **Install Whisper** (if not already installed):
   ```bash
   pip install faster-whisper
   ```

2. **Process Real L2-ARCTIC Data**:
   - Select 5-10 speakers from different language backgrounds
   - Transcribe 20-50 utterances per speaker using Whisper
   - Perform alignment and calculate real WER
   - Generate actual statistics and visualizations

4. **Render and Test**:
   - Run `quarto render nlp_presentation.qmd`
   - Verify all code cells execute correctly
   - Check that visualizations display properly

---

## Questions to Consider

- Which error types are most common across all speakers?
- Do certain language backgrounds show specific pronunciation patterns?
- How does WER correlate with speaker proficiency level?
- Can we automatically group speakers by accent based on error patterns?

Focus on getting **real results** with the L2-ARCTIC dataset rather than building infrastructure. Your NLP class cares about the linguistic analysis, not the software engineering!
