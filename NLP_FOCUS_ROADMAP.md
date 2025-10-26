# NLP Project: Focused Roadmap

## Project Goal
Build an NLP system to detect pronunciation errors in non-native English speakers by comparing spoken text (from ASR) to expected reference text.

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

## Recommended Work Order

### Week 1-2: Foundation
1. Get L2-ARCTIC dataset downloaded to `data/raw/`
2. Implement ASR transcription using Whisper
3. Test basic pipeline with a few audio files

### Week 3: Core NLP
4. Enhance text preprocessing (SpaCy tokenization + lemmatization)
5. Implement robust alignment algorithm
6. Add error classification logic

### Week 4: Analysis & Metrics
7. Calculate WER, CER across dataset
8. Generate visualizations (error frequency charts)
9. Analyze patterns by speaker/accent

### Week 5: Polish & Report
10. Create final visualizations for presentation
11. Write up methodology and results
12. Prepare demo with sample audio files

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

## Next Steps (Start Here!)

1. **Install dependencies**:
   ```bash
   pip install -e ".[ml,dev,experiments]"
   ```

2. **Download L2-ARCTIC dataset**:
   - Go to http://www.openslr.org/96/
   - Download and extract to `data/raw/l2arctic/`

3. **Implement Whisper ASR**:
   - Complete the `transcribe()` function in `ped/asr.py`
   - Test on a single audio file

4. **Run the pipeline**:
   - Use `scripts/run_text_alignment.py` as a template
   - Create new script for audio processing

5. **Start analysis**:
   - Create a Jupyter notebook in `notebooks/`
   - Load results and generate visualizations

---

## Questions to Consider

- Which error types are most common across all speakers?
- Do certain language backgrounds show specific pronunciation patterns?
- How does WER correlate with speaker proficiency level?
- Can we automatically group speakers by accent based on error patterns?

Focus on getting **real results** with the L2-ARCTIC dataset rather than building infrastructure. Your NLP class cares about the linguistic analysis, not the software engineering!
