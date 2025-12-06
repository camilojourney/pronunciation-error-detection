"""
Generate Interactive HTML Dashboard for Pronunciation Error Analysis
===================================================================

This script creates an interactive web dashboard that displays:
- Individual utterance results with phoneme-level error highlighting
- Filtering and searching capabilities
- Visual representation of errors with IPA transcription
- Aggregated statistics

Author: Camilo Martinez
Course: Natural Language Processing
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

# Use consolidated phoneme sources module
from phoneme_sources import (
    get_all_phoneme_sources,
    arpabet_to_ipa,
    format_ipa,
    PhonemeSource,
    PhonemeWord,
    get_mfa_phonemes,
    get_wav2vec_phonemes
)

# Keep backward compatibility for existing code
from analysis_utils import (
    word_to_phonemes,
    format_phonemes_ipa
)

# Note: Old modules are deprecated but imports kept for backward compatibility
try:
    from forced_alignment import (
        get_phonemes_from_audio,
        mfa_phone_to_ipa,
        UtteranceAlignment,
        parse_l2arctic_textgrid
    )
    FORCED_ALIGNMENT_AVAILABLE = True
except ImportError:
    # Use new phoneme_sources module instead
    FORCED_ALIGNMENT_AVAILABLE = False

try:
    from wav2vec_phoneme import get_phonemes_for_display
    WAV2VEC_AVAILABLE = True
except ImportError:
    # Use new phoneme_sources module instead
    WAV2VEC_AVAILABLE = False


def get_annotation_path(audio_path: str) -> Optional[str]:
    """Get the TextGrid annotation path for an audio file."""
    audio_path = Path(audio_path)

    # L2-ARCTIC structure: speaker/wav/file.wav -> speaker/annotation/file.TextGrid
    if 'wav' in str(audio_path):
        annotation_path = Path(str(audio_path).replace('/wav/', '/annotation/').replace('.wav', '.TextGrid'))
        if annotation_path.exists():
            return str(annotation_path)
    return None


def generate_html_dashboard(
    results_csv: str = 'data/results/speaker_results.csv',
    output_html: str = 'data/results/pronunciation_dashboard.html'
):
    """
    Generate an interactive HTML dashboard from processing results.

    Args:
        results_csv: Path to speaker_results.csv
        output_html: Path to output HTML file
    """
    # Load results
    df = pd.read_csv(results_csv)

    # Generate HTML
    html = generate_dashboard_html(df)

    # Write to file
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✓ Dashboard generated: {output_html}")
    print(f"  Open in browser: file://{output_path.absolute()}")


def generate_dashboard_html(df: pd.DataFrame) -> str:
    """Generate complete HTML dashboard with styling and interactivity."""

    # Generate utterance cards
    utterance_cards_html = ""
    for idx, row in df.iterrows():
        utterance_cards_html += generate_utterance_card(row, idx)

    # Generate statistics summary
    stats_html = generate_statistics_summary(df)

    # Complete HTML template
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pronunciation Error Analysis Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        {get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎤 Pronunciation Error Analysis Dashboard</h1>
            <p class="subtitle">Phoneme-Level Error Detection with Intelligibility Analysis</p>
        </header>

        {stats_html}

        <div class="filters">
            <h2>🔍 Filters & Search</h2>
            <div class="filter-controls">
                <input type="text" id="searchBox" placeholder="Search by speaker, language, or text..." onkeyup="filterUtterances()">

                <select id="languageFilter" onchange="filterUtterances()">
                    <option value="all">All Languages</option>
                    {generate_language_options(df)}
                </select>

                <select id="errorFilter" onchange="filterUtterances()">
                    <option value="all">All Utterances</option>
                    <option value="errors">Only Errors</option>
                    <option value="critical">Critical Errors Only</option>
                </select>
            </div>
        </div>

        <div class="utterances-container">
            <h2>📝 Utterance Analysis</h2>
            <div id="utterancesList">
                {utterance_cards_html}
            </div>
        </div>
    </div>

    <script>
        {get_javascript()}
    </script>
</body>
</html>
"""
    return html


def generate_utterance_card(row: pd.Series, idx: int) -> str:
    """Generate HTML card for a single utterance with phoneme-level details."""

    speaker_id = row['speaker_id']
    native_lang = row['native_language']
    file_id = row['file_id']
    audio_path = row.get('audio_path', '')
    ref_text = row['reference']
    hyp_text = row['hypothesis']
    wer = row['wer']
    critical_errors = row.get('critical_errors', 0)
    high_impact = row.get('high_impact_errors', 0)

    # Get all 3 phoneme sources using unified API
    phoneme_sources_html = ""
    if audio_path:
        try:
            sources = get_all_phoneme_sources(
                audio_path=audio_path,
                expected_text=ref_text
            )
            phoneme_sources_html = generate_three_sources_display(sources, ref_text)
        except Exception as e:
            # If phoneme extraction fails, fall back to text-only display
            phoneme_sources_html = f"""
            <div class="phoneme-section">
                <p style="color: #666; font-style: italic;">⚠️ Phoneme extraction unavailable: {str(e)}</p>
            </div>
            """

    # Also show Whisper transcription for comparison
    whisper_html = f"""
    <div class="phoneme-section">
        <h4>🎤 Whisper Transcription</h4>
        <p class="text-display">{hyp_text}</p>
    </div>
    """

    # Determine error severity class
    severity_class = "no-errors"
    if critical_errors > 0:
        if high_impact > 0:
            severity_class = "high-severity"
        else:
            severity_class = "medium-severity"

    # Audio player HTML (if audio path is available)
    audio_html = ""
    if audio_path:
        from pathlib import Path
        try:
            audio_rel_path = Path(audio_path).relative_to(Path.cwd())
            audio_html = f"""
            <div class="audio-player">
                <label>🎧 Listen to pronunciation:</label>
                <audio controls preload="none">
                    <source src="../../{audio_rel_path}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """
        except:
            audio_html = f"""
            <div class="audio-player">
                <label>🎧 Listen to pronunciation:</label>
                <audio controls preload="none">
                    <source src="file://{audio_path}" type="audio/wav">
                    Your browser does not support the audio element.
                </audio>
            </div>
            """

    card_html = f"""
    <div class="utterance-card {severity_class}" data-speaker="{speaker_id}" data-language="{native_lang}" data-errors="{critical_errors}">
        <div class="card-header">
            <div class="speaker-info">
                <span class="speaker-id">{speaker_id}</span>
                <span class="language-badge">{native_lang}</span>
                <span class="file-id">{file_id}</span>
            </div>
            <div class="metrics">
                <span class="metric wer">WER: {wer:.1%}</span>
            </div>
        </div>

        {audio_html}

        <div class="card-body">
            {phoneme_sources_html}
            {whisper_html}
        </div>
    </div>
    """
    return card_html


def generate_three_sources_display(sources: Dict[str, PhonemeSource], ref_text: str) -> str:
    """
    Generate HTML display for all 3 phoneme sources with difference highlighting.

    Args:
        sources: Dictionary with 'dictionary', 'whisper_large_mfa', 'whisper_large_wav2vec'
        ref_text: Reference text for context

    Returns:
        HTML string showing all 3 sources with differences highlighted
    """
    html_parts = []

    # Source 1: Dictionary (Expected)
    dict_src = sources.get('dictionary')
    if dict_src and dict_src.success and dict_src.words:
        words_html = []
        for pw in dict_src.words:
            words_html.append(f'<span class="phoneme-word">{pw.word} <span class="ipa">{pw.ipa}</span></span>')

        html_parts.append(f"""
        <div class="phoneme-section">
            <h4>📚 Dictionary (Expected Pronunciation)</h4>
            <div class="phoneme-words">
                {' '.join(words_html)}
            </div>
        </div>
        """)

    # Source 2: Whisper Large-v3 + MFA (Actual via alignment)
    mfa_src = sources.get('whisper_large_mfa')
    if mfa_src and mfa_src.success and mfa_src.words:
        words_html = []
        differences = []

        # Compare with dictionary to highlight differences
        dict_map = {pw.word.lower(): pw.ipa for pw in dict_src.words} if dict_src and dict_src.words else {}

        for pw in mfa_src.words:
            word_lower = pw.word.lower()
            expected_ipa = dict_map.get(word_lower, '')
            actual_ipa = pw.ipa

            # Check if different (normalize by removing slashes)
            if expected_ipa and expected_ipa.strip('/') != actual_ipa.strip('/'):
                # Highlight difference in red
                words_html.append(f'<span class="phoneme-word error">{pw.word} <span class="ipa">{pw.ipa}</span></span>')
                differences.append((pw.word, expected_ipa, actual_ipa))
            else:
                words_html.append(f'<span class="phoneme-word">{pw.word} <span class="ipa">{pw.ipa}</span></span>')

        html_parts.append(f"""
        <div class="phoneme-section">
            <h4>🔬 Whisper Large-v3 + MFA (Actual via Forced Alignment)</h4>
            <div class="phoneme-words">
                {' '.join(words_html)}
            </div>
        </div>
        """)
    elif mfa_src and not mfa_src.success:
        html_parts.append(f"""
        <div class="phoneme-section">
            <h4>🔬 Whisper Large-v3 + MFA</h4>
            <p style="color: #999; font-style: italic;">⚠️ MFA not available (requires praatio library)</p>
        </div>
        """)

    # Source 3: Whisper Large-v3 + Wav2Vec2 (Actual via recognition)
    wav2vec_src = sources.get('whisper_large_wav2vec')
    if wav2vec_src and wav2vec_src.success and wav2vec_src.words:
        words_html = []

        # Compare with dictionary to highlight differences
        dict_map = {pw.word.lower(): pw.ipa for pw in dict_src.words} if dict_src and dict_src.words else {}

        for pw in wav2vec_src.words:
            word_lower = pw.word.lower()
            expected_ipa = dict_map.get(word_lower, '')
            actual_ipa = pw.ipa

            # Check if different (normalize by removing slashes)
            if expected_ipa and expected_ipa.strip('/') != actual_ipa.strip('/'):
                # Highlight difference in red
                words_html.append(f'<span class="phoneme-word error">{pw.word} <span class="ipa">{pw.ipa}</span></span>')
            else:
                words_html.append(f'<span class="phoneme-word">{pw.word} <span class="ipa">{pw.ipa}</span></span>')

        html_parts.append(f"""
        <div class="phoneme-section">
            <h4>🎧 Whisper Large-v3 + Wav2Vec2 (Actual via Direct Recognition)</h4>
            <div class="phoneme-words">
                {' '.join(words_html)}
            </div>
        </div>
        """)
    elif wav2vec_src and not wav2vec_src.success:
        html_parts.append(f"""
        <div class="phoneme-section">
            <h4>🎧 Whisper Large-v3 + Wav2Vec2</h4>
            <p style="color: #999; font-style: italic;">⚠️ Wav2Vec2 error: {wav2vec_src.error}</p>
        </div>
        """)

    return '\n'.join(html_parts)


def generate_wav2vec_display(wav2vec_result: dict) -> str:
    """
    Generate display showing phonemes from Wav2Vec2 direct audio recognition.
    """
    if not wav2vec_result.get('success'):
        return ""

    raw_output = wav2vec_result.get('raw_output', '')

    return f'''
    <div class="text-section wav2vec-phonemes">
        <h4>🎧 Wav2Vec2 (Direct Audio Recognition)</h4>
        <div class="phoneme-stream">
            /{raw_output}/
        </div>
    </div>
    '''


def generate_mfa_phonemes_display(alignment, ref_text: str) -> str:
    """
    Generate display showing ACTUAL phonemes from MFA forced alignment.

    Compares actual pronunciation against expected dictionary phonemes.
    """
    if not alignment or not alignment.words:
        return ""

    words_html = ""
    errors_found = 0

    for word_align in alignment.words:
        word = word_align.word

        # Actual phonemes from MFA
        actual_ipa = [mfa_phone_to_ipa(p.phoneme) for p in word_align.phonemes]
        actual_str = " ".join(actual_ipa) if actual_ipa else "—"

        # Expected phonemes from dictionary
        expected = word_to_phonemes(word)
        if expected:
            expected_ipa = [p.strip('/').strip() for p in format_phonemes_ipa(expected).split()]
        else:
            expected_ipa = []

        # Check for pronunciation error
        has_error = actual_ipa != expected_ipa
        error_class = "pronunciation-error" if has_error else ""
        if has_error:
            errors_found += 1

        words_html += f'''
        <div class="word-block {error_class}">
            <span class="word-text">{word}</span>
            <span class="word-ipa">/{actual_str}/</span>
        </div>
        '''

    error_badge = f'<span class="error-badge">{errors_found} errors</span>' if errors_found > 0 else '<span class="success-badge">✓ Perfect</span>'

    return f'''
    <div class="text-section mfa-phonemes">
        <h4>🎯 Actual Pronunciation (MFA) {error_badge}</h4>
        <div class="word-container">
            {words_html}
        </div>
    </div>
    '''


def generate_text_with_phonemes(text: str, label: str) -> str:
    """
    Generate a simple display of text with phonemes directly below each word.
    No complex alignment - just show what phonemes each word should have.
    """
    import re

    # Clean and split into words
    words = text.split()

    words_html = ""
    for word in words:
        # Clean the word for phoneme lookup (remove punctuation)
        clean_word = re.sub(r'[^\w\']', '', word.lower())

        if clean_word:
            phonemes = word_to_phonemes(clean_word)
            ipa = format_phonemes_ipa(phonemes) if phonemes else "—"
        else:
            ipa = ""

        words_html += f'''
        <div class="word-block">
            <span class="word-text">{word}</span>
            <span class="word-ipa">{ipa}</span>
        </div>
        '''

    return f'''
    <div class="text-section">
        <div class="section-label">{label}:</div>
        <div class="words-container">
            {words_html}
        </div>
    </div>
    '''


def generate_statistics_summary(df: pd.DataFrame) -> str:
    """Generate summary statistics HTML."""

    total_utterances = len(df)
    avg_wer = df['wer'].mean()
    utterances_with_errors = (df['critical_errors'] > 0).sum()
    total_critical = df['critical_errors'].sum()

    return f"""
    <div class="statistics-summary">
        <h2>📊 Summary Statistics</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_utterances}</div>
                <div class="stat-label">Total Utterances</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_wer:.1%}</div>
                <div class="stat-label">Average WER</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{utterances_with_errors}</div>
                <div class="stat-label">With Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_critical}</div>
                <div class="stat-label">Total Critical Errors</div>
            </div>
        </div>
    </div>
    """


def generate_language_options(df: pd.DataFrame) -> str:
    """Generate language filter options."""
    languages = df['native_language'].unique()
    options = ""
    for lang in sorted(languages):
        options += f'<option value="{lang}">{lang}</option>\n'
    return options


def get_css_styles() -> str:
    """Return CSS styles for the dashboard."""
    return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
            text-align: center;
        }

        header h1 {
            color: #333;
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .subtitle {
            color: #666;
            font-size: 1.1em;
        }

        .statistics-summary {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .statistics-summary h2 {
            color: #333;
            margin-bottom: 20px;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-value {
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.9em;
            opacity: 0.9;
        }

        .filters {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .filters h2 {
            color: #333;
            margin-bottom: 15px;
        }

        .filter-controls {
            display: grid;
            grid-template-columns: 2fr 1fr 1fr;
            gap: 15px;
        }

        input[type="text"], select {
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 6px;
            font-size: 1em;
            transition: border-color 0.3s;
        }

        input[type="text"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }

        .utterances-container {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .utterances-container h2 {
            color: #333;
            margin-bottom: 20px;
        }

        .utterance-card {
            background: #fafafa;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #eee;
        }

        .audio-player {
            background: #f5f5f5;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }

        .audio-player label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #555;
        }

        .audio-player audio {
            width: 100%;
            max-width: 500px;
            height: 40px;
        }

        .speaker-info {
            display: flex;
            gap: 10px;
            align-items: center;
        }

        .speaker-id {
            font-weight: bold;
            font-size: 1.2em;
            color: #333;
        }

        .language-badge {
            background: #667eea;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
        }

        .file-id {
            color: #888;
            font-size: 0.9em;
        }

        .metrics {
            display: flex;
            gap: 15px;
        }

        .metric {
            padding: 5px 12px;
            border-radius: 5px;
            font-size: 0.9em;
        }

        .metric.wer {
            background: #e3f2fd;
            color: #1976d2;
        }

        .hidden {
            display: none !important;
        }

        /* ===== Minimalistic Text + Phoneme Display ===== */
        .text-section {
            margin-bottom: 20px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }

        .text-section:last-child {
            margin-bottom: 0;
        }

        .section-label {
            font-weight: 600;
            color: #555;
            font-size: 0.85em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
        }

        .words-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: flex-start;
        }

        .word-block {
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 8px 12px;
            background: #f8f9fa;
            border-radius: 6px;
            border: 1px solid #e9ecef;
            min-width: 50px;
        }

        .word-text {
            font-size: 1.1em;
            font-weight: 500;
            color: #333;
            margin-bottom: 4px;
        }

        .word-ipa {
            font-family: 'Noto Sans', 'Roboto', 'Arial', sans-serif;
            font-size: 1.1em;
            color: #5f6368;
            letter-spacing: 1px;
            font-weight: 400;
            line-height: 1.6;
        }

        /* Actual phonemes from audio annotation */
        .text-section.actual-phonemes {
            background: #e8f5e9;
            border-color: #4CAF50;
        }

        .text-section.actual-phonemes .section-label {
            color: #2e7d32;
        }

        .word-block.actual {
            background: #c8e6c9;
            border-color: #81c784;
        }

        .word-block.actual .word-ipa {
            color: #1b5e20;
        }

        .word-duration {
            font-size: 0.7em;
            color: #666;
            margin-top: 2px;
        }

        /* MFA Actual Pronunciation Section */
        .text-section.mfa-phonemes {
            background: #fff3e0;
            border-color: #ff9800;
        }

        .text-section.mfa-phonemes h4 {
            color: #e65100;
            margin: 0 0 12px 0;
            font-size: 0.95em;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .text-section.mfa-phonemes .word-container {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .text-section.mfa-phonemes .word-block {
            background: #ffe0b2;
            border-color: #ffb74d;
        }

        .text-section.mfa-phonemes .word-block .word-ipa {
            color: #e65100;
        }

        /* Pronunciation error highlighting */
        .word-block.pronunciation-error {
            background: #ffebee !important;
            border-color: #f44336 !important;
            box-shadow: 0 0 0 2px rgba(244, 67, 54, 0.3);
        }

        .word-block.pronunciation-error .word-text {
            color: #c62828;
        }

        .word-block.pronunciation-error .word-ipa {
            color: #b71c1c !important;
            font-weight: bold;
        }

        .error-badge {
            background: #f44336;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: 500;
        }

        .success-badge {
            background: #4CAF50;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: 500;
        }

        /* Wav2Vec2 Section */
        .text-section.wav2vec-phonemes {
            background: #e3f2fd;
            border-color: #2196F3;
        }

        .text-section.wav2vec-phonemes h4 {
            color: #1565c0;
            margin: 0 0 12px 0;
            font-size: 0.95em;
        }

        .phoneme-stream {
            font-family: 'Noto Sans', 'Roboto', 'Arial', sans-serif;
            font-size: 1.2em;
            color: #0d47a1;
            background: #bbdefb;
            padding: 12px 16px;
            border-radius: 6px;
            word-spacing: 3px;
            letter-spacing: 1px;
            line-height: 1.8;
            font-weight: 400;
        }

        /* New 3-source comparison styles */
        .phoneme-section {
            margin-bottom: 20px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }

        .phoneme-section h4 {
            margin: 0 0 12px 0;
            color: #333;
            font-size: 0.95em;
            font-weight: 600;
        }

        .phoneme-words {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            line-height: 2;
        }

        .phoneme-word {
            display: inline-flex;
            flex-direction: column;
            align-items: center;
            padding: 6px 10px;
            background: white;
            border-radius: 5px;
            border: 1px solid #ddd;
        }

        .phoneme-word.error {
            background: #ffebee;
            border-color: #f44336;
        }

        .phoneme-word .ipa {
            font-family: 'Noto Sans', 'Roboto', 'Arial', sans-serif;
            font-size: 0.95em;
            color: #666;
            margin-top: 2px;
            font-weight: 400;
            letter-spacing: 0.5px;
        }

        .phoneme-word.error .ipa {
            color: #c62828;
            font-weight: 500;
        }

        .text-display {
            font-size: 1.05em;
            color: #555;
            line-height: 1.6;
        }
    """


def get_javascript() -> str:
    """Return JavaScript for dashboard interactivity."""
    return """
        function filterUtterances() {
            const searchText = document.getElementById('searchBox').value.toLowerCase();
            const languageFilter = document.getElementById('languageFilter').value;
            const errorFilter = document.getElementById('errorFilter').value;

            const cards = document.querySelectorAll('.utterance-card');

            cards.forEach(card => {
                const text = card.textContent.toLowerCase();
                const language = card.dataset.language;
                const errors = parseInt(card.dataset.errors);

                let show = true;

                // Search filter
                if (searchText && !text.includes(searchText)) {
                    show = false;
                }

                // Language filter
                if (languageFilter !== 'all' && language !== languageFilter) {
                    show = false;
                }

                // Error filter
                if (errorFilter === 'errors' && errors === 0) {
                    show = false;
                } else if (errorFilter === 'critical' && errors === 0) {
                    show = false;
                }

                if (show) {
                    card.classList.remove('hidden');
                } else {
                    card.classList.add('hidden');
                }
            });
        }
    """


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1:
        results_csv = sys.argv[1]
    else:
        results_csv = 'data/results/speaker_results.csv'

    if len(sys.argv) > 2:
        output_html = sys.argv[2]
    else:
        output_html = 'data/results/pronunciation_dashboard.html'

    generate_html_dashboard(results_csv, output_html)
