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
from typing import List, Dict
from analysis_utils import (
    word_to_phonemes,
    format_phonemes_ipa,
    arpabet_to_ipa,
    analyze_phoneme_alignment,
    assess_intelligibility_impact
)


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

    # Generate word-by-word comparison with phoneme info
    ref_words = ref_text.lower().split()
    hyp_words = hyp_text.lower().split()

    comparison_html = generate_word_comparison(ref_words, hyp_words)

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
        # Convert absolute path to relative path from dashboard location
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
            # If relative path fails, use absolute
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
                <span class="metric errors">Errors: {critical_errors}</span>
            </div>
        </div>

        {audio_html}

        <div class="card-body">
            <div class="text-comparison">
                {comparison_html}
            </div>
        </div>
    </div>
    """
    return card_html


def generate_word_comparison(ref_words: List[str], hyp_words: List[str]) -> str:
    """Generate HTML comparison of reference vs hypothesis with phoneme details."""

    # Simple alignment (for demo - in production use proper alignment)
    max_len = max(len(ref_words), len(hyp_words))

    ref_html = '<div class="reference-line"><strong>Expected:</strong> '
    hyp_html = '<div class="hypothesis-line"><strong>Actual:</strong> '
    phoneme_html = '<div class="phoneme-details">'

    for i in range(max_len):
        ref_word = ref_words[i] if i < len(ref_words) else ""
        hyp_word = hyp_words[i] if i < len(hyp_words) else ""

        if ref_word and hyp_word:
            if ref_word == hyp_word:
                # Correct
                ref_html += f'<span class="word-correct">{ref_word}</span> '
                hyp_html += f'<span class="word-correct">{hyp_word}</span> '
            else:
                # Error - check if minimal pair
                impact = assess_intelligibility_impact(ref_word, hyp_word)
                error_class = f"word-error-{impact.level.lower()}"

                # Get phonemes
                ref_phonemes = word_to_phonemes(ref_word)
                hyp_phonemes = word_to_phonemes(hyp_word)
                ref_ipa = format_phonemes_ipa(ref_phonemes)
                hyp_ipa = format_phonemes_ipa(hyp_phonemes)

                ref_html += f'<span class="{error_class}" title="{ref_ipa}">{ref_word}</span> '
                hyp_html += f'<span class="{error_class}" title="{hyp_ipa}">{hyp_word}</span> '

                # Add phoneme detail
                phoneme_html += f'''
                <div class="phoneme-error">
                    <span class="error-badge-{impact.level.lower()}">{impact.level}</span>
                    <strong>'{ref_word}' → '{hyp_word}'</strong><br>
                    <span class="ipa">Expected: {ref_ipa}</span><br>
                    <span class="ipa">Actual: {hyp_ipa}</span><br>
                    <span class="explanation">{impact.explanation}</span>
                </div>
                '''
        elif ref_word:
            # Deletion
            ref_html += f'<span class="word-deleted">{ref_word}</span> '
            hyp_html += '<span class="word-deleted">—</span> '
        elif hyp_word:
            # Insertion
            ref_html += '<span class="word-inserted">—</span> '
            hyp_html += f'<span class="word-inserted">{hyp_word}</span> '

    ref_html += '</div>'
    hyp_html += '</div>'
    phoneme_html += '</div>'

    return ref_html + hyp_html + phoneme_html


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
            background: #f9f9f9;
            border-left: 5px solid #4CAF50;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 8px;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .utterance-card:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        }

        .utterance-card.high-severity {
            border-left-color: #f44336;
            background: #ffebee;
        }

        .utterance-card.medium-severity {
            border-left-color: #ff9800;
            background: #fff3e0;
        }

        .utterance-card.no-errors {
            border-left-color: #4CAF50;
        }

        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 15px;
            border-bottom: 1px solid #ddd;
        }

        .audio-player {
            background: #f5f5f5;
            padding: 15px;
            margin: 15px 0;
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

        .metric.errors {
            background: #ffebee;
            color: #c62828;
        }

        .text-comparison {
            font-family: 'Courier New', monospace;
            line-height: 1.8;
        }

        .reference-line, .hypothesis-line {
            margin-bottom: 10px;
            font-size: 1.05em;
        }

        .reference-line strong, .hypothesis-line strong {
            color: #666;
            margin-right: 10px;
        }

        .word-correct {
            color: #4CAF50;
            padding: 2px 4px;
        }

        .word-error-high {
            background: #f44336;
            color: white;
            padding: 3px 6px;
            border-radius: 4px;
            cursor: help;
        }

        .word-error-medium {
            background: #ff9800;
            color: white;
            padding: 3px 6px;
            border-radius: 4px;
            cursor: help;
        }

        .word-error-low {
            background: #ffc107;
            color: #333;
            padding: 3px 6px;
            border-radius: 4px;
            cursor: help;
        }

        .word-deleted {
            background: #ffcdd2;
            color: #c62828;
            padding: 3px 6px;
            border-radius: 4px;
            text-decoration: line-through;
        }

        .word-inserted {
            background: #b3e5fc;
            color: #0277bd;
            padding: 3px 6px;
            border-radius: 4px;
        }

        .phoneme-details {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }

        .phoneme-error {
            margin-bottom: 12px;
            padding: 10px;
            background: #f5f5f5;
            border-radius: 4px;
        }

        .error-badge-high {
            background: #f44336;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 8px;
        }

        .error-badge-medium {
            background: #ff9800;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 8px;
        }

        .error-badge-low {
            background: #ffc107;
            color: #333;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.8em;
            font-weight: bold;
            margin-right: 8px;
        }

        .ipa {
            font-family: 'Lucida Sans Unicode', 'Arial Unicode MS', sans-serif;
            color: #1976d2;
            font-size: 0.95em;
        }

        .explanation {
            color: #666;
            font-style: italic;
            font-size: 0.9em;
        }

        .hidden {
            display: none !important;
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
