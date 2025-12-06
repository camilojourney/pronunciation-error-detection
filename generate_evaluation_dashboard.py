"""
Generate Interactive Evaluation Dashboard with Plotly
=====================================================

Creates an HTML dashboard showing:
- Overall performance metrics (accuracy, precision, recall, F1)
- Interactive Plotly visualizations
- Confusion matrices
- Per-language breakdowns
- Hyperparameter tuning results

Author: Camilo Martinez
Course: Natural Language Processing
"""

import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import Counter

from evaluation_metrics import EvaluationResult, aggregate_results_by_language, get_top_confusions


def generate_confusion_matrix_plot(results: List[EvaluationResult]) -> str:
    """
    Generate Plotly confusion matrix heatmap as JSON.

    Args:
        results: List of evaluation results

    Returns:
        JSON string for Plotly heatmap
    """
    # Collect all confusions
    confusion_data = Counter()
    for result in results:
        if result.phoneme_level_metrics:
            for (expected, actual), count in result.phoneme_level_metrics.confusion_matrix.items():
                confusion_data[(expected, actual)] += count

    if not confusion_data:
        return json.dumps({})

    # Get unique phonemes
    all_phonemes = sorted(set([p for pair in confusion_data.keys() for p in pair]))

    # Build matrix
    matrix = []
    for expected in all_phonemes:
        row = []
        for actual in all_phonemes:
            row.append(confusion_data.get((expected, actual), 0))
        matrix.append(row)

    # Create Plotly heatmap
    plot_data = {
        'data': [{
            'type': 'heatmap',
            'z': matrix,
            'x': all_phonemes,
            'y': all_phonemes,
            'colorscale': 'Reds',
            'hovertemplate': 'Expected: %{y}<br>Actual: %{x}<br>Count: %{z}<extra></extra>'
        }],
        'layout': {
            'title': 'Phoneme Confusion Matrix',
            'xaxis': {'title': 'Actual Phoneme (PPL)', 'side': 'bottom'},
            'yaxis': {'title': 'Expected Phoneme (CPL)'},
            'height': 600,
            'margin': {'l': 100, 'r': 50, 't': 80, 'b': 100}
        }
    }

    return json.dumps(plot_data)


def generate_precision_recall_by_language_plot(lang_df: pd.DataFrame) -> str:
    """
    Generate Plotly grouped bar chart for precision/recall by language.

    Args:
        lang_df: DataFrame with per-language metrics

    Returns:
        JSON string for Plotly bar chart
    """
    if lang_df.empty:
        return json.dumps({})

    plot_data = {
        'data': [
            {
                'type': 'bar',
                'name': 'Precision',
                'x': lang_df['language'].tolist(),
                'y': lang_df['word_precision'].tolist(),
                'marker': {'color': '#3498db'}
            },
            {
                'type': 'bar',
                'name': 'Recall',
                'x': lang_df['language'].tolist(),
                'y': lang_df['word_recall'].tolist(),
                'marker': {'color': '#e74c3c'}
            },
            {
                'type': 'bar',
                'name': 'F1 Score',
                'x': lang_df['language'].tolist(),
                'y': lang_df['word_f1'].tolist(),
                'marker': {'color': '#2ecc71'}
            }
        ],
        'layout': {
            'title': 'Word-Level Metrics by Native Language',
            'xaxis': {'title': 'Native Language'},
            'yaxis': {'title': 'Score', 'range': [0, 1]},
            'barmode': 'group',
            'height': 400
        }
    }

    return json.dumps(plot_data)


def generate_error_distribution_plot(results: List[EvaluationResult]) -> str:
    """
    Generate Plotly pie chart showing error type distribution.

    Args:
        results: List of evaluation results

    Returns:
        JSON string for Plotly pie chart
    """
    total_subs = sum(r.gt_substitutions for r in results)
    total_dels = sum(r.gt_deletions for r in results)
    total_ins = sum(r.gt_insertions for r in results)

    plot_data = {
        'data': [{
            'type': 'pie',
            'labels': ['Substitutions', 'Deletions', 'Insertions'],
            'values': [total_subs, total_dels, total_ins],
            'marker': {'colors': ['#e74c3c', '#f39c12', '#9b59b6']},
            'hovertemplate': '%{label}: %{value} (%{percent})<extra></extra>'
        }],
        'layout': {
            'title': 'Error Type Distribution (Ground Truth)',
            'height': 400
        }
    }

    return json.dumps(plot_data)


def generate_per_vs_wer_scatter_plot(results: List[EvaluationResult]) -> str:
    """
    Generate Plotly scatter plot comparing PER vs WER.

    Args:
        results: List of evaluation results

    Returns:
        JSON string for Plotly scatter plot
    """
    per_values = [r.gt_phoneme_error_rate * 100 for r in results]
    wer_values = [r.pred_wer * 100 for r in results]
    languages = [r.native_language or 'Unknown' for r in results]

    # Get unique languages for color mapping
    unique_langs = sorted(set(languages))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    color_map = {lang: colors[i % len(colors)] for i, lang in enumerate(unique_langs)}

    # Create traces per language
    traces = []
    for lang in unique_langs:
        lang_per = [per_values[i] for i, l in enumerate(languages) if l == lang]
        lang_wer = [wer_values[i] for i, l in enumerate(languages) if l == lang]

        traces.append({
            'type': 'scatter',
            'mode': 'markers',
            'name': lang,
            'x': lang_per,
            'y': lang_wer,
            'marker': {'color': color_map[lang], 'size': 8},
            'hovertemplate': f'{lang}<br>PER: %{{x:.1f}}%<br>WER: %{{y:.1f}}%<extra></extra>'
        })

    plot_data = {
        'data': traces,
        'layout': {
            'title': 'Phoneme Error Rate (Ground Truth) vs Word Error Rate (Prediction)',
            'xaxis': {'title': 'Phoneme Error Rate (%)'},
            'yaxis': {'title': 'Word Error Rate (%)'},
            'height': 500,
            'hovermode': 'closest'
        }
    }

    return json.dumps(plot_data)


def generate_summary_cards(results: List[EvaluationResult], lang_df: pd.DataFrame) -> str:
    """
    Generate HTML summary cards with overall statistics.

    Args:
        results: List of evaluation results
        lang_df: DataFrame with per-language metrics

    Returns:
        HTML string for summary cards
    """
    total_files = len(results)
    total_languages = len(lang_df) if not lang_df.empty else 0

    # Calculate overall word-level metrics
    total_tp = sum(r.word_level_metrics.true_positives for r in results if r.word_level_metrics)
    total_fp = sum(r.word_level_metrics.false_positives for r in results if r.word_level_metrics)
    total_tn = sum(r.word_level_metrics.true_negatives for r in results if r.word_level_metrics)
    total_fn = sum(r.word_level_metrics.false_negatives for r in results if r.word_level_metrics)

    total = total_tp + total_fp + total_tn + total_fn
    accuracy = (total_tp + total_tn) / total if total > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Average PER from ground truth
    avg_per = sum(r.gt_phoneme_error_rate for r in results) / len(results) if results else 0

    html = f"""
    <div class="statistics-summary">
        <h2>Overall Performance Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{total_files}</div>
                <div class="stat-label">Files Evaluated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{total_languages}</div>
                <div class="stat-label">Languages</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{avg_per:.1%}</div>
                <div class="stat-label">Avg PER (GT)</div>
            </div>
            <div class="stat-card metric-accuracy">
                <div class="stat-value">{accuracy:.3f}</div>
                <div class="stat-label">Accuracy</div>
            </div>
            <div class="stat-card metric-precision">
                <div class="stat-value">{precision:.3f}</div>
                <div class="stat-label">Precision</div>
            </div>
            <div class="stat-card metric-recall">
                <div class="stat-value">{recall:.3f}</div>
                <div class="stat-label">Recall</div>
            </div>
            <div class="stat-card metric-f1">
                <div class="stat-value">{f1:.3f}</div>
                <div class="stat-label">F1 Score</div>
            </div>
        </div>
    </div>
    """

    return html


def generate_language_table(lang_df: pd.DataFrame) -> str:
    """Generate HTML table for per-language metrics"""
    if lang_df.empty:
        return "<p>No language data available</p>"

    html = """
    <div class="table-container">
        <h3>Metrics by Native Language</h3>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Language</th>
                    <th>Sample Size</th>
                    <th>Avg PER</th>
                    <th>Word Precision</th>
                    <th>Word Recall</th>
                    <th>Word F1</th>
                    <th>Phoneme F1</th>
                </tr>
            </thead>
            <tbody>
    """

    for _, row in lang_df.iterrows():
        html += f"""
                <tr>
                    <td><strong>{row['language']}</strong></td>
                    <td>{row['sample_size']}</td>
                    <td>{row['avg_per']:.1%}</td>
                    <td>{row['word_precision']:.3f}</td>
                    <td>{row['word_recall']:.3f}</td>
                    <td>{row['word_f1']:.3f}</td>
                    <td>{row['phoneme_f1']:.3f}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    """

    return html


def generate_top_confusions_table(results: List[EvaluationResult], top_n: int = 15) -> str:
    """Generate HTML table showing top phoneme confusions"""
    confusions_df = get_top_confusions(results, top_n=top_n)

    if confusions_df.empty:
        return "<p>No confusion data available</p>"

    html = """
    <div class="table-container">
        <h3>Top Phoneme Confusions</h3>
        <table class="metrics-table">
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Expected</th>
                    <th>Actual</th>
                    <th>Count</th>
                    <th>Example</th>
                </tr>
            </thead>
            <tbody>
    """

    # Example words for common confusions
    examples = {
        ('TH', 'S'): '"think" → "sink"',
        ('DH', 'D'): '"this" → "dis"',
        ('R', 'L'): '"right" → "light"',
        ('L', 'R'): '"light" → "right"',
        ('V', 'B'): '"van" → "ban"',
        ('P', 'B'): '"pen" → "ben"',
        ('F', 'P'): '"fight" → "pight"',
    }

    for i, row in confusions_df.iterrows():
        expected = row['expected_phoneme']
        actual = row['actual_phoneme']
        count = row['count']
        example = examples.get((expected, actual), '—')

        html += f"""
                <tr>
                    <td>{i+1}</td>
                    <td><strong>{expected}</strong></td>
                    <td><strong>{actual}</strong></td>
                    <td>{count}</td>
                    <td style="font-style: italic;">{example}</td>
                </tr>
        """

    html += """
            </tbody>
        </table>
    </div>
    """

    return html


def get_css_styles() -> str:
    """Return CSS styles for the dashboard"""
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
            max-width: 1400px;
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
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }

        .stat-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat-card.metric-accuracy { background: linear-gradient(135deg, #3498db 0%, #2980b9 100%); }
        .stat-card.metric-precision { background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); }
        .stat-card.metric-recall { background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); }
        .stat-card.metric-f1 { background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%); }

        .stat-value {
            font-size: 2.2em;
            font-weight: bold;
            margin-bottom: 5px;
        }

        .stat-label {
            font-size: 0.85em;
            opacity: 0.9;
        }

        .section {
            background: white;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }

        .section h2 {
            color: #333;
            margin-bottom: 20px;
        }

        .table-container {
            overflow-x: auto;
            margin-top: 20px;
        }

        .metrics-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
        }

        .metrics-table th {
            background: #f8f9fa;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            color: #333;
            border-bottom: 2px solid #dee2e6;
        }

        .metrics-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #e9ecef;
        }

        .metrics-table tbody tr:hover {
            background: #f8f9fa;
        }

        .plot-container {
            margin-top: 20px;
        }
    """


def generate_evaluation_dashboard(
    results: List[EvaluationResult],
    output_html: str = 'data/results/evaluation_dashboard.html'
):
    """
    Generate interactive HTML evaluation dashboard.

    Args:
        results: List of EvaluationResult objects
        output_html: Path to output HTML file
    """
    # Aggregate by language
    lang_df = aggregate_results_by_language(results)

    # Generate plots as JSON
    confusion_matrix_json = generate_confusion_matrix_plot(results)
    precision_recall_json = generate_precision_recall_by_language_plot(lang_df)
    error_distribution_json = generate_error_distribution_plot(results)
    per_wer_scatter_json = generate_per_vs_wer_scatter_plot(results)

    # Generate HTML components
    summary_cards = generate_summary_cards(results, lang_df)
    language_table = generate_language_table(lang_df)
    confusions_table = generate_top_confusions_table(results)

    # Complete HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Dashboard - Pronunciation Error Detection</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        {get_css_styles()}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Evaluation Dashboard</h1>
            <p class="subtitle">Pronunciation Error Detection Performance Metrics</p>
        </header>

        {summary_cards}

        <div class="section">
            <h2>Performance Visualizations</h2>

            <div class="plot-container">
                <div id="precision-recall-chart"></div>
            </div>

            <div class="plot-container">
                <div id="error-distribution"></div>
            </div>

            <div class="plot-container">
                <div id="per-wer-scatter"></div>
            </div>

            <div class="plot-container">
                <div id="confusion-matrix"></div>
            </div>
        </div>

        <div class="section">
            <h2>Detailed Results</h2>
            {language_table}
            {confusions_table}
        </div>
    </div>

    <script>
        // Plot 1: Precision-Recall by Language
        const prData = {precision_recall_json};
        if (prData.data && prData.data.length > 0) {{
            Plotly.newPlot('precision-recall-chart', prData.data, prData.layout, {{responsive: true}});
        }}

        // Plot 2: Error Distribution
        const errorData = {error_distribution_json};
        if (errorData.data && errorData.data.length > 0) {{
            Plotly.newPlot('error-distribution', errorData.data, errorData.layout, {{responsive: true}});
        }}

        // Plot 3: PER vs WER Scatter
        const scatterData = {per_wer_scatter_json};
        if (scatterData.data && scatterData.data.length > 0) {{
            Plotly.newPlot('per-wer-scatter', scatterData.data, scatterData.layout, {{responsive: true}});
        }}

        // Plot 4: Confusion Matrix
        const confusionData = {confusion_matrix_json};
        if (confusionData.data && confusionData.data.length > 0) {{
            Plotly.newPlot('confusion-matrix', confusionData.data, confusionData.layout, {{responsive: true}});
        }}
    </script>
</body>
</html>
"""

    # Write to file
    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nEvaluation dashboard generated: {output_html}")
    print(f"  Open in browser: file://{output_path.absolute()}")


if __name__ == '__main__':
    print("Evaluation Dashboard Generator")
    print("Import and call: generate_evaluation_dashboard(results, 'output.html')")
