"""Analyze demo pronunciation data and generate results for Quarto report."""

import json
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd
from ped.pipeline import run_text_pipeline
from ped.errors import get_error_summary


def analyze_demo_data():
    """Analyze all demo data and generate comprehensive results."""

    # Load demo data
    data_file = Path("data/processed/demo_data.json")
    with open(data_file) as f:
        demo_data = json.load(f)

    # Process each sample
    results = []
    all_errors = []

    for item in demo_data:
        result = run_text_pipeline(
            ref_text=item["reference"],
            hyp_text=item["hypothesis"],
            analyze=True
        )

        # Extract error info
        error_analysis = result.error_analysis
        error_summary = get_error_summary(error_analysis)

        results.append({
            "speaker_id": item["speaker_id"],
            "native_language": item["native_language"],
            "reference": item["reference"],
            "hypothesis": item["hypothesis"],
            "wer": result.wer,
            "total_errors": error_summary["total_errors"],
            "insertions": error_summary["insertions"],
            "deletions": error_summary["deletions"],
            "substitutions": error_summary["substitutions"],
            "error_rate": error_summary["error_rate"]
        })

        # Collect all errors for pattern analysis
        for error in error_analysis.errors:
            all_errors.append({
                "speaker_id": item["speaker_id"],
                "native_language": item["native_language"],
                "error_type": error.error_type,
                "reference_word": error.reference_word,
                "hypothesis_word": error.hypothesis_word,
                "position": error.position
            })

    # Save results
    results_dir = Path("data/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save speaker-level results
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_dir / "speaker_results.csv", index=False)

    # Save error details
    errors_df = pd.DataFrame(all_errors)
    errors_df.to_csv(results_dir / "error_details.csv", index=False)

    # Calculate aggregate statistics
    stats = {
        "total_speakers": len(results),
        "total_languages": len(set(item["native_language"] for item in demo_data)),
        "avg_wer": float(results_df["wer"].mean()),
        "avg_errors_per_speaker": float(results_df["total_errors"].mean()),
        "total_substitutions": int(results_df["substitutions"].sum()),
        "total_deletions": int(results_df["deletions"].sum()),
        "total_insertions": int(results_df["insertions"].sum()),
    }

    # Error patterns by language
    language_patterns = defaultdict(lambda: {"substitutions": [], "deletions": [], "insertions": []})

    for error in all_errors:
        lang = error["native_language"]
        error_type = error["error_type"]
        if error_type == "substitution":
            language_patterns[lang]["substitutions"].append(
                (error["reference_word"], error["hypothesis_word"])
            )
        elif error_type == "deletion":
            language_patterns[lang]["deletions"].append(error["reference_word"])
        elif error_type == "insertion":
            language_patterns[lang]["insertions"].append(error["hypothesis_word"])

    # Most common substitution patterns
    substitution_patterns = Counter()
    for error in all_errors:
        if error["error_type"] == "substitution":
            substitution_patterns[(error["reference_word"], error["hypothesis_word"])] += 1

    # Save aggregate statistics
    with open(results_dir / "aggregate_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Save substitution patterns
    patterns_data = [
        {"reference": ref, "hypothesis": hyp, "count": count}
        for (ref, hyp), count in substitution_patterns.most_common(20)
    ]
    pd.DataFrame(patterns_data).to_csv(results_dir / "substitution_patterns.csv", index=False)

    # Error distribution by language
    lang_error_counts = errors_df.groupby(["native_language", "error_type"]).size().reset_index(name="count")
    lang_error_counts.to_csv(results_dir / "errors_by_language.csv", index=False)

    print(f"✓ Analysis complete!")
    print(f"  Total speakers analyzed: {stats['total_speakers']}")
    print(f"  Average WER: {stats['avg_wer']:.3f}")
    print(f"  Total errors: {len(all_errors)}")
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - speaker_results.csv")
    print(f"  - error_details.csv")
    print(f"  - aggregate_stats.json")
    print(f"  - substitution_patterns.csv")
    print(f"  - errors_by_language.csv")

    return stats


if __name__ == "__main__":
    analyze_demo_data()
