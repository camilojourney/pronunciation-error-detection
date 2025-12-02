"""
Process L2-ARCTIC audio files with Whisper ASR and intelligibility analysis.

This script:
1. Loads the L2-ARCTIC manifest
2. Processes audio files with Whisper ASR
3. Analyzes errors with intelligibility classification
4. Saves results to CSV files
"""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

from analysis_utils import (
    run_audio_pipeline,
    get_speaker_language_mapping,
    analyze_intelligibility
)


def process_l2arctic_sample(
    manifest_path='data/processed/l2arctic_manifest.json',
    selected_speakers=None,
    sample_size=50,
    model_size='base',
    output_dir='data/results'
):
    """
    Process L2-ARCTIC audio files and analyze pronunciation errors.

    Args:
        manifest_path: Path to L2-ARCTIC manifest JSON
        selected_speakers: List of speaker IDs to process (None = all)
        sample_size: Number of files per speaker
        model_size: Whisper model size ('tiny', 'base', 'small', 'medium')
        output_dir: Directory to save results
    """

    print("=" * 80)
    print("L2-ARCTIC PRONUNCIATION ERROR ANALYSIS")
    print("=" * 80)

    # Load manifest
    print(f"\n📁 Loading manifest from: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"✓ Loaded {len(manifest):,} audio files")

    # Get speaker-language mapping
    speaker_lang = get_speaker_language_mapping()

    # Filter by selected speakers if specified
    if selected_speakers:
        manifest = [item for item in manifest if item['speaker_id'] in selected_speakers]
        print(f"✓ Filtered to {len(selected_speakers)} speakers: {', '.join(selected_speakers)}")

    # Limit to sample_size per speaker
    speaker_counts = {}
    filtered_manifest = []

    for item in manifest:
        speaker_id = item['speaker_id']
        if speaker_id not in speaker_counts:
            speaker_counts[speaker_id] = 0

        if speaker_counts[speaker_id] < sample_size:
            filtered_manifest.append(item)
            speaker_counts[speaker_id] += 1

    manifest = filtered_manifest
    total_files = len(manifest)

    print(f"\n📊 Processing plan:")
    print(f"   Total files to process: {total_files}")
    print(f"   Sample size per speaker: {sample_size}")
    print(f"   Whisper model: {model_size}")

    for speaker_id, count in sorted(speaker_counts.items()):
        lang = speaker_lang.get(speaker_id, 'Unknown')
        print(f"   {speaker_id} ({lang}): {count} files")

    # Confirm processing
    print(f"\n⏱️  Estimated time: ~{total_files * 4 / 60:.1f} minutes")
    response = input("\nProceed with processing? (y/n): ")

    if response.lower() != 'y':
        print("❌ Processing cancelled")
        return

    # Process audio files
    results = []
    errors = []

    print(f"\n🎙️  Processing audio files...")

    for item in tqdm(manifest, desc="Processing"):
        try:
            # Run ASR pipeline
            result = run_audio_pipeline(
                audio_path=item['audio_path'],
                ref_text=item['reference_text'],
                model_size=model_size
            )

            # Intelligibility analysis
            intel = analyze_intelligibility(result.error_analysis)

            # Store results
            results.append({
                'speaker_id': item['speaker_id'],
                'native_language': item['native_language'],
                'file_id': item['file_id'],
                'reference': item['reference_text'],
                'hypothesis': result.hyp,
                'wer': result.wer,
                'total_errors': result.error_analysis.total_errors,
                'substitutions': result.error_analysis.substitution_count,
                'deletions': result.error_analysis.deletion_count,
                'insertions': result.error_analysis.insertion_count,
                'critical_errors': intel.total_critical_errors,
                'high_impact_errors': len(intel.high_impact_errors),
                'medium_impact_errors': len(intel.medium_impact_errors),
                'accent_features': len(intel.low_impact_errors)
            })

        except Exception as e:
            errors.append({
                'file_id': item['file_id'],
                'error': str(e)
            })
            print(f"\n⚠️  Error processing {item['file_id']}: {e}")

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_path = f"{output_dir}/speaker_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Saved detailed results to: {results_path}")

    # Generate aggregate statistics
    print("\n" + "=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80)

    print(f"\n📊 Overall:")
    print(f"   Files processed: {len(results_df)}")
    print(f"   Average WER: {results_df['wer'].mean():.2%}")
    print(f"   Average critical errors: {results_df['critical_errors'].mean():.2f}")
    print(f"   Average accent features: {results_df['accent_features'].mean():.2f}")

    # By language
    print(f"\n📊 By Language:")
    lang_stats = results_df.groupby('native_language').agg({
        'wer': 'mean',
        'critical_errors': 'mean',
        'accent_features': 'mean'
    }).round(3)
    print(lang_stats)

    # Save language statistics
    lang_stats_path = f"{output_dir}/language_statistics.csv"
    lang_stats.to_csv(lang_stats_path)
    print(f"\n✓ Saved language statistics to: {lang_stats_path}")

    # By speaker
    print(f"\n📊 By Speaker:")
    speaker_stats = results_df.groupby('speaker_id').agg({
        'wer': 'mean',
        'critical_errors': 'mean',
        'accent_features': 'mean',
        'native_language': 'first'
    }).round(3)
    print(speaker_stats)

    # Save speaker statistics
    speaker_stats_path = f"{output_dir}/speaker_statistics.csv"
    speaker_stats.to_csv(speaker_stats_path)
    print(f"\n✓ Saved speaker statistics to: {speaker_stats_path}")

    # Extract error details
    print(f"\n📊 Extracting error patterns...")
    error_details = []

    for item in manifest[:len(results)]:
        result = run_audio_pipeline(
            audio_path=item['audio_path'],
            ref_text=item['reference_text'],
            model_size=model_size
        )
        intel = analyze_intelligibility(result.error_analysis)

        # High impact errors
        for ref, hyp, explanation in intel.high_impact_errors:
            error_details.append({
                'speaker_id': item['speaker_id'],
                'native_language': item['native_language'],
                'file_id': item['file_id'],
                'error_type': 'HIGH_IMPACT',
                'reference_word': ref,
                'hypothesis_word': hyp,
                'explanation': explanation
            })

        # Medium impact errors
        for ref, hyp, explanation in intel.medium_impact_errors:
            error_details.append({
                'speaker_id': item['speaker_id'],
                'native_language': item['native_language'],
                'file_id': item['file_id'],
                'error_type': 'MEDIUM_IMPACT',
                'reference_word': ref,
                'hypothesis_word': hyp,
                'explanation': explanation
            })

    # Save error details
    if error_details:
        error_details_df = pd.DataFrame(error_details)
        error_details_path = f"{output_dir}/error_details.csv"
        error_details_df.to_csv(error_details_path, index=False)
        print(f"✓ Saved error details to: {error_details_path}")

        # Most common substitutions
        print(f"\n📊 Most Common HIGH IMPACT Substitutions:")
        high_impact = error_details_df[error_details_df['error_type'] == 'HIGH_IMPACT']
        if len(high_impact) > 0:
            substitution_counts = high_impact.groupby(['reference_word', 'hypothesis_word']).size()
            top_substitutions = substitution_counts.sort_values(ascending=False).head(10)
            for (ref, hyp), count in top_substitutions.items():
                print(f"   '{ref}' → '{hyp}': {count} times")

    print("\n" + "=" * 80)
    print("✅ PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}/")
    print(f"  - speaker_results.csv")
    print(f"  - language_statistics.csv")
    print(f"  - speaker_statistics.csv")
    if error_details:
        print(f"  - error_details.csv")

    if errors:
        print(f"\n⚠️  {len(errors)} files failed to process")


if __name__ == "__main__":
    # Default: process 5 speakers with 50 utterances each
    selected_speakers = ['ABA', 'BWC', 'ASI', 'HJK', 'EBVS']

    # You can also process all speakers by passing None
    # selected_speakers = None

    process_l2arctic_sample(
        selected_speakers=selected_speakers,
        sample_size=50,
        model_size='base'
    )
