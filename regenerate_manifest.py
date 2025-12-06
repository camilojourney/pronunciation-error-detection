"""
Regenerate l2arctic_manifest.json with all 24 speakers.

Creates a hierarchical manifest with:
- Metadata summary
- Per-speaker info with language
- Per-file paths (wav, textgrid, annotation) and reference text
- Tracks which files have human annotations (ground truth)
"""
import json
from pathlib import Path

# Speaker to language mapping
SPEAKER_LANG = {
    # Arabic
    'ABA': 'Arabic',
    'SKA': 'Arabic',
    'YBAA': 'Arabic',
    'ZHAA': 'Arabic',
    # Mandarin
    'BWC': 'Mandarin',
    'LXC': 'Mandarin',
    'NCC': 'Mandarin',
    'TXHC': 'Mandarin',
    # Hindi
    'ASI': 'Hindi',
    'RRBI': 'Hindi',
    'SVBI': 'Hindi',
    'TNI': 'Hindi',
    # Korean
    'HJK': 'Korean',
    'HKK': 'Korean',
    'YDCK': 'Korean',
    'YKWK': 'Korean',
    # Spanish
    'EBVS': 'Spanish',
    'ERMS': 'Spanish',
    'MBMPS': 'Spanish',
    'NJS': 'Spanish',
    # Vietnamese
    'HQTV': 'Vietnamese',
    'PNV': 'Vietnamese',
    'THV': 'Vietnamese',
    'TLV': 'Vietnamese',
}

BASE_DIR = 'l2arctic_release_v5'


def load_prompts(prompts_file: Path) -> dict:
    """Load reference texts from PROMPTS file."""
    prompts = {}
    with open(prompts_file) as f:
        for line in f:
            line = line.strip()
            if line and '(' in line:
                # Format: ( arctic_a0001 "Author of the danger trail..." )
                parts = line.split('"')
                if len(parts) >= 2:
                    file_id = line.split()[1]
                    text = parts[1]
                    prompts[file_id] = text
    return prompts


def build_manifest() -> dict:
    """Build hierarchical manifest with all speaker data."""
    base_path = Path(BASE_DIR)
    prompts = load_prompts(base_path / 'PROMPTS')

    print(f"✓ Loaded {len(prompts)} prompts")

    # Initialize manifest structure
    manifest = {
        "metadata": {
            "base_dir": BASE_DIR,
            "total_speakers": 0,
            "total_files": 0,  # Only annotated files (ground truth)
            "languages": list(set(SPEAKER_LANG.values())),
            "speakers_per_language": {}
        },
        "speakers": {}
    }

    # Count speakers per language
    for lang in manifest["metadata"]["languages"]:
        speakers = [s for s, l in SPEAKER_LANG.items() if l == lang]
        manifest["metadata"]["speakers_per_language"][lang] = speakers

    total_files = 0

    # Process each speaker
    for speaker_id in sorted(SPEAKER_LANG.keys()):
        speaker_path = base_path / speaker_id

        if not speaker_path.exists():
            print(f"⚠️  WARNING: {speaker_path} does not exist")
            continue

        # Get all file types
        wav_dir = speaker_path / 'wav'
        textgrid_dir = speaker_path / 'textgrid'
        annotation_dir = speaker_path / 'annotation'

        wav_files = set(f.stem for f in wav_dir.glob('*.wav')) if wav_dir.exists() else set()
        textgrid_files = set(f.stem for f in textgrid_dir.glob('*.TextGrid')) if textgrid_dir.exists() else set()
        annotation_files = set(f.stem for f in annotation_dir.glob('*.TextGrid')) if annotation_dir.exists() else set()

        # Build speaker entry - ONLY include files with annotations (ground truth)
        speaker_data = {
            "native_language": SPEAKER_LANG[speaker_id],
            "total_files": len(annotation_files),  # Only count annotated files
            "files": {}
        }

        # Process only files that have annotations (ground truth)
        for file_id in sorted(annotation_files):
            file_entry = {
                "reference_text": prompts.get(file_id, ""),
                "wav": f"{BASE_DIR}/{speaker_id}/wav/{file_id}.wav",
                "textgrid": f"{BASE_DIR}/{speaker_id}/textgrid/{file_id}.TextGrid" if file_id in textgrid_files else None,
                "annotation": f"{BASE_DIR}/{speaker_id}/annotation/{file_id}.TextGrid",
            }
            speaker_data["files"][file_id] = file_entry

        manifest["speakers"][speaker_id] = speaker_data
        total_files += len(annotation_files)

        print(f"  {speaker_id} ({SPEAKER_LANG[speaker_id]}): {len(annotation_files)} annotated files")

    # Update metadata
    manifest["metadata"]["total_speakers"] = len(manifest["speakers"])
    manifest["metadata"]["total_files"] = total_files

    return manifest


def main():
    print("=" * 60)
    print("L2-ARCTIC MANIFEST GENERATOR")
    print("=" * 60)

    manifest = build_manifest()

    # Save manifest
    output_path = Path('data/processed/l2arctic_manifest.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Print summary
    meta = manifest["metadata"]
    print("\n" + "=" * 60)
    print("📊 MANIFEST SUMMARY (ANNOTATED FILES ONLY)")
    print("=" * 60)
    print(f"  Total speakers: {meta['total_speakers']}")
    print(f"  Total annotated files: {meta['total_files']:,} (ground truth)")
    print(f"\n  By language:")
    for lang, speakers in sorted(meta['speakers_per_language'].items()):
        speaker_files = sum(manifest['speakers'][s]['total_files'] for s in speakers)
        print(f"    {lang}: {len(speakers)} speakers, {speaker_files} annotated files")

    print(f"\n✓ Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
