"""Generate demo data for NLP presentation."""

import json
from pathlib import Path

# Sample pronunciation errors from non-native speakers
DEMO_DATA = [
    {
        "speaker_id": "arabic_01",
        "native_language": "Arabic",
        "reference": "The weather is very nice today",
        "hypothesis": "The weazer is wery nice today",
        "audio_path": "demo/arabic_01.wav"
    },
    {
        "speaker_id": "spanish_01",
        "native_language": "Spanish",
        "reference": "I need to finish my homework",
        "hypothesis": "I need to feenish my homework",
        "audio_path": "demo/spanish_01.wav"
    },
    {
        "speaker_id": "chinese_01",
        "native_language": "Mandarin",
        "reference": "The restaurant is very expensive",
        "hypothesis": "The lestaurant is vely expensive",
        "audio_path": "demo/chinese_01.wav"
    },
    {
        "speaker_id": "korean_01",
        "native_language": "Korean",
        "reference": "Please repeat what you said",
        "hypothesis": "Please lepeat what you said",
        "audio_path": "demo/korean_01.wav"
    },
    {
        "speaker_id": "hindi_01",
        "native_language": "Hindi",
        "reference": "I will visit the museum tomorrow",
        "hypothesis": "I will wisit the museum tomorrow",
        "audio_path": "demo/hindi_01.wav"
    },
    {
        "speaker_id": "arabic_02",
        "native_language": "Arabic",
        "reference": "Can you help me with this problem",
        "hypothesis": "Can you help me wis zis broblem",
        "audio_path": "demo/arabic_02.wav"
    },
    {
        "speaker_id": "spanish_02",
        "native_language": "Spanish",
        "reference": "The book is on the table",
        "hypothesis": "The book is on the tebol",
        "audio_path": "demo/spanish_02.wav"
    },
    {
        "speaker_id": "chinese_02",
        "native_language": "Mandarin",
        "reference": "I really like chocolate ice cream",
        "hypothesis": "I leally like chocolate ice cleam",
        "audio_path": "demo/chinese_02.wav"
    },
    {
        "speaker_id": "korean_02",
        "native_language": "Korean",
        "reference": "The flight leaves at three o'clock",
        "hypothesis": "The plight leaves at three o'clock",
        "audio_path": "demo/korean_02.wav"
    },
    {
        "speaker_id": "hindi_02",
        "native_language": "Hindi",
        "reference": "Where is the nearest pharmacy",
        "hypothesis": "Where is the nearest parmacy",
        "audio_path": "demo/hindi_02.wav"
    },
    {
        "speaker_id": "french_01",
        "native_language": "French",
        "reference": "The meeting starts in five minutes",
        "hypothesis": "Ze meeting starts in five minutes",
        "audio_path": "demo/french_01.wav"
    },
    {
        "speaker_id": "german_01",
        "native_language": "German",
        "reference": "I think this is a good idea",
        "hypothesis": "I sink zis is a good idea",
        "audio_path": "demo/german_01.wav"
    },
    {
        "speaker_id": "japanese_01",
        "native_language": "Japanese",
        "reference": "The library closes at seven",
        "hypothesis": "The library closes at seben",
        "audio_path": "demo/japanese_01.wav"
    },
    {
        "speaker_id": "russian_01",
        "native_language": "Russian",
        "reference": "This is a very important question",
        "hypothesis": "Zis is a verry important kvestion",
        "audio_path": "demo/russian_01.wav"
    },
    {
        "speaker_id": "vietnamese_01",
        "native_language": "Vietnamese",
        "reference": "I want to learn English better",
        "hypothesis": "I want to learn Englih better",
        "audio_path": "demo/vietnamese_01.wav"
    },
]


def main():
    """Generate demo data JSON file."""
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "demo_data.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(DEMO_DATA, f, indent=2, ensure_ascii=False)

    print(f"✓ Generated demo data: {output_file}")
    print(f"  Total samples: {len(DEMO_DATA)}")
    print(f"  Languages: {len(set(item['native_language'] for item in DEMO_DATA))}")


if __name__ == "__main__":
    main()
