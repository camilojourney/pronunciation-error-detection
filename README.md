# Pronunciation Error Detection (PED)

Turn speech into insight. This repo builds a pipeline that transcribes audio (Whisper), cleans and normalizes text, aligns expected vs. spoken, and detects pronunciation proxies at word/phoneme level.

- Infrastructure recommendations: see `docs/infra-recommendations.md`
- Project report: see `docs/project-report.md`

## Quickstart (text-only demo)
Run a minimal alignment + WER demo without external deps.

```
python3 scripts/run_text_alignment.py --ref "this is a test" --hyp "this test"
```

Expected output shows token ops and WER.

## Structure
```
ped/                # core python package
scripts/            # CLI tools
docs/               # reports and infra guidance
tests/              # unit tests
notebooks/          # experiments (empty placeholder)
```

## Next (audio + phonemes)
- Add Whisper (`faster-whisper`) + ffmpeg for ASR.
- Add `phonemizer` + `g2p-en` for phoneme-level analysis.
- Add DVC to track datasets and artifacts.

## macOS notes
- For ASR later: `brew install ffmpeg`

## License
MIT
