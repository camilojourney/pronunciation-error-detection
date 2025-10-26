"""
Example 1: Basic Text Alignment
This demonstrates how to use the PED system for text-to-text comparison.
This is useful for understanding alignment before adding ASR.
"""

from ped.pipeline import run_text_pipeline

# Example 1: Perfect match
print("=" * 60)
print("Example 1: Perfect Match")
print("=" * 60)
ref1 = "The quick brown fox jumps over the lazy dog"
hyp1 = "The quick brown fox jumps over the lazy dog"
result1 = run_text_pipeline(ref1, hyp1)
print(f"Reference: {result1.ref}")
print(f"Hypothesis: {result1.hyp}")
print(f"WER: {result1.wer:.2%}")
print(f"Operations: {result1.operations}")
print()

# Example 2: Word deletion (missing words)
print("=" * 60)
print("Example 2: Word Deletion (Non-native speaker skips words)")
print("=" * 60)
ref2 = "I would like to go to the store"
hyp2 = "I like go to store"  # Missing "would", "the"
result2 = run_text_pipeline(ref2, hyp2)
print(f"Reference: {result2.ref}")
print(f"Hypothesis: {result2.hyp}")
print(f"WER: {result2.wer:.2%}")
print(f"Operations: {result2.operations}")
print()

# Example 3: Word substitution (pronunciation error proxy)
print("=" * 60)
print("Example 3: Word Substitution (Mispronunciation)")
print("=" * 60)
ref3 = "She sells seashells by the seashore"
hyp3 = "She sells she shells by the she shore"  # 'seashells' → 'she shells'
result3 = run_text_pipeline(ref3, hyp3)
print(f"Reference: {result3.ref}")
print(f"Hypothesis: {result3.hyp}")
print(f"WER: {result3.wer:.2%}")
print(f"Operations: {result3.operations}")
print()

# Example 4: Complex errors (realistic non-native speech)
print("=" * 60)
print("Example 4: Complex Errors (Realistic Non-Native)")
print("=" * 60)
ref4 = "The weather is beautiful today"
hyp4 = "The wether is beatiful to day"  # Multiple misspellings
result4 = run_text_pipeline(ref4, hyp4)
print(f"Reference: {result4.ref}")
print(f"Hypothesis: {result4.hyp}")
print(f"WER: {result4.wer:.2%}")
print(f"Operations: {result4.operations}")
print()

# Example 5: Insertion (extra words)
print("=" * 60)
print("Example 5: Word Insertion (Extra words)")
print("=" * 60)
ref5 = "I am going home"
hyp5 = "I am am going to home now"  # Extra "am", "to", "now"
result5 = run_text_pipeline(ref5, hyp5)
print(f"Reference: {result5.ref}")
print(f"Hypothesis: {result5.hyp}")
print(f"WER: {result5.wer:.2%}")
print(f"Operations: {result5.operations}")
print()

# Summary statistics
print("=" * 60)
print("Summary Statistics")
print("=" * 60)
examples = [result1, result2, result3, result4, result5]
avg_wer = sum(r.wer for r in examples) / len(examples)
print(f"Average WER across examples: {avg_wer:.2%}")
print(f"Best WER: {min(r.wer for r in examples):.2%}")
print(f"Worst WER: {max(r.wer for r in examples):.2%}")
