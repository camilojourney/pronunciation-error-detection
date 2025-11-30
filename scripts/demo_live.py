#!/usr/bin/env python3
"""Interactive demo script for live demonstration during presentation."""

from ped.pipeline import run_text_pipeline
from ped.errors import get_error_summary
import sys


def print_separator():
    print("=" * 70)


def demo_single_example(reference: str, hypothesis: str, speaker_info: str = ""):
    """Demonstrate error detection on a single example."""
    print_separator()
    if speaker_info:
        print(f"SPEAKER: {speaker_info}")
    print(f"REFERENCE:  {reference}")
    print(f"HYPOTHESIS: {hypothesis}")
    print_separator()

    # Run pipeline
    result = run_text_pipeline(reference, hypothesis, analyze=True)

    # Display results
    print(f"Word Error Rate: {result.wer:.2%}")
    print(f"\nError Analysis:")
    print(f"  Total Errors:   {result.error_analysis.total_errors}")
    print(f"  - Substitutions: {result.error_analysis.substitution_count}")
    print(f"  - Deletions:     {result.error_analysis.deletion_count}")
    print(f"  - Insertions:    {result.error_analysis.insertion_count}")

    if result.error_analysis.common_errors:
        print(f"\nSubstitution Patterns:")
        for ref, hyp, count in result.error_analysis.common_errors[:5]:
            print(f"  '{ref}' → '{hyp}' ({count}x)")

    # Show alignment
    print(f"\nDetailed Alignment:")
    for i, op in enumerate(result.operations):
        if op.op == "equal":
            status = "✓"
            color = ""
        else:
            status = "✗"
            color = ""

        ref_words = " ".join(op.ref) if op.ref else "<none>"
        hyp_words = " ".join(op.hyp) if op.hyp else "<none>"

        print(f"  {status} {op.op:10s} | Ref: {ref_words:20s} | Hyp: {hyp_words:20s}")

    print_separator()
    print()


def main():
    """Run interactive demo."""
    print("\n" + "=" * 70)
    print(" " * 15 + "PRONUNCIATION ERROR DETECTION DEMO")
    print("=" * 70)
    print()

    # Example 1: Arabic speaker
    demo_single_example(
        reference="The weather is very nice today",
        hypothesis="The weazer is wery nice today",
        speaker_info="Arabic Speaker (th→z, v→w substitutions)"
    )

    # Example 2: Spanish speaker
    demo_single_example(
        reference="I need to finish my homework",
        hypothesis="I need to feenish my homework",
        speaker_info="Spanish Speaker (i→ee vowel lengthening)"
    )

    # Example 3: Mandarin speaker
    demo_single_example(
        reference="The restaurant is very expensive",
        hypothesis="The lestaurant is vely expensive",
        speaker_info="Mandarin Speaker (r→l, r→l substitutions)"
    )

    # Example 4: German speaker
    demo_single_example(
        reference="I think this is a good idea",
        hypothesis="I sink zis is a good idea",
        speaker_info="German Speaker (th→z, th→s substitutions)"
    )

    # Interactive mode
    print("\n" + "=" * 70)
    print("INTERACTIVE MODE - Try your own examples!")
    print("=" * 70)
    print("Enter 'quit' to exit\n")

    while True:
        try:
            print("Enter reference text (what should be said):")
            reference = input("> ").strip()

            if reference.lower() == 'quit':
                break

            if not reference:
                continue

            print("\nEnter hypothesis text (what was actually said):")
            hypothesis = input("> ").strip()

            if hypothesis.lower() == 'quit':
                break

            if not hypothesis:
                continue

            print()
            demo_single_example(reference, hypothesis, speaker_info="Custom Example")

        except KeyboardInterrupt:
            print("\n\nExiting demo...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            continue

    print("\nThank you for trying the demo!")


if __name__ == "__main__":
    main()
