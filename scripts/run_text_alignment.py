#!/usr/bin/env python3
import argparse
import os
import sys

# Add repo root to sys.path when running without installation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ped.text import tokenize, align_tokens  # noqa: E402
from ped.metrics import wer  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Run text-only alignment and WER")
    ap.add_argument("--ref", required=True, help="Reference text")
    ap.add_argument("--hyp", required=True, help="Hypothesis/spoken text")
    args = ap.parse_args()

    ref_toks = tokenize(args.ref)
    hyp_toks = tokenize(args.hyp)
    ops = align_tokens(ref_toks, hyp_toks)
    score = wer(ref_toks, hyp_toks)

    print("Reference tokens:", ref_toks)
    print("Hypothesis tokens:", hyp_toks)
    print(f"WER: {score:.3f}")
    print("Operations:")
    for op in ops:
        print(f"- {op.op:7s} | ref={op.ref} -> hyp={op.hyp}")


if __name__ == "__main__":
    main()
