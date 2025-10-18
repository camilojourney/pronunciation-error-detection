from ped.text import tokenize, align_tokens
from ped.metrics import wer


def test_tokenize_basic():
    assert tokenize("This, is a TEST!") == ["this", "is", "a", "test"]


def test_alignment_and_wer():
    ref = tokenize("this is a test")
    hyp = tokenize("this test")
    ops = align_tokens(ref, hyp)
    # Expect deletions for 'is' and 'a'
    tags = [o.op for o in ops]
    assert "delete" in tags or "replace" in tags
    assert abs(wer(ref, hyp) - (2 / 4)) < 1e-6
