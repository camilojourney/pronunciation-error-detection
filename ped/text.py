from dataclasses import dataclass
from typing import List, Tuple
import difflib
import re


def clean_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s']", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(s: str) -> List[str]:
    return clean_text(s).split()


@dataclass
class TokenOp:
    op: str  # 'equal' | 'replace' | 'delete' | 'insert'
    ref: List[str]
    hyp: List[str]


def align_tokens(ref_tokens: List[str], hyp_tokens: List[str]) -> List[TokenOp]:
    sm = difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    ops: List[TokenOp] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        ops.append(TokenOp(tag, ref_tokens[i1:i2], hyp_tokens[j1:j2]))
    return ops
