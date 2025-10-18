from typing import List


def wer(ref_tokens: List[str], hyp_tokens: List[str]) -> float:
    """Compute Word Error Rate (S+D+I)/N using dynamic programming.

    Returns 0.0 when ref is empty and hyp is empty; 1.0 if ref empty and hyp non-empty.
    """
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    # DP matrix of edit distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost,  # substitution
            )
    return dp[n][m] / max(1, n)
