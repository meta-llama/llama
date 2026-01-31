# llama/nlp_metrics.py
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, List, Tuple


_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def tokenize(text: str) -> List[str]:
    # simple, robust tokenization without extra deps
    return _WORD_RE.findall(text.lower())


def ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    if n <= 0:
        return []
    return [tuple(tokens[i : i + n]) for i in range(0, max(0, len(tokens) - n + 1))]


def distinct_n(text: str, n: int = 2) -> float:
    toks = tokenize(text)
    ng = ngrams(toks, n)
    if not ng:
        return 0.0
    return len(set(ng)) / len(ng)


def repetition_rate(text: str, n: int = 3) -> float:
    toks = tokenize(text)
    ng = ngrams(toks, n)
    if not ng:
        return 0.0
    c = Counter(ng)
    repeated = sum(v for v in c.values() if v > 1)
    return repeated / len(ng)


def _lcs_len(a: List[str], b: List[str]) -> int:
    # classic DP LCS length, O(len(a)*len(b)) – fine for short eval strings
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def rouge_l_f1(hypothesis: str, reference: str) -> float:
    h = tokenize(hypothesis)
    r = tokenize(reference)
    if not h or not r:
        return 0.0
    lcs = _lcs_len(h, r)
    prec = lcs / len(h)
    rec = lcs / len(r)
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)


def _modified_precision(h: List[str], r: List[str], n: int) -> float:
    h_ngrams = Counter(ngrams(h, n))
    r_ngrams = Counter(ngrams(r, n))
    if not h_ngrams:
        return 0.0
    clipped = {k: min(v, r_ngrams.get(k, 0)) for k, v in h_ngrams.items()}
    return sum(clipped.values()) / sum(h_ngrams.values())


def bleu(hypothesis: str, reference: str, max_n: int = 4, smooth: float = 1e-9) -> float:
    h = tokenize(hypothesis)
    r = tokenize(reference)
    if not h or not r:
        return 0.0

    # brevity penalty
    bp = 1.0 if len(h) > len(r) else math.exp(1 - (len(r) / max(1, len(h))))

    # geometric mean of modified n-gram precisions (with smoothing)
    log_p_sum = 0.0
    for n in range(1, max_n + 1):
        p = _modified_precision(h, r, n)
        log_p_sum += math.log(max(p, smooth))

    return bp * math.exp(log_p_sum / max_n)


@dataclass
class NLPMetrics:
    bleu4: float
    rouge_l_f1: float
    distinct2: float
    repetition3: float

    def pretty(self) -> str:
        return (
            f"BLEU-4: {self.bleu4:.3f} | "
            f"ROUGE-L(F1): {self.rouge_l_f1:.3f} | "
            f"Distinct-2: {self.distinct2:.3f} | "
            f"Repetition-3: {self.repetition3:.3f}"
        )


def evaluate(generated: str, reference: str) -> NLPMetrics:
    return NLPMetrics(
        bleu4=bleu(generated, reference, max_n=4),
        rouge_l_f1=rouge_l_f1(generated, reference),
        distinct2=distinct_n(generated, n=2),
        repetition3=repetition_rate(generated, n=3),
    )
