import math

from llama.nlp_metrics import (
    bleu,
    distinct_n,
    evaluate,
    repetition_rate,
    rouge_l_f1,
)


def test_bleu_smoke():
    score = bleu("the cat sat on the mat", "the cat sat on the mat")
    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)


def test_rouge_l_f1_smoke():
    score = rouge_l_f1("the cat sat on the mat", "the cat sat on the mat")
    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)


def test_distinct_n_smoke():
    score = distinct_n("one two three four five", n=2)
    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)


def test_repetition_rate_smoke():
    score = repetition_rate("repeat repeat repeat repeat", n=2)
    assert 0.0 <= score <= 1.0
    assert not math.isnan(score)


def test_evaluate_smoke():
    metrics = evaluate("the cat sat on the mat", "the cat sat on the mat")
    assert 0.0 <= metrics.bleu4 <= 1.0
    assert 0.0 <= metrics.rouge_l_f1 <= 1.0
    assert 0.0 <= metrics.distinct2 <= 1.0
    assert 0.0 <= metrics.repetition3 <= 1.0
