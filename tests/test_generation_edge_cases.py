# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

"""Lightweight tests for generation edge cases that do not require model weights."""

import unittest
from unittest.mock import MagicMock

import torch

from llama.generation import Llama, sample_top_p


class TestSampleTopP(unittest.TestCase):
    def test_normal_distribution(self):
        probs = torch.tensor([[0.1, 0.2, 0.7]])
        token = sample_top_p(probs, p=0.9)
        self.assertEqual(token.shape, (1, 1))
        self.assertTrue(0 <= int(token.item()) < 3)

    def test_degenerate_all_zero_probs(self):
        probs = torch.zeros(2, 8)
        token = sample_top_p(probs, p=0.9)
        self.assertEqual(token.shape, (2, 1))

    def test_nan_probs_fallback(self):
        probs = torch.full((1, 4), float("nan"))
        token = sample_top_p(probs, p=0.5)
        self.assertEqual(token.shape, (1, 1))


class TestEmptyBatches(unittest.TestCase):
    def setUp(self):
        self.llama = Llama(model=MagicMock(), tokenizer=MagicMock())
        self.llama.model.params = MagicMock(max_seq_len=128, max_batch_size=4)

    def test_text_completion_empty(self):
        self.assertEqual(self.llama.text_completion([]), [])

    def test_chat_completion_empty(self):
        self.assertEqual(self.llama.chat_completion([]), [])

    def test_chat_completion_empty_dialog(self):
        with self.assertRaises(ValueError):
            self.llama.chat_completion([[]])

    def test_chat_completion_system_without_user(self):
        with self.assertRaises(ValueError):
            self.llama.chat_completion(
                [[{"role": "system", "content": "Be helpful"}]]
            )

    def test_generate_empty(self):
        tokens, logprobs = self.llama.generate([], max_gen_len=8)
        self.assertEqual(tokens, [])
        self.assertIsNone(logprobs)


if __name__ == "__main__":
    unittest.main()
