# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List

from sentencepiece import SentencePieceProcessor

logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str):
        """
        Initialize the Tokenizer.

        Args:
            model_path (str): Path to the SentencePiece model file.
        """
        # Reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encode the input string.

        Args:
            s (str): Input string to be encoded.
            bos (bool): If True, add the beginning-of-sequence (BOS) token.
            eos (bool): If True, add the end-of-sequence (EOS) token.

        Returns:
            List[int]: Encoded token IDs as a list of integers.
        """
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decode the list of token IDs.

        Args:
            t (List[int]): List of token IDs to be decoded.

        Returns:
            str: Decoded string.
        """
        return self.sp_model.decode(t)
