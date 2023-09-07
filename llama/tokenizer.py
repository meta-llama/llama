# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from typing import List, Optional

from sentencepiece import SentencePieceProcessor


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: Optional[str] = None):

        if model_path is not None:
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.sp_model = SentencePieceProcessor(model_file=model_path)
            logger.info(f"Reloaded SentencePiece model from {model_path}")

            # BOS / EOS / PAD / UNK token IDs
            self.n_words: int = self.sp_model.vocab_size()
            self.bos_id: int = self.sp_model.bos_id()
            self.eos_id: int = self.sp_model.eos_id()
            self.pad_id: int = self.sp_model.pad_id()
            self.unk_id: int = self.sp_model.unk_id()
            logger.info(
                f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
            )
            assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert isinstance(s, str), "Input 's' must be a string"
        try:
            t = self.sp_model.encode(s)
        except Exception as e:
            raise ValueError(f"Error during tokenization: {e}")
        
        # Handle unknown tokens
        t = [token_id if token_id in range(self.n_words) else self.unk_id for token_id in t]
        
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
