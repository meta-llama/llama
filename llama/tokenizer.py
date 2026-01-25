# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from logging import getLogger
from pathlib import Path
from typing import List

from sentencepiece import SentencePieceProcessor


logger = getLogger()


class Tokenizer:
    """tokenizing and encoding/decoding text using SentencePiece or Tiktoken."""
    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a SentencePiece model or Tiktoken model file.

        Args:
            model_path (str): The path to the SentencePiece model file or Tiktoken model file.
        """
        assert os.path.isfile(model_path), f"Tokenizer model not found: {model_path}"
        
        self.sp_model = None
        self.tiktoken_model = None

        if model_path.endswith(".model"):
            self.sp_model = SentencePieceProcessor(model_file=model_path)
            self.n_words: int = self.sp_model.vocab_size()
            self.bos_id: int = self.sp_model.bos_id()
            self.eos_id: int = self.sp_model.eos_id()
            self.pad_id: int = self.sp_model.pad_id()
            logger.info(f"Loaded SentencePiece model from {model_path}")
        else:
            try:
                import tiktoken
                from tiktoken.load import load_tiktoken_bpe
            except ImportError as e:
                logger.error("Tiktoken not installed. Install with: pip install tiktoken")
                raise ImportError(
                    "Tiktoken is required for Llama 3 models. Install with: pip install tiktoken"
                ) from e
            
            mergeable_ranks = load_tiktoken_bpe(model_path)
            
            num_base_tokens = len(mergeable_ranks)
            special_tokens = [
                "<|begin_of_text|>",
                "<|end_of_text|>",
                "<|reserved_special_token_0|>",
                "<|reserved_special_token_1|>",
                "<|finetune_right_pad_id|>",
                "<|step_id|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eom_id|>",
                "<|eot_id|>",
                "<|python_tag|>",
            ]
            reserved_tokens = [
                f"<|reserved_special_token_{2+i}|>"
                for i in range(256 - len(special_tokens))
            ]
            special_tokens.extend(reserved_tokens)
            
            self.tiktoken_model = tiktoken.Encoding(
                name=Path(model_path).name,
                pat_str=r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
                mergeable_ranks=mergeable_ranks,
                special_tokens={
                    token: num_base_tokens + i for i, token in enumerate(special_tokens)
                },
            )
            self.n_words = self.tiktoken_model.n_vocab
            self.bos_id = self.tiktoken_model.encode_single_token("<|begin_of_text|>")
            self.eos_id = self.tiktoken_model.encode_single_token("<|end_of_text|>")
            self.pad_id = self.tiktoken_model.encode_single_token("<|finetune_right_pad_id|>")
            
            logger.info(f"Loaded Tiktoken model from {model_path}")

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.

        Returns:
            List[int]: A list of token IDs.
        """
        if not isinstance(s, str):
            raise TypeError(f"Expected str, got {type(s)}")
        
        t = self.sp_model.encode(s) if self.sp_model else self.tiktoken_model.encode(s)
            
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decodes a list of token IDs into a string.

        Args:
            t (List[int]): The list of token IDs to be decoded.

        Returns:
            str: The decoded string.
        """
        if self.sp_model:
            return self.sp_model.decode(t)
        else:
            return self.tiktoken_model.decode(t)
