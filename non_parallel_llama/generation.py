# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import List, Literal, Optional, Tuple, TypedDict
from pathlib import Path
import json
import time
import sys
import os

import torch.nn.functional as F
import torch

from non_parallel_llama.model import ModelArgs, Transformer
from non_parallel_llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]

B_INST, E_INST        = "[INST]", "[/INST]"
B_SYS, E_SYS          = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class Llama:

    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        
        self.model     = model
        self.tokenizer = tokenizer
        
        torch.cuda.set_device(0)

    @torch.inference_mode()
    def generate(
        self,
        prompts      : List[str],
        max_gen_len  : int,
        temperature  : float = 0.6,
        top_p        : float = 0.9,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        
        bsz    = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_size = min(len(t) for t in prompts)
        max_prompt_size = max(len(t) for t in prompts)
        
        total_len       = min(params.max_seq_len, max_gen_len + max_prompt_size)
        tokens          = torch.full((bsz, total_len), 
                                    self.tokenizer.pad_id, dtype = torch.long)
        
        for idx, token in enumerate(prompts):
            tokens[idx, : len(token)] = torch.tensor(token, dtype = torch.long)
        
        input_text_mask = tokens != self.tokenizer.pad_id
        prev_pos        = 0
        eos_reached     = torch.tensor([False] * bsz)
        
        
        for cur_pos in range(min_prompt_size, total_len):
            
            logits = self.model.forward(tokens[:, prev_pos: cur_pos], prev_pos)
            if temperature > 0:
                probs      = torch.softmax(logits / temperature, dim = -1)
                next_token = sample_top_p(probs, top_p)
                
            else:
                next_token = torch.argmax(logits, dim = -1)
                
            next_token = next_token.reshape(-1)
            next_token = torch.where(
                            input_text_mask[:, cur_pos], tokens[:, cur_pos],
                            next_token
                        )
            
            tokens[:, cur_pos] = next_token
            eos_reached       |= (~input_text_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            prev_pos           = cur_pos
            
            if all(eos_reached): break
            
        out_tokens = []
        for idx, token in enumerate(tokens.tolist()):
            
            start = len(prompts[idx])
            token = token[start: len(prompts[idx]) + max_gen_len]

            if self.tokenizer.eos_id in token:
                eos_idx = token.index(self.tokenizer.eos_id)
                token   = token[:eos_idx]
            
            out_tokens.append(token)

        return out_tokens
    
    
    def chat_completion(self, dialogs: List[Dialog], temperature: float = 0.6,
                        top_p: float = 0.9, max_gen_len: Optional[int] = None) -> List[ChatPrediction]:
        
        if max_gen_len is None: max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        
        for dialog in dialogs:
            
            if dialog[0]['role'] != 'system':
                dialog = [
                    {
                        "role" : "system",
                        "content" : DEFAULT_SYSTEM_PROMPT
                    }
                ] + dialog
                
            dialog = [
                {
                    "role"    : dialog[1]["role"],
                    "content" : f'{B_SYS}{dialog[0]["content"]}{E_SYS}{dialog[1]["content"]}' 
                }
            ] + dialog[2:]
            
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(f'{B_INST} {(prompt["content"]).strip()} {E_INST} {(answer["content"]).strip}',
                                          bos = True, eos = True) for prompt, answer in zip(dialog[::2], dialog[1::2])
                ], []
            )
            
            dialog_tokens += self.tokenizer.encode(
                                f'{B_INST} {(dialog[-1]["content"]).strip()} {E_INST}',
                                bos = True, eos = False,
                            )
            
            prompt_tokens.append(dialog_tokens)
            
        generation_tokens = self.generate(prompts     = prompt_tokens, max_gen_len = max_gen_len,
                                          temperature = temperature  , top_p       = top_p)
        
        return [
            {"generation" : {"role"    : "assistant",
                             "content" : self.tokenizer.decode(token)}}
            for token in generation_tokens
        ]
        
def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
