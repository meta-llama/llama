# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import List

import torch

from llama.tokenizer import Tokenizer
from llama.model import Transformer


class LLaMA:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(
        self,
        prompts: List[str],
        max_gen_len: int,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.0, #0.95,
        repetition_penalty: float = (1.0 / 0.85),
        token_callback=None,
    ) -> List[str]:
        bsz = len(prompts)
        params = self.model.params
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]

        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.tokenizer.pad_id).cuda().long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.tokenizer.pad_id
        start_pos = min_prompt_size
        prev_pos = 0
        prev_text = ''
        for cur_pos in range(start_pos, total_len):
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            # repetition penalty from CTRL paper (https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                logits_new = logits.clone()
                batch_size = len(tokens)
                for i in range(batch_size):
                    for token in set(tokens[i].tolist()):
                        # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                        if logits[i, token] < 0:
                            logits_new[i, token] = logits[i, token] * repetition_penalty
                        else:
                            logits_new[i, token] = logits[i, token] / repetition_penalty
                logits = logits_new

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample(probs, top_p=top_p, top_k=top_k)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            if next_token == self.tokenizer.eos_id:
                break
            tokens[:, cur_pos] = next_token
            if token_callback is not None:
                assert len(prompts) == 1
                text, = self.decode(tokens)
                #assert text.startswith(prev_text)
                if not text.startswith(prev_text):
                    # Some kind of bogus token generation; abort early.
                    break
                next_word = text[len(prev_text):]
                prev_text = text
                token_callback(next_word)
            prev_pos = cur_pos

        return self.decode(tokens)

    def decode(self, tokens):
        decoded = []
        for i, t in enumerate(tokens.tolist()):
            t = [token for token in t if token != -1]
            # # cut to max gen len
            # t = t[: len(prompt_tokens[i]) + max_gen_len]
            while self.tokenizer.eos_id in t:
                pos = t.index(self.tokenizer.eos_id)
                t[pos:pos+1] = self.tokenizer.encode('\n<|endoftext|>\n', bos=False, eos=False)
            decoded.append(self.tokenizer.decode(t))
        return decoded

def sample(probs, top_p=0.0, top_k=40):
    if top_k > 0:
        probs_sort, probs_idx = torch.topk(probs, top_k)
    else:
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    if top_p > 0.0:
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        mask = probs_sum - probs_sort > top_p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
