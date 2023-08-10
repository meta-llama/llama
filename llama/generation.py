# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer
from llama.xla_model_parallel import get_model_parallel_rank, get_model_parallel_world_size, set_g_group

USE_CUDA = os.environ.get('USE_CUDA', False)

# Some how xla init will slow down the CUDA speed.
if not USE_CUDA:
    import torch_xla.core.xla_model as xm
    import torch_xla.experimental.xla_sharding as xs
    from torch_xla import runtime as xr
    import numpy as np

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

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""


class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        dynamo: bool = True,
        spmd: bool = True,
    ) -> "Llama":
        # if not model_parallel_is_initialized():
        #     if model_parallel_size is None:
        #         model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        #     initialize_model_parallel(model_parallel_size)

        # seed must be the same in all processes
        if USE_CUDA:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = '12356'
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group("nccl", rank=int(os.environ.get("RANK", 0)), world_size=int(os.environ.get("WORLD_SIZE", 1)))
            set_g_group()
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(local_rank)
        else:
            device = xm.xla_device()
            xm.set_rng_state(1, device=device)
        torch.manual_seed(1)

        rank = get_model_parallel_rank()
        model_parallel_size = get_model_parallel_world_size()

        if rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        if len(checkpoints) > 0:
            assert model_parallel_size == len(
                checkpoints
            ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
            ckpt_path = checkpoints[rank]
            checkpoint = torch.load(ckpt_path, map_location="cpu")
        else:
            print(f"no checkpoint files found in {ckpt_dir}, init model without loading checkpoint.")
            checkpoint = None
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        if USE_CUDA:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)
        model = Transformer(model_args)
        if checkpoint:
            model.load_state_dict(checkpoint, strict=False)
        model = model.to(device)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer, device, dynamo, spmd)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, device: torch.device, dynamo: bool = True, spmd: bool = True):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._generate_one_token_fn = self._generate_one_token

        if spmd:
            num_devices = xr.global_runtime_device_count()  # updated way to get device count
            device_ids = np.arange(num_devices)
            x_dim = 2 # hard-coded for v5
            yz_dim = 4 # hard-coded for v5

            # manually shard the kv cache
            four_d_mesh = xs.Mesh(device_ids, (1, 1, x_dim, yz_dim))
            for layer in model.layers:
                xs.mark_sharding(layer.attention.cache_k, four_d_mesh, (0, 1, 2, None))
                xs.mark_sharding(layer.attention.cache_v, four_d_mesh, (0, 1, 2, None))

            col_mesh = xs.Mesh(device_ids, (1, num_devices))
            row_mesh = xs.Mesh(device_ids, (num_devices, 1))
            two_d_mesh = xs.Mesh(device_ids, (x_dim, yz_dim))
            two_d_mesh_transpose = xs.Mesh(device_ids, (yz_dim, x_dim))

            for name, layer in model.named_modules():
                if 'tok_embeddings' in name:
                    xs.mark_sharding(layer.weight, row_mesh, (0, 1))
                if 'attention.' in name:
                    if 'wo' in name:
                        xs.mark_sharding(layer.weight, two_d_mesh_transpose, (0, 1))
                    else:
                        xs.mark_sharding(layer.weight, two_d_mesh, (0, 1))
                if 'feed_forward.' in name:
                    if 'w2' in name:
                        xs.mark_sharding(layer.weight, two_d_mesh_transpose, (0, 1))
                    else:
                        xs.mark_sharding(layer.weight, two_d_mesh, (0, 1))
                if 'output' in name:
                    xs.mark_sharding(layer.weight, col_mesh, (0, 1))

        if dynamo:
            if USE_CUDA:
                # Inductor errors out when compiles _generate_one_token_fn.
                # TODO(alanwaketan): figure out why.
                self.model = torch.compile(self.model, fullgraph=True)
            else:
                self._generate_one_token_fn = torch.compile(
                    self._generate_one_token_fn,
                    backend="torchxla_trace_once",
                    fullgraph=True)
            
    def _generate_one_token(self, tokens, input_tokens, input_text_mask,
                            cur_pos_tensor, input_pos_tensor,
                            output_pos_tensor, temperature_tensor,
                            top_p_tensor, with_temp, logprobs, token_logprobs, eos_reached, pad_id):
        if logprobs:
            full_logits = self.model(input_tokens, input_pos_tensor, None)
            logits = full_logits.index_select(1, output_pos_tensor - input_pos_tensor[0]).squeeze(dim=1)
        else:
            logits = self.model(input_tokens, input_pos_tensor, output_pos_tensor)
        if with_temp:
            probs = torch.softmax(logits / temperature_tensor, dim=-1)
            next_token = sample_top_p(probs, top_p_tensor)
        else:
            next_token = torch.argmax(logits, dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        input_text_mask_tmp = input_text_mask.index_select(1, cur_pos_tensor).squeeze(dim=1)
        tokens_tmp = tokens.index_select(1, cur_pos_tensor).squeeze(dim=1)
        next_token = torch.where(input_text_mask_tmp, tokens_tmp, next_token)
        tokens = tokens.index_copy(1, cur_pos_tensor, next_token.unsqueeze(1))
        if logprobs:
            new_logprobs = -F.cross_entropy(
                input=full_logits.transpose(1, 2),
                target=tokens.index_select(1, input_pos_tensor + 1),
                reduction="none",
                ignore_index=pad_id,
            )
            token_logprobs = token_logprobs.index_copy(1, input_pos_tensor + 1, new_logprobs)
        # prepare for the next iteration
        input_pos_tensor = cur_pos_tensor.unsqueeze(0)
        cur_pos_tensor = cur_pos_tensor + 1
        output_pos_tensor = cur_pos_tensor - 1
        input_tokens = tokens.index_select(1, input_pos_tensor)

        eos_reached = eos_reached | (~input_text_mask_tmp) & (
            next_token == self.tokenizer.eos_id
        )
        
        return tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, token_logprobs, eos_reached

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert min_prompt_len >= 1 and max_prompt_len < params.max_seq_len
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((params.max_batch_size, params.max_seq_len), pad_id, dtype=torch.long)
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long)
        tokens = tokens.to(self.device)
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)
        else:
            token_logprobs = None

        input_text_mask = tokens != pad_id
        eos_reached = torch.tensor([False] * bsz, device=self.device)

        # Passing tensors instead of floats into self._generate_one_token_fn,
        # so that different values would not trigger compilations of new graphs
        temperature_tensor = torch.tensor(float(temperature)).to(self.device)
        top_p_tensor = torch.tensor(float(top_p)).to(self.device)
        with_temp = temperature > 0

        if self.device.type == "xla":
            xm.mark_step()

        decoding_start_time = time.time()
        prev_pos = 0
        buckets = [128, 256, 384, 512]
        while prev_pos < min_prompt_len:
            remaining = min_prompt_len - prev_pos
            section_len = 0
            for bucket in buckets:
                if bucket >= remaining:
                    section_len = bucket
                    break
            if section_len == 0:
                section_len = buckets[-1]

            assert prev_pos + section_len <= params.max_seq_len
            cur_pos = min(min_prompt_len, prev_pos + section_len)
            print(f"Processing prompt pos [{prev_pos}, {prev_pos + section_len}), section length {section_len}, cur_pos {cur_pos}")
            cur_pos_tensor = torch.tensor(cur_pos).to(self.device)
            input_pos_tensor = torch.arange(prev_pos, prev_pos + section_len).to(self.device)
            output_pos_tensor = cur_pos_tensor - 1
            input_tokens = tokens.index_select(1, input_pos_tensor)
            if self.device.type == "xla":
                xm.mark_step()

            tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, token_logprobs, eos_reached \
                = self._generate_one_token_fn(
                    tokens, input_tokens, input_text_mask,
                    cur_pos_tensor, input_pos_tensor,
                    output_pos_tensor, temperature_tensor,
                    top_p_tensor, with_temp, logprobs, token_logprobs, eos_reached, pad_id
                )
            if self.device.type == "xla":
                xm.mark_step()

            prev_pos = cur_pos

        assert cur_pos_tensor.item() == prev_pos + 1 and prev_pos == min_prompt_len
        for cur_pos in range(prev_pos + 1, total_len):
            tokens, input_tokens, cur_pos_tensor, input_pos_tensor, output_pos_tensor, token_logprobs, eos_reached \
                = self._generate_one_token_fn(
                    tokens, input_tokens, input_text_mask,
                    cur_pos_tensor, input_pos_tensor,
                    output_pos_tensor, temperature_tensor,
                    top_p_tensor, with_temp, logprobs, token_logprobs, eos_reached, pad_id
                )
            if self.device.type == "xla":
                xm.mark_step()
            if cur_pos % 10 == 0:
                if all(eos_reached):
                    break

        print(f"Processed prompts with {min_prompt_len} to {max_prompt_len} tokens, and generated {cur_pos_tensor.item() - max_prompt_len} tokens")
        print(f"Totally decoded {total_len - 1} tokens in {time.time() - decoding_start_time:.5f} seconds")

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            if i >= len(prompt_tokens):
                break
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        for dialog in dialogs:
            if dialog[0]["role"] != "system":
                dialog = [
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT,
                    }
                ] + dialog
            dialog = [
                {
                    "role": dialog[1]["role"],
                    "content": B_SYS
                    + dialog[0]["content"]
                    + E_SYS
                    + dialog[1]["content"],
                }
            ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t),
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        return [
            {"generation": {"role": "assistant", "content": self.tokenizer.decode(t)}}
            for t in generation_tokens
        ]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = (probs_sum - probs_sort) > p
    probs_sort = torch.where(mask, 0.0, probs_sort)
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token
