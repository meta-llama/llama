# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

import os
import torch
import fire
import time
import json

from pathlib import Path


os.environ["BITSANDBYTES_NOWELCOME"] = "1"
from llama import ModelArgs, Transformer, Tokenizer, LLaMA, default_quantize

def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    quantize: bool,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    torch.set_default_tensor_type(torch.HalfTensor)
    print("Allocating transformer on host")
    ctx_tok = default_quantize.set(quantize)
    model = Transformer(model_args)
    default_quantize.reset(ctx_tok)
    key_to_dim = {
        "w1": 0,
        "w2": -1,
        "w3": 0,
        "wo": -1,
        "wq": 0,
        "wk": 0,
        "wv": 0,
        "output": 0,
        "tok_embeddings": -1,
        "ffn_norm": None,
        "attention_norm": None,
        "norm": None,
        "rope": None,
    }

    # ?
    torch.set_default_tensor_type(torch.FloatTensor)

    # load the state dict incrementally, to avoid memory problems
    for i, ckpt in enumerate(checkpoints):
        print(f"Loading checkpoint {i}")
        checkpoint = torch.load(ckpt, map_location="cpu")
        for parameter_name, parameter in model.named_parameters():
            short_name = parameter_name.split(".")[-2]
            if key_to_dim[short_name] is None and i == 0:
                parameter.data = checkpoint[parameter_name]
            elif key_to_dim[short_name] == 0:
                size = checkpoint[parameter_name].size(0)
                parameter.data[size * i : size * (i + 1), :] = checkpoint[
                    parameter_name
                ]
            elif key_to_dim[short_name] == -1:
                size = checkpoint[parameter_name].size(-1)
                parameter.data[:, size * i : size * (i + 1)] = checkpoint[
                    parameter_name
                ]
            del checkpoint[parameter_name]
        del checkpoint

    model.cuda()

    generator = LLaMA(model, tokenizer)
    print(
        f"Loaded in {time.time() - start_time:.2f} seconds with {torch.cuda.max_memory_allocated() / 1024 ** 3:.2f} GiB"
    )
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.8,
    top_p: float = 0.95,
    repetition_penalty_range: int = 1024,
    repetition_penalty_slope: float = 0,
    repetition_penalty: float = 1.15,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
    use_int8: bool = True,
):
    generator = load(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, use_int8)

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        """Welcome.
The following conversation took place at Harvard University.
Former Treasurer Secretary Larry Summers invited Ray Dalio, the founder, chairman and
co-CIO of Bridgewater Associates, the world's largest hedge fund, to discuss Dalio's unique
views on economics.

Dalio:""",
    ]
    results = generator.generate(
        prompts,
        max_gen_len=1024,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty_range=repetition_penalty_range,
        repetition_penalty_slope=repetition_penalty_slope,
        repetition_penalty=repetition_penalty,
    )

    for result in results:
        print(result)
        print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)
