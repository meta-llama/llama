# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel(seed: int) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(seed)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.7,
    # top_p: float = 0.95,
    top_p: float = 0.0,
    top_k: int = 40,
    repetition_penalty: float = (1 / 0.85),
    max_seq_len: int = 512,
    max_gen_len: int = 256,
    max_batch_size: int = 32,
    seed: int = 1,
    count: int = 5,
):
    local_rank, world_size = setup_model_parallel(seed)
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    print("\n")
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(json.dumps(dict(
        seed=seed,
        temp=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        max_seq_len=max_seq_len,
        max_gen_len=max_gen_len,
    )))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt

        # "I believe the meaning of life is",
        # "Simply put, the theory of relativity states that",
        # "Building a website can be done in a few simple steps:\n1.",
        # "Here's how to build it in a few simple steps:\n1.",

        "This is Captain Jean-Luc Picard",
        "I am Lieutenant Commander Data",
        "The Klingons are attacking",

#         # Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
#         """Tweet: "I hate it when my phone battery dies."
# Sentiment: Negative
# ###
# Tweet: "My day has been 👍"
# Sentiment: Positive
# ###
# Tweet: "This is the link to the article"
# Sentiment: Neutral
# ###
# Tweet: "This new music video was incredibile"
# Sentiment:""",
#         """Translate English to French:
#
# sea otter => loutre de mer
#
# peppermint => menthe poivrée
#
# plush girafe => girafe peluche
#
# cheese =>""",
    ]
    i = 0
    while i < count or count <= 0:
        i += 1
        for prompt in prompts:
            print(f"\n============== sample {i} =================\n")
            width = 0
            def callback(text):
                nonlocal width
                text = text.replace('\n', '\n\n')
                chars = []
                for i, c in enumerate(text):
                    if c == ' ' and width >= 60:
                        chars.append('\n')
                        width = 0
                    else:
                        width += 1
                        chars.append(c)
                        if c == '\n':
                            width = 0
                text = ''.join(chars)
                print(text, end='', flush=True)
            text, = generator.generate(
                [prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, top_k=top_k, repetition_penalty=repetition_penalty, token_callback=callback,
            )


if __name__ == "__main__":
    fire.Fire(main)
