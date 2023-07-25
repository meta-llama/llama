# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from typing import Optional

import fire
import os
import torch

from llama import Llama

USE_CUDA = os.environ.get('USE_CUDA', False)

# Some how xla init will slow down the CUDA speed.
if USE_CUDA:
    import torch.multiprocessing as xmp
else:
    import torch_xla.debug.profiler as xp
    import torch_xla.distributed.xla_multiprocessing as xmp


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    if not USE_CUDA:
        server = xp.start_server(9012, only_on_master=False)
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    dialogs = [
        [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
        [
            {"role": "user", "content": "I am going to Paris, what should I see?"},
            {
                "role": "assistant",
                "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
            },
            {"role": "user", "content": "What is so great about #1?"},
        ],
        [
            {"role": "system", "content": "Always answer with Haiku"},
            {"role": "user", "content": "I am going to Paris, what should I see?"},
        ],
        [
            {
                "role": "system",
                "content": "Always answer with emojis",
            },
            {"role": "user", "content": "How to go from Beijing to NY?"},
        ],
    ]
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    for dialog, result in zip(dialogs, results):
        for msg in dialog:
            print(f"{msg['role'].capitalize()}: {msg['content']}\n")
        print(
            f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
        )
        print("\n==================================\n")


def _fn(
    idx,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    if USE_CUDA:
        os.environ['WORLD_SIZE'] = torch.cuda.device_count()
        os.environ['RANK'] = idx
        os.environ['LOCAL_RANK'] = idx
    main(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len)


def mp_main(
    mp: bool,
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_batch_size: int = 4,
    max_gen_len: Optional[int] = None,
):
    if mp:
        if USE_CUDA:
            kwargs = {"nprocs": torch.cuda.device_count(),
                      "join": True}
        else:
            kwargs = {}
        xmp.spawn(_fn,
                  args=(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len), **kwargs)
    else:
        main(ckpt_dir, tokenizer_path, temperature, top_p, max_seq_len, max_batch_size, max_gen_len)


if __name__ == "__main__":
    fire.Fire(mp_main)
