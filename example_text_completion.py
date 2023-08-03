# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
import os

import fire

from llama import Llama

import torch
import torch.distributed as dist


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
):
    torch.cuda.empty_cache()

    # This is to be able to run directly from PyCharm
    os.environ["MASTER_ADDR"] = "DESKTOP-FONP7QD"
    os.environ["MASTER_PORT"] = "50546"
    #

    # Init with gloo
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    global_free_bytes, total_gpu_mem = get_mem_info()
    print(f"Cuda support: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}, version: {torch.version.cuda}, mem_get_info: ({global_free_bytes} MBytes, {total_gpu_mem} MBytes)")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    prompts = [
        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,
        
        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:
        
        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
    ]
    for i in range(1, 1000):
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        for prompt, result in zip(prompts, results):
            print(prompt)
            print(f"> {result['generation']}")
            print("\n==================================\n")


def get_mem_info():
    global_free_bytes, total_gpu_mem = torch.cuda.mem_get_info()
    global_free_bytes = global_free_bytes / 1024 / 1024
    total_gpu_mem = total_gpu_mem / 1024 / 1024
    return global_free_bytes, total_gpu_mem


if __name__ == "__main__":
   fire.Fire(main)
