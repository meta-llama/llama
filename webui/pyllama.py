import os
import sys
import torch
import fire
import time
import json

import gradio as gr

from typing import Tuple
from pathlib import Path
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)
    generator = LLaMA(model, tokenizer)
    return generator


def process(prompt: str):
    print("Received:\n", prompt)
    prompts = [prompt]
    results = generator.generate(
        prompts, max_gen_len=256, temperature=temperature, top_p=top_p
    )
    print("Generated:\n", results[0])
    return str(results[0])


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/llama_data/7B")
    parser.add_argument("--tokenizer_path", type=str, default="/llama_data/tokenizer.model")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    ckpt_dir = args.ckpt_dir
    tokenizer_path = args.tokenizer_path
    temperature = 0.8
    top_p = 0.95
    max_seq_len = 512
    max_batch_size = 1

    local_rank, world_size = 0, 1
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    
    demo = gr.Interface(
        fn = process,
        inputs = gr.Textbox(lines=10, placeholder="Your prompt here..."),
        outputs = "text",
    )

    demo.launch(server_name="0.0.0.0")