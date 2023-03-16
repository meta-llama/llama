# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import re

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


def load(ckpt_dir: str, tokenizer_path: str, local_rank: int, world_size: int) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert (
        world_size == len(checkpoints)
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(max_seq_len=1350, max_batch_size=32, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator

def generate(prompt: str, generator, temperature: float = 0.8, top_p: float = 0.95):
    g = generator
    results = g.generate(prompt, max_gen_len=5, temperature=temperature, top_p=top_p)
    return results

def evaluate_sat(sat, g):
    df = sat
    answers = []
    score = 0
    results = []
    q = ['Answer this question-'+str(i) for i in df['text']]
    
    # generate answers for the questions in the sat dataset in batches of 3 questions
    for i in range(0, len(q), 3):
        # skip questions 108 to 110
        if i == 108:
            continue
        ans = generate(q[i:i+3], g, temperature=0.8, top_p=0.95)
        print(ans)
        answers.extend(ans)
    
    for i in range(len(answers)):
        x = answers[i].replace('<pad>', '').replace('</s>', '').replace('.', '').replace('?', '')
        match = re.search(r"Answer: ([A-Z])", x)
        if match:
            a = match.group(1)
            results.append({'question': df['text'][i], 'answer': a, 'correct_answer': df['answer'][i]})
            try:
                if a == df['answer'][i]:
                    score += 1
            except:
                print("error at ", i)
                pass

    print("Score: ", score/len(answers))
    # print(results)
    return results

def main(ckpt_dir: str, tokenizer_path: str, temperature: float = 0.8, top_p: float = 0.95):
    
    print(ckpt_dir)
    print(tokenizer_path)
    print(temperature)
    print(top_p)

    import pandas as pd

    # Read train data from parquet file in data/train-00000-of-00001-be16864a4346f8b0.parquet
    train = pd.read_parquet('sat/train-00000-of-00001-be16864a4346f8b0.parquet')

    # Read test data from parquet file in data/test-00000-of-00001-8026e2bb5cef708b.parquet
    test = pd.read_parquet('sat/test-00000-of-00001-8026e2bb5cef708b.parquet')

    # Read validation data from parquet file in data/validation-00000-of-00001-6242383510343be0.parquet
    validation = pd.read_parquet('sat/validation-00000-of-00001-6242383510343be0.parquet')

    # combine train, test, and validation data into one dataframe called sat
    sat = pd.concat([train, test, validation], ignore_index=True)
    print("length",len(sat))

    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, 'w')

    generator = load(ckpt_dir, tokenizer_path, local_rank, world_size)

    # evaluate the model on the sat dataset
    results = evaluate_sat(sat.head(20), generator)

    # save the results to a json file
    with open('results.json', 'w') as f:
        json.dump(results, f)


if __name__ == "__main__":
    fire.Fire(main)
