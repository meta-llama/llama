import os
import torch
import fire
import time
import json

from pathlib import Path

from fairscale.nn.model_parallel.initialize import initialize_model_parallel

from genesis_llm import ModelArgs, Transformer, Tokenizer, Genesis


def main(ckpt_dir: str, tokenizer_path: str):
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    torch.distributed.init_process_group("nccl")
    initialize_model_parallel(world_size)
    torch.cuda.set_device(local_rank)

    torch.manual_seed(1)
    is_main = int(local_rank) == 0
    start_time = time.time()
    ckpt_path = sorted(Path(ckpt_dir).glob("*.pth"))[local_rank]
    print(f"Rank {local_rank} loading {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / 'params.json', 'r') as f:
        params = json.loads(f.read())["model"]  # TODO: clean

    params = {x: y for x,y in params.items() if x in ModelArgs.fields()}

    model_args: ModelArgs = ModelArgs(max_seq_len = 1024, max_batch_size=32, **params)
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = Genesis(model, tokenizer)
    if is_main:
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

    prompts = [
        "Today I wrote a ",
        "Making an apple pie is easy, "
    ]

    results = generator.generate(
        prompts,
        max_gen_len=256
    )
    if is_main:
        for prompt, result in zip(prompts, results):
            print(prompt + result)
            print("\n==================================\n")


if __name__ == "__main__":
    fire.Fire(main)