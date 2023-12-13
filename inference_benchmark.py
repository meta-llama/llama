import torch
from torch.utils.data import DataLoader
import time
from datasets import load_dataset

### Setup ###
BATCH_SIZE = 1
BATCH_COUNT = 5
NUM_WORKERS = 1

# Manual seed for reproducatibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

from llama import Llama
from typing import List

prompts: List[str] = [
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

def get_device():
    return torch.device(DEVICE_CUDA if torch.cuda.is_available() else DEVICE_CPU)

def get_data_loader(num_workers=1):
    dataset = load_dataset("HuggingFaceH4/no_robots")
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )
    return dataloader


def get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator


def run_benchmark(dataloader, model):
    load_time_per_batch = torch.zeros(BATCH_COUNT)
    inference_time_per_batch = torch.zeros(BATCH_COUNT)
    total_time_per_batch = torch.zeros(BATCH_COUNT)
    
    device = get_device()
    model.to(device)
    print("Working on device: {}".format(device))
    
    
    for batch_idx in range(BATCH_COUNT):
        print("Starting BATCH {} of {}".format(batch_idx + 1, BATCH_COUNT))
        (output, load_time, inference_time), batch_time = measure_runtime(run_batch_inference,
                                                              dataloader,
                                                              model)
        load_time_per_batch[batch_idx] = load_time
        inference_time_per_batch[batch_idx] = inference_time
        total_time_per_batch[batch_idx] = batch_time

        print("Finished Batch {} of {}".format(batch_idx + 1, BATCH_COUNT))
        print("Batch load time: {}".format(load_time))
        print("Batch inference time: {}".format(inference_time))
        print("Batch total time: {}".format(batch_time))
    return model, load_time_per_batch, inference_time_per_batch, total_time_per_batch


def measure_runtime(func, *func_args):
    start = time.perf_counter()
    result = func(*func_args)
    end = time.perf_counter()
    elapsed = end - start
    return result, elapsed


def run_batch_inference(dataloader, model):
    x, load_time = measure_runtime(
        __get_next_batch, dataloader)

    device = get_device()
    x = x.to(device)
    # y = y.to(device)

    output, inference_time = measure_runtime(
        inference,
        model,
        x)
    
    return output, load_time, inference_time

def inference(
    generator: Llama,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: int = 64,
):
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

def __get_next_batch(dataloader):
    return next(iter(dataloader))


def benchmark():
    print("Starting up...")

    print("Building data loaders...")
    data_loader = get_data_loader()

    print("Initializing Model...")
    net = get_model()

    print("Running inference benchmark...\n")
    _, load, inference, total = run_benchmark(data_loader, net)

    print("Results...")
    print("Data-loading times")
    print("> per epoch: ", load)
    print("> average: ", torch.mean(load))
    print("\nInference time for each epoch")
    print("> per epoch", inference)
    print("> average", torch.mean(inference))
    print("\nTotal time for each epoch")
    print("> per epoch", total)
    print("> average", torch.mean(total))

if __name__ == "__main__":
    benchmark()