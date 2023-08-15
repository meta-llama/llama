# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from contextlib import nullcontext
import fire
import time
import torch
import torch.profiler

from llama import Llama


def benchmark(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int = 128,
    warmup_iterations: int = 2,
    test_iterations: int = 5,
    use_cuda_graph : bool = True,
    profile : bool = False,
):
    # Build the Llama generator
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
    )

    # Sample prompt for warmup and benchmarking
    prompt = "The theory of everything is"

    # Warmup Iterations
    for i in range(warmup_iterations):
        print(f"Warmup iteration {i}")
        _ = generator.text_completion([prompt], use_cuda_graph=use_cuda_graph)

    # Ensure GPU operations have completed
    torch.cuda.synchronize()

    # Benchmark Iterations
    start_time = time.perf_counter()
    total_tokens = 0
    benchmark_schedule = torch.profiler.schedule(wait=0, warmup=2, active=1, repeat=1)
    with torch.profiler.profile(
        schedule=benchmark_schedule,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'./log_cudagraph_{use_cuda_graph}'),
        record_shapes=True,
    ) if profile else nullcontext as prof:
        for i in range(test_iterations):
            print(f'Benchmark iteration {i}')
            result = generator.text_completion([prompt], use_cuda_graph=use_cuda_graph)
            total_tokens += len(result[0]['generation'].split())
            if profile:
                prof.step()

    # Ensure GPU operations have completed
    torch.cuda.synchronize()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    seconds_per_example = elapsed_time / test_iterations
    tokens_per_second = total_tokens / elapsed_time

    print(f"Results after {test_iterations} iterations:")
    print(f"Seconds per example: {seconds_per_example:.4f} sec")
    print(f"Tokens per second: {tokens_per_second:.2f} tokens/sec")


if __name__ == "__main__":
    fire.Fire(benchmark)
