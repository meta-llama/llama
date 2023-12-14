import torch
from torch.utils.data import DataLoader
import time
from datasets import load_dataset
import fire
from torch.profiler import profile, record_function, ProfilerActivity

### Setup ###
BATCH_SIZE = 1
BATCH_COUNT = 5
NUM_WORKERS = 1
PROFILE_MEMORY = True

# https://huggingface.co/datasets/gsm8k
HUGGING_FACE_GSMK_DATASET_ID = "gsm8k"

# Manual seed for reproducatibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

DEVICE_CUDA = 'cuda'
DEVICE_CPU = 'cpu'

from llama import Llama
from typing import List

def get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

def inference(
    generator: Llama,
    prompts: List[str],
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_gen_len: int = 64,
):
    with torch.no_grad():
        results = generator.text_completion(
            prompts,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        return zip(prompts, results)



def script_model(ckpt_dir, 
              tokenizer_path, 
              max_seq_len, 
              max_batch_size):
    print("Starting up...")
    print("Initializing Model...")
    llama = get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    print("Attempting to script model...")
    scripted_llama = torch.jit.script(llama)
    print("Successfully scripted model!")
    
    print("Saving scripted model...")
    scripted_llama.save("scripted_llama.pt")


if __name__ == "__main__":
    fire.Fire(script_model)
