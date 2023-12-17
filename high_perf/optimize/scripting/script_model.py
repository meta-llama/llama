import torch
import fire

from high_perf.common.model.get_model import get_model
# from llama import Llama

# def get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
#     generator = Llama.build(
#         ckpt_dir=ckpt_dir,
#         tokenizer_path=tokenizer_path,
#         max_seq_len=max_seq_len,
#         max_batch_size=max_batch_size,
#     )
#     return generator

def script_model(ckpt_dir, 
              tokenizer_path, 
              max_seq_len, 
              max_batch_size):
    print("Starting up...")

    torch.cuda.empty_cache()
    print("Initializing Model...")
    llama = get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size)

    print("Attempting to script model...")
    scripted_llama = torch.jit.script(llama)
    print("Successfully scripted model!")
    
    print("Saving scripted model...")
    scripted_llama.save("scripted_llama.pt")


if __name__ == "__main__":
    fire.Fire(script_model)
