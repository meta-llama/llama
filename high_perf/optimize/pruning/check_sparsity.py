import torch
from llama import Llama


def get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

def calculate_model_sparsity(llama):
    num_zeros = 0
    total_params = 0
    for transformer_block in llama.model.layers:
        all_layer_weights = torch.cat(
            (transformer_block.attention.wq.weight.flatten(), 
            transformer_block.attention.wk.weight.flatten(), 
            transformer_block.attention.wv.weight.flatten(), 
            transformer_block.attention_norm.weight.flatten(), 
            transformer_block.ffn_norm.weight.flatten()), 0)

        num_zeros += torch.sum(all_layer_weights == 0).item()
        total_params += all_layer_weights.numel()
        del all_layer_weights
        
    return num_zeros/total_params

print("Starting up...")
llama = get_model("/home/gyt2107/hpml_llama/llama-2-7b/", "/home/gyt2107/hpml_llama/tokenizer.model", 512, 6)
print("Model loaded")
print("Calculating sparsity...")
sparsity = calculate_model_sparsity(llama)
print(f'sparsity = {sparsity}')