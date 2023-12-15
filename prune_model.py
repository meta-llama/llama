import torch
from llama import Llama
import fire
import torch
import torch.nn.utils.prune as prune


def get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

def calculate_model_sparsity(llama):
    transformer = llama.model
    num_zeros = 0
    total_params = 0
    for transformer_block in transformer.layers:
        all_layer_weights = torch.cat(
            (transformer_block.attention.wq.weight.flatten(), 
            transformer_block.attention.wk.weight.flatten(), 
            transformer_block.attention.wv.weight.flatten(), 
            transformer_block.attention_norm.weight.flatten(), 
            transformer_block.ffn_norm.weight.flatten()), 0)

        num_zeros += torch.sum(all_layer_weights == 0).item()
        
        total_params += all_layer_weights.numel()
        
    return num_zeros/total_params


def prune_model(llama):
    transformer = llama.model
    
    print(f'model type = {type(transformer)}')
    
    for idx, transformer_block in enumerate(transformer.layers):
        print(f'pruning layer {idx}')
        if idx > 5:
            break
        # prune.random_unstructured(transformer_block, name="attn_norm_w", amount=0.3) # name has to be a torch.nn.Parameter
        # prune.random_unstructured(transformer_block.attention.wq, name="weight", amount=0.3)
        # prune.random_unstructured(transformer_block.attention.wk, name="weight", amount=0.3)
        # prune.random_unstructured(transformer_block.attention.wv, name="weight", amount=0.3)
        # prune.random_unstructured(transformer_block.attention.wo, name="weight", amount=0.3)

        # prune_model_inner(transformer_block, name="attn_norm_w", amount=0.3)
        # prune_model_inner(transformer_block.attention.wq, name="weight", amount=0.3)
        # prune_model_inner(transformer_block.attention.wk, name="weight", amount=0.3)
        # prune_model_inner(transformer_block.attention.wv, name="weight", amount=0.3)
        prune_model_inner(transformer_block.attention.wo, name="weight", amount=0.3)

def prune_model_inner(module, name, amount):
    prune.random_unstructured(module, name=name, amount=amount)

def main():
    llama = get_model("/home/gyt2107/hpml_llama/llama-2-7b/", "tokenizer.model", 512, 6)
    init_sparsity = calculate_model_sparsity(llama)
    print(f'init_sparsity = {init_sparsity}')
    prune_model(llama)
    final_sparsity = calculate_model_sparsity(llama)
    print(f'final_sparsity = {final_sparsity}')

if __name__ == "__main__":
    fire.Fire(main)