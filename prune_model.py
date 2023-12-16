import torch
from llama import Llama
import fire
import torch
import torch.nn.utils.prune as prune
import sys
import nvidia_smi
import os

def check_mem():
    nvidia_smi.nvmlInit()

    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    # card id 0 hardcoded here, there is also a call to get all available card ids, so we could iterate

    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)

    print("Total memory:", info.total)
    print("Free memory:", info.free)
    print("Used memory:", info.used)

    nvidia_smi.nvmlShutdown()


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


def prune_model(llama):
    print(f'model type = {type(llama.model)}')
    print(f'model layers = {len(llama.model.layers)}')

    for idx, transformer_block in enumerate(llama.model.layers):
        check_mem()
        print(f'pruning layer {idx}')
        torch.cuda.empty_cache()
        if idx > 1:
            break
        prune.random_unstructured(transformer_block, name="attn_norm_w", amount=0.3) # name has to be a torch.nn.Parameter
        torch.cuda.empty_cache()
        prune.random_unstructured(transformer_block.attention.wq, name="weight", amount=0.3)
        torch.cuda.empty_cache()
        prune.random_unstructured(transformer_block.attention.wk, name="weight", amount=0.3)
        torch.cuda.empty_cache()
        prune.random_unstructured(transformer_block.attention.wv, name="weight", amount=0.3)
        torch.cuda.empty_cache()
        prune.random_unstructured(transformer_block.attention.wo, name="weight", amount=0.3)        
        torch.cuda.empty_cache()


def prune_model_layer(llama, layer_idx):
    print(f'model type = {type(llama.model)}')
    print(f'model layers = {len(llama.model.layers)}')
    transformer_block = llama.model.layers[layer_idx]
    check_mem()
    print(f'pruning layer {layer_idx}')
    torch.cuda.empty_cache()
    prune.random_unstructured(transformer_block, name="attn_norm_w", amount=0.3) # name has to be a torch.nn.Parameter
    torch.cuda.empty_cache()
    prune.random_unstructured(transformer_block.attention.wq, name="weight", amount=0.3)
    torch.cuda.empty_cache()
    prune.random_unstructured(transformer_block.attention.wk, name="weight", amount=0.3)
    torch.cuda.empty_cache()
    prune.random_unstructured(transformer_block.attention.wv, name="weight", amount=0.3)
    torch.cuda.empty_cache()
    prune.random_unstructured(transformer_block.attention.wo, name="weight", amount=0.3)        
    torch.cuda.empty_cache()


def main():
    layer = int(sys.argv[1])
    print("Starting up...")

    checkpoint_dir = "/home/gyt2107/hpml_llama/llama-2-7b-pruned/"
    llama = get_model(checkpoint_dir, "tokenizer.model", 512, 4)
    check_mem()
    print("Model loaded")

    print("Calculating sparsity...")
    init_sparsity = calculate_model_sparsity(llama)
    print(f'init_sparsity = {init_sparsity}')
    
    torch.cuda.empty_cache()
    check_mem()
    
    print(f"Pruning model layer {layer}...")
    prune_model_layer(llama, layer)
    print("Pruning done")

    torch.cuda.empty_cache()
    check_mem()

    print("Calculating sparsity...")
    final_sparsity = calculate_model_sparsity(llama)
    print(f'final_sparsity = {final_sparsity}')

    print("Saving model...")
    save_path = os.path.join(checkpoint_dir, f"consolidated.00.pth")
    torch.save(llama.model.state_dict(), save_path)

if __name__ == "__main__":
    fire.Fire(main)