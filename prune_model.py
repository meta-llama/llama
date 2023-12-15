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

def prune_model(llama):
    transformer = llama.model
    
    print(f'model type = {type(transformer)}')
    
    # set up pruning:
    total_params = 0
    before_num_zeros = 0
    after_num_zeros = 0
    for idx, transformer_block in enumerate(transformer.layers):
        print("shape of weights before pruning: ", transformer_block.attention.wq.weight.shape)
        print("shape of weights before pruning: ", transformer_block.attention.wk.weight.shape)
        print("shape of weights before pruning: ", transformer_block.attention.wv.weight.shape)
        print("shape of weights before pruning: ", transformer_block.attention_norm.weight.shape)

        # flatten all weights and append into one tensor
        before_total_weights = torch.cat(
            (transformer_block.attention.wq.weight.flatten(), 
            transformer_block.attention.wk.weight.flatten(), 
            transformer_block.attention.wv.weight.flatten(), 
            transformer_block.attention_norm.weight.flatten(), 
            transformer_block.ffn_norm.weight.flatten()), 0)
        print("shape of weights before pruning: ", before_total_weights.shape)

        before_num_zeros += torch.sum(before_total_weights == 0).item()
        
        total_params += before_total_weights.numel()
        
        layer=prune.random_unstructured(transformer_block, name="attn_norm_w", amount=0.3) # name has to be a torch.nn.Parameter
        layer=prune.random_unstructured(transformer_block.attention, name="wq", amount=0.3)
        layer=prune.random_unstructured(transformer_block.attention, name="wk", amount=0.3)
        layer=prune.random_unstructured(transformer_block.attention, name="wv", amount=0.3)
        
        # flatten all weights and append into one tensor
        after_total_weights = torch.cat(
            (layer.attention.wq.weight.flatten(), 
            layer.attention.wk.weight.flatten(), 
            layer.attention.wv.weight.flatten(), 
            layer.attention_norm.weight.flatten(), 
            layer.ffn_norm.weight.flatten()), 0)
        after_num_zeros += torch.sum(after_total_weights == 0).item()        

        if idx % 10**2 == 0:
            print(f'before_total_weights = {before_total_weights}')
            print(f'before_num_zeros = {before_num_zeros}')
            print(f'after_total_weights = {after_total_weights}')
            print(f'after_num_zeros = {after_num_zeros}')
            print(f'total_params = {total_params}')
            print(f'we are {idx} layers in')

    print(f'before_total_weights = {before_total_weights}')
    print(f'before_num_zeros = {before_num_zeros}')
    print(f'after_total_weights = {after_total_weights}')
    print(f'after_num_zeros = {after_num_zeros}')
    print(f'total_params = {total_params}')

    print(f'Sparsity of the TransformerBlock (before pruning): {before_num_zeros/total_params}')
    print(f'Sparsity of the TransformerBlock (after pruning): {after_num_zeros/total_params}')

def main():
    llama = get_model("/home/gyt2107/hpml_llama/llama-2-7b/", "tokenizer.model", 512, 6)
    prune_model(llama)

if __name__ == "__main__":
    fire.Fire(main)