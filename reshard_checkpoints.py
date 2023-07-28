from copy import deepcopy
import os
import shutil
import torch
import fire
import json
from pathlib import Path

from fairscale.nn.model_parallel.utils import divide_and_check_no_remainder, split_tensor_along_last_dim

from llama import ModelArgs, Transformer, Tokenizer
from llama.xla_model_parallel import (
    ParallelEmbedding,
    RowParallelLinear,
    ColumnParallelLinear,
)


@torch.no_grad()
def reshard(original_mp, target_mp, ckpt_dir, output_dir, tokenizer_path):
    assert target_mp > original_mp > 0
    factor = divide_and_check_no_remainder(target_mp, original_mp)

    os.makedirs(output_dir, exist_ok=True)

    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert original_mp == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {original_mp}"

    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    new_params = deepcopy(params)
    n_heads = params["n_heads"]
    assert n_heads % target_mp == 0, f"n_heads = {n_heads}, target_mp = {target_mp}"
    n_kv_heads = params.get("n_kv_heads", None)
    kv_head_duplicate = None
    if n_kv_heads is not None:
        if n_kv_heads >= target_mp:
            assert n_kv_heads % target_mp == 0, f"n_kv_heads = {n_kv_heads}, target_mp = {target_mp}"
        else:
            assert target_mp % n_kv_heads == 0, f"n_kv_heads = {n_kv_heads}, target_mp = {target_mp}"
            new_n_kv_heads = target_mp
            kv_head_duplicate = new_n_kv_heads // n_kv_heads
            new_params["n_kv_heads"] = new_n_kv_heads


    state_dict_key_filter = set([
        "tok_embeddings",
        "attention.wq",
        "attention.wk",
        "attention.wv",
        "attention.wo",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "output",
    ])

    original_rank = -1
    reload_model = False
    for target_rank in range(target_mp):
        if target_rank // factor != original_rank:
            original_rank = target_rank // factor
            reload_model = True

        if reload_model:
            ckpt_path = checkpoints[original_rank]
            checkpoint = torch.load(ckpt_path, map_location="cpu")

            model_args: ModelArgs = ModelArgs(**params)
            tokenizer = Tokenizer(model_path=tokenizer_path)
            model_args.vocab_size = tokenizer.n_words
            torch.set_default_tensor_type(torch.BFloat16Tensor)
            original_model = Transformer(model_args,
                                         world_size=original_mp,
                                         rank=original_rank,
                                         groups=None)
            original_model.load_state_dict(checkpoint, strict=False)
            reload_model = False

        new_model_args: ModelArgs = ModelArgs(**new_params)
        new_model_args.vocab_size = tokenizer.n_words
        shard_rank = target_rank % factor
        target_model = Transformer(new_model_args,
                                   world_size=target_mp,
                                   rank=target_rank,
                                   groups=None)
        filtered_checkpoint = {
            k: v
            for k, v in checkpoint.items()
            if not any(filter_key in k for filter_key in state_dict_key_filter)
        }
        target_model.load_state_dict(filtered_checkpoint, strict=False)
        for name, module in target_model.named_modules():
            if isinstance(module, ParallelEmbedding):
                source_module = original_model.get_submodule(name)
                weight_shard = split_tensor_along_last_dim(
                    source_module.weight.data,
                    factor)[shard_rank].contiguous()
                assert weight_shard.size() == module.weight.size()
                module.weight.copy_(weight_shard)
            elif isinstance(module, RowParallelLinear):
                source_module = original_model.get_submodule(name)
                assert module.bias is None and source_module.bias is None
                weight_shard = split_tensor_along_last_dim(
                    source_module.weight.data,
                    factor)[shard_rank].contiguous()
                assert weight_shard.size() == module.weight.size()
                module.weight.copy_(weight_shard)
            elif isinstance(module, ColumnParallelLinear):
                source_module = original_model.get_submodule(name)
                assert module.bias is None and source_module.bias is None
                if (".wk" not in name and ".wv" not in name) or (kv_head_duplicate is None):
                    weight_shard = split_tensor_along_last_dim(
                        source_module.weight.data.transpose(0, 1),
                        factor)[shard_rank].transpose(0, 1).contiguous()
                else:
                    weight_shard = split_tensor_along_last_dim(
                        source_module.weight.data.transpose(0, 1),
                        factor // kv_head_duplicate)[shard_rank // kv_head_duplicate].transpose(0, 1).contiguous()
                assert weight_shard.size() == module.weight.size()
                module.weight.copy_(weight_shard)

        state_dict = {
            k: v
            for k, v in target_model.state_dict().items()
            if k in checkpoint.keys()
        }
        torch.save(state_dict, Path(output_dir) / f"{target_rank:03}.pth")

    with open(Path(output_dir) / "params.json", "w") as f:
        json.dump(new_params, f)


if __name__ == "__main__":
    fire.Fire(reshard)
