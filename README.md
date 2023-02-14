TODO: 
- make sure people have cuda pytorch ?
- actual readme
- fix params
- copyright headers
- LICENSE

```
pip install -r requirements.txt
```

For thib, my env is : /private/home/tlacroix/anaconda3/envs/genesis_llm

```
torchrun --nproc_per_node 2 example.py --ckpt_dir /large_experiments/theorem/genesis/consolidated_ckpts/13B_1T_consolidated_fp16_mp2/ --tokenizer_path /large_experiments/theorem/datasets/tokenizers/tokenizer_final_32k.minus_inf_ws.model
```