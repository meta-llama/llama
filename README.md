# TODO:
- fix params
- copyright headers
- LICENSE
- script to download checkpoints / tokenizer

# Genesis language models

This repository is intended as a minimal, hackable and readable example to load [Genesis](http://link_to_the_paper) models and run inference.

### Setup
In a conda env with pytorch / cuda available, run
```
pip install -r requirements.txt
```
Then in this repository
```
pip install -e .
```

### Inference
The provided `example.py` can be run on a single or multi-gpu node with `torchrun` and will output completions for two pre-defined prompts :
```
torchrun --nproc_per_node MP example.py --ckpt_dir /path/to/checkpoint --tokenizer_path /path/to/tokenizer
```