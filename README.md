# Llama 2 on Apple Silicon GPU

See the original README for instructions. Only this additional change is required:
- Before using any of the `torchrun` commands, use this in the terminal to bypass unsupported operators:

```
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## Current status

Llama 2 currently does not work on Apple Silicon Mac GPUs. This repository will be updated as support for it is added in PyTorch.
See this [tracking issue in PyTorch](https://github.com/pytorch/pytorch/issues/105665) for the latest information.
