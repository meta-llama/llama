# Model Scripting

Please note that we were not able to successfully script Llama. This is because the models' TransformerBlocks and the transformer blocks' Attention and Feedforward networks all rely on [Fairscale](https://github.com/facebookresearch/fairscale) implementations of [ColumnParallelLinear](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L218-L296) and [RowParallelLinear](https://github.com/facebookresearch/fairscale/blob/164cc0f3170b4a3951dd84dda29c3e1504ac4d6e/fairscale/nn/model_parallel/layers.py#L299-L387).

These are layers optimized for parallel processing with complex control flow that is incompatible with scripting.

Time permitting, we would have explored re-implementing these layers to be jit-scriptable.
