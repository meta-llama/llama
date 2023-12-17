# Model Quantization

Please note that quantization of Llama 2 is not possible on an Nvidia P100 GPU, because the Llama weights are distributed with half precision (16-bits), and P100 does not support 8-bit quantization.

However, this script could be used in this optizatoni pipeline for a full-precisioni model.
