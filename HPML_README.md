### HPML FINAL PROJECT

# Sanjay Sharma, sgs2185
# George Tamer, gyt2107

The Llama model can be downloaded from Facebook's official repo: https://github.com/facebookresearch/llama. We downloaded the model from here by filling out Facebook's request form which gave us access to download. We have then created our own repo (which includes the Llama model) at this link: https://github.com/gtamer2/hpml_llama

The example_chat_completion.py and example_text_completion.py files are benchmark files provided by Facebook.

Before running anything, please make sure you run
```
pip install -e .
pip install -r requirements.txt
```
and that you have a CUDA enabled system that is compatible with your PyTorch version.

We have an inference_benchmark.py file which will benchmark the code and assess performance per epoch for data-loading times, CUDA times, and total time taken, as well as CUDA memory usage.

This can be run using
```
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 6
```
^^In this command, you run the inference_benchmark.py file with many command line arguments specifying the checkpoint directory of the llama-2-7b file, the path to the tokenizer file, and the max sequence length and the max batch size.

We also have other files called prune_model.py, quantize_model.py, and script_model.py for our work we were trying to accomplish regarding pruning, quantizing, and scripting the model.






