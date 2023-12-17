# HPML FINAL PROJECT

## Overview

**Project Members:** Sanjay Sharma (sgs2185) and George Tamer (gyt2107)

- original llama code: `llama/`
- our code: `high_perf`

## Download

The Llama model can be downloaded from Facebook's official repo: https://github.com/facebookresearch/llama. We downloaded the model from here by filling out Facebook's request form which gave us access to download. We have then created our own repo (which includes the Llama model) at this link: https://github.com/gtamer2/hpml_llama

In order to download the model weights and tokenizer, please visit the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept the Meta License.

Once your request is approved, you will receive a signed URL over email. Then run the _download.sh_ script, passing the URL provided when prompted to start the download.

Pre-requisites: Make sure you have `wget` and `md5sum` installed. Then to run the script: `./download.sh`.

Keep in mind that the links expire after 24 hours and a certain amount of downloads. If you start seeing errors such as `403: Forbidden`, you can always re-request a link.

## Quick Start

Please follow the steps below to get up and running with our project.

1. If needed, provision a Virtual Machine with an Nvidia GPU which has at least 16GB of memory and a host machine with at least 30GB of memory. We provisioned a VM from GCP compute engine with 8 Intel CPUs, 30GB of RAM, and an Nvidia P100 GPU with 16GB of memory.

2. SSH into your machine

3. Activate a cconda environment or python virtual environment with PyTorch and Cuda available

4. Clone this repository, or a fork of it.

5. Before your first time running any scrpits, install dependencies by runnnig:

```
pip install -e .
pip install -r requirements.txt
```

6. Follow the detailsi before to run benchmarks and optiimizations.

## Reproducinig Results

### Overview

1. Run the inference bencharks for latency and truthfulness
2. Run the model pruning script
3. Run the inference bencharks for latency and truthfulness

### Inference Latency Benchmark

The file _inference_benchmark.py_ which benchmarks the code and profiles per-epoch and average performance for data-loading time, inference time, and total epoch time. We also profile CUDA memory usage.

You can run this from the root of the repository with the following:

```
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 6
```

^^In this command, you run the inference_benchmark.py file with many command line arguments specifying the checkpoint directory of the llama-2-7b file, the path to the tokenizer file, and the max sequence length and the max batch size.

### Inference Truthfulness Benchmark

We used the [TruthfulQA](https://github.com/sylinrl/TruthfulQA/tree/main) language model benchmark to compare the mdoel's output quality before and after applying latency optimimizations.

The script **_TODO_** runs the benchmark and outputs a score.

### Pruning the Model

The file _prune_model.py_ prunes one transformer block of a llama transformer at a time. You can run this with: `torchrun prune_model.py <layer index to prune>`.

Alternatively, you can prune the entire model by running the script

### Quantizing the Model

If you would like to quantize the model

### Scripting the Model
