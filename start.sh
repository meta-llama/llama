# 原版启动
# torchrun --nproc_per_node 1 example_text_completion.py \
#     --ckpt_dir ./llama-2-7b/ \
#     --tokenizer_path tokenizer.model \
#     --max_seq_len 128 --max_batch_size 4
# 修改后http启动
torchrun --nproc_per_node 1 llama2_api.py
