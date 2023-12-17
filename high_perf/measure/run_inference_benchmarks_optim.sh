# TODO: parameterize. this works for now.

echo "Running inference benchmarks"

if [ ! -d "benchmark_outputs_pruned" ]; then
  echo "Creating benchmark_outputs_pruned directory"
  mkdir benchmark_outputs_pruned
fi

echo "Batch size 1, num workers 0"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 1 --num_workers 0 > benchmark_outputs_pruned/batch_size_1_num_workers_0.txt
echo "Batch size 2, num workers 0"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 2 --num_workers 0 > benchmark_outputs_pruned/batch_size_2_num_workers_0.txt
echo "Batch size 4, num workers 0"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 4 --num_workers 0 > benchmark_outputs_pruned/batch_size_4_num_workers_0.txt
echo "Batch size 8, num workers 0"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 8 --num_workers 0 > benchmark_outputs_pruned/batch_size_8_num_workers_0.txt

echo "Batch size 1, num workers 1"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 1 --num_workers 1 > benchmark_outputs_pruned/batch_size_1_num_workers_1.txt
echo "Batch size 2, num workers 1"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 2 --num_workers 1 > benchmark_outputs_pruned/batch_size_2_num_workers_1.txt
echo "Batch size 4, num workers 1"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 4 --num_workers 1 > benchmark_outputs_pruned/batch_size_4_num_workers_1.txt
echo "Batch size 8, num workers 1"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 8 --num_workers 1 > benchmark_outputs_pruned/batch_size_8_num_workers_1.txt

echo "Batch size 1, num workers 2"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 1 --num_workers 2 > benchmark_outputs_pruned/batch_size_1_num_workers_2.txt
echo "Batch size 2, num workers 2"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 2 --num_workers 2 > benchmark_outputs_pruned/batch_size_2_num_workers_2.txt
echo "Batch size 4, num workers 2"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 4 --num_workers 2 > benchmark_outputs_pruned/batch_size_4_num_workers_2.txt
echo "Batch size 8, num workers 2"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 8 --num_workers 2 > benchmark_outputs_pruned/batch_size_8_num_workers_2.txt

echo "Batch size 1, num workers 4"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 1 --num_workers 4 > benchmark_outputs_pruned/batch_size_1_num_workers_4.txt
echo "Batch size 2, num workers 4"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 2 --num_workers 4 > benchmark_outputs_pruned/batch_size_2_num_workers_4.txt
echo "Batch size 4, num workers 4"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 4 --num_workers 4 > benchmark_outputs_pruned/batch_size_4_num_workers_4.txt
echo "Batch size 8, num workers 4"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 8 --num_workers 4 > benchmark_outputs_pruned/batch_size_8_num_workers_4.txt

echo "Batch size 1, num workers 8"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 1 --num_workers 8 > benchmark_outputs_pruned/batch_size_1_num_workers_8.txt
echo "Batch size 2, num workers 8"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 2 --num_workers 8 > benchmark_outputs_pruned/batch_size_2_num_workers_8.txt
echo "Batch size 4, num workers 8"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 4 --num_workers 8 > benchmark_outputs_pruned/batch_size_4_num_workers_8.txt
echo "Batch size 8, num workers 8"
torchrun inference_benchmark.py --ckpt_dir /home/gyt2107/hpml_llama/llama-2-7b/ --tokenizer_path /home/gyt2107/hpml_llama/tokenizer.model --max_seq_len 512 --max_batch_size 8 --batch_size 8 --num_workers 8 > benchmark_outputs_pruned/batch_size_8_num_workers_8.txt

echo "DONE. Exiting."