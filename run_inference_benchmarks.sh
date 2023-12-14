# TODO: parameterize. this works for now.

echo "Running inference benchmarks"

if [ ! -d "benchmark_outputs" ]; then
  echo "Creating benchmark_outputs directory"
  mkdir benchmark_outputs
fi

echo "Batch size 1, num workers 0"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 1 --num_workers 0 > benchmark_outputs/batch_size_1_num_workers_0.txt
echo "Batch size 16, num workers 0"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 16 --num_workers 0 > benchmark_outputs/batch_size_16_num_workers_0.txt
echo "Batch size 32, num workers 0"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 32 --num_workers 0 > benchmark_outputs/batch_size_32_num_workers_0.txt
echo "Batch size 64, num workers 0"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 64 --num_workers 0 > benchmark_outputs/batch_size_64_num_workers_0.txt

echo "Batch size 1, num workers 1"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 1 --num_workers 1 > benchmark_outputs/batch_size_1_num_workers_1.txt
echo "Batch size 16, num workers 1"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 16 --num_workers 1 > benchmark_outputs/batch_size_16_num_workers_1.txt
echo "Batch size 32, num workers 1"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 32 --num_workers 1 > benchmark_outputs/batch_size_32_num_workers_1.txt
echo "Batch size 64, num workers 1"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 64 --num_workers 1 > benchmark_outputs/batch_size_64_num_workers_1.txt

echo "Batch size 1, num workers 2"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 1 --num_workers 2 > benchmark_outputs/batch_size_1_num_workers_2.txt
echo "Batch size 16, num workers 2"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 16 --num_workers 2 > benchmark_outputs/batch_size_16_num_workers_2.txt
echo "Batch size 32, num workers 2"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 32 --num_workers 2 > benchmark_outputs/batch_size_32_num_workers_2.txt
echo "Batch size 64, num workers 2"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 64 --num_workers 2 > benchmark_outputs/batch_size_64_num_workers_2.txt

echo "Batch size 1, num workers 4"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 1 --num_workers 4 > benchmark_outputs/batch_size_1_num_workers_4.txt
echo "Batch size 16, num workers 4"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 16 --num_workers 4 > benchmark_outputs/batch_size_16_num_workers_4.txt
echo "Batch size 32, num workers 4"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 32 --num_workers 4 > benchmark_outputs/batch_size_32_num_workers_4.txt
echo "Batch size 64, num workers 4"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 64 --num_workers 4 > benchmark_outputs/batch_size_64_num_workers_4.txt

echo "Batch size 1, num workers 8"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 1 --num_workers 8 > benchmark_outputs/batch_size_1_num_workers_8.txt
echo "Batch size 16, num workers 8"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 16 --num_workers 8 > benchmark_outputs/batch_size_16_num_workers_8.txt
echo "Batch size 32, num workers 8"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 32 --num_workers 8 > benchmark_outputs/batch_size_32_num_workers_8.txt
echo "Batch size 64, num workers 8"
torchrun inference_benchmark.py --ckpt_dir llama-2-7b/ --tokenizer_path tokenizer.model --max_seq_len 512 --max_batch_size 64 --batch_size 64 --num_workers 8 > benchmark_outputs/batch_size_64_num_workers_8.txt

echo "DONE. Exiting."