
import fire
import os
import sys

from lm_eval import evaluator, tasks

from llama import EvalHarnessAdaptor
from llama.utils import setup_model_parallel, load


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    task_list: list = None,
    num_fewshot: int = 0,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 32,
):
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")

    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )

    # apply model to tasks from eval harness
    adaptor = EvalHarnessAdaptor(
        device='0',
        gpt2=generator,
        tokenizer=generator.tokenizer,
        batch_size=1,
        temperature=temperature,
        top_p=top_p,
    )

    if not task_list:
        task_list = tasks.ALL_TASKS
    task_dict = tasks.get_task_dict(task_list)
    results = evaluator.evaluate(adaptor, task_dict, num_fewshot=num_fewshot)
    print(results)


if __name__ == "__main__":
    fire.Fire(main)
