from typing import Optional

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.2,
    top_p: float = 0.95,
    max_seq_len: int = 512,
    max_batch_size: int = 8,
    max_gen_len: Optional[int] = None,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    instructions = []

    while True:
        try:
            role_input = input("Enter a role (default is 'user'): ")

            if role_input == "":
                role_input = "user"
            elif role_input != "user" and role_input != "assistant" and role_input != "system":
                print("Invalid role. Role must be 'user', 'assistant', or 'system'.")
                continue

            content_input = input("Enter a prompt: ")

            inst = {"role": role_input, "content": content_input}
            instructions.append([inst])

            results = generator.chat_completion(
                instructions,  # type: ignore
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            # get the last instruction and result
            instruction = instructions[-1]
            result = results[-1]

            for msg in instruction:
                print(f"{msg['role'].capitalize()}: {msg['content']}\n")
            print(
                f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
            )
            print("\n==================================\n")

        except EOFError:
            # Handle Ctrl+D
            print("EOF received, exiting...")
            break

if __name__ == "__main__":
    fire.Fire(main)