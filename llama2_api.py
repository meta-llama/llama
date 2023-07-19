from flask import Flask, request, jsonify
import fire

from llama import Llama

app = Flask(__name__)

ckpt_dir = "./llama-2-7b/"
tokenizer_path = "./tokenizer.model"
max_seq_len = 128
max_batch_size = 4

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


@app.route("/generate", methods=["POST"])
def generate_text():
    data = request.json
    prompts = data.get("prompts", [])
    max_gen_len = data.get("max_gen_len", 64)
    temperature = data.get("temperature", 0.6)
    top_p = data.get("top_p", 0.9)

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    generated_texts = []
    for prompt, result in zip(prompts, results):
        generated_texts.append({
            "prompt": prompt,
            "generation": result["generation"]
        })

    return jsonify({"generated_texts": generated_texts})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
