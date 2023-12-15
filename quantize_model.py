import os
import torch
from llama import Llama
import fire

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')
    
backend = "qnnpack"


def get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

def quantize_model(model):
    model.qconfig = torch.ao.quantization.default_qconfig
    torch.backends.quantized.engine = backend
    torch.quantization.prepare(model, inplace=True)
    torch.quantization.convert(model, inplace=True)


def main():
    model = get_model("llama-2-7b/", "tokenizer.model", 512, 1)
    print("model size before in-place quantization")
    print_model_size(model)

    quantize_model(model)
    print("model size after in-place quantization")
    print_model_size(model)

    # save the quantized model
    torch.save(model.state_dict(), "quantized_model.pt")

if __name__ == "__main__":
    fire.Fire(main)