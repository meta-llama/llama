import os
import torch
from llama import Llama
import fire
from optimization.gsmk_dataset import get_data_loader
    
backend = "qnnpack"

def print_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    print("%.2f MB" %(os.path.getsize("tmp.pt")/1e6))
    os.remove('tmp.pt')



def get_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    return generator

def quantize_model(llama):
    model = llama.model
    # setup quantization
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
    torch.backends.quantized.engine = backend
    torch.quantization.prepare(model, inplace=True)

    # calibrate model to real world data
    dataloader = get_data_loader(3, 0)
    for _ in range(10):
        batch = next(iter(dataloader))
        llama.text_completion(
            batch,
            max_gen_len=512,
            temperature=0.6,
            top_p=0.9,
        )

    # convert in place
    torch.quantization.convert(model, inplace=True)


def main():
    ROOT_DIR = "/home/gyt2107/hpml_llama/"
    llama = get_model(os.path.join(ROOT_DIR, "llama-2-7b/"), os.path.join(ROOT_DIR, "tokenizer.model"), 512, 6)
    print("model size before in-place quantization")
    print_model_size(llama.model)

    quantize_model(llama)
    print("model size after in-place quantization")
    print_model_size(llama.model)

    # save the quantized model
    torch.save(llama.model.state_dict(), "quantized_model.pt")

if __name__ == "__main__":
    fire.Fire(main)