# FAQ
## <a name="1"></a>1. The download.sh script doesn't work on default bash in MacOS X:

Please see answers from theses issues:
 - https://github.com/facebookresearch/llama/issues/41#issuecomment-1451290160
 - https://github.com/facebookresearch/llama/issues/53#issue-1606582963


## <a name="2"></a>2. Generations are bad! 

Keep in mind these models are not finetuned for question answering. As such, they should be prompted so that the expected answer is the natural continuation of the prompt.

Here are a few examples of prompts (from [issue#69](https://github.com/facebookresearch/llama/issues/69)) geared towards finetuned models, and how to modify them to get the expected results:
 - Do not prompt with "What is the meaning of life? Be concise and do not repeat yourself." but with "I believe the meaning of life is"
 - Do not prompt with "Explain the theory of relativity." but with "Simply put, the theory of relativity states that"
 - Do not prompt with "Ten easy steps to build a website..." but with "Building a website can be done in 10 simple steps:\n"

To be able to directly prompt the models with questions / instructions, you can either:
 - Prompt it with few-shot examples so that the model understands the task you have in mind.
 - Finetune the models on datasets of instructions to make them more robust to input prompts.

We've updated `example.py` with more sample prompts. Overall, always keep in mind that models are very sensitive to prompts (particularly when they have not been finetuned).

## <a name="3"></a>3. CUDA Out of memory errors

The `example.py` file pre-allocates a cache according to these settings:
```python
model_args: ModelArgs = ModelArgs(max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params)
```

Accounting for 14GB of memory for the model weights (7B model), this leaves 16GB available for the decoding cache which stores 2 * 2 * n_layers * max_batch_size * max_seq_len * n_heads * head_dim bytes.

With default parameters, this cache was about 17GB (2 * 2 * 32 * 32 * 1024 * 32 * 128) for the 7B model.

We've added command line options to `example.py` and changed the default `max_seq_len` to 512 which should allow decoding on 30GB GPUs.

Feel free to lower these settings according to your hardware.

## <a name="4"></a>4. Other languages
The model was trained primarily on English, but also on a few other languages with Latin or Cyrillic alphabets.

For instance, LLaMA was trained on Wikipedia for the 20 following languages: bg, ca, cs, da, de, en, es, fr, hr, hu, it, nl, pl, pt, ro, ru, sl, sr, sv, uk.

LLaMA's tokenizer splits unseen characters into UTF-8 bytes, as a result, it might also be able to process other languages like Chinese or Japanese, even though they use different characters.

Although the fraction of these languages in the training was negligible, LLaMA still showcases some abilities in Chinese-English translation:

```
Prompt = "J'aime le chocolat = I like chocolate\n祝你一天过得愉快 ="
Output = "I wish you a nice day"
```