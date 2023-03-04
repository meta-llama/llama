# LLaMa CPU fork

This is a fork of https://github.com/facebookresearch/llama to run on CPU.

Usage and setup is exactly the same.

Tested on 7B model. Even with 32GiB of ram, you'll need swap space or zram enabled to load the model (maybe it's doing some conversions?), but once it actually starts doing inference it settles down at a more reasonable <20GiB of ram.

On a Ryzen 7900X, the 7B model is able to infer several words per second, quite a lot better than you'd expect!
