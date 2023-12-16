#!/bin/bash

for i in {1..2}; do
    torchrun prune_model.py tokenizer.model last_tokenizer.model
    echo $i
done
