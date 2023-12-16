#!/bin/bash

for i in {0..15}; do
    torchrun prune_model.py $i
    echo $i
done
