#!/bin/bash

for i in {1..15}; do
    torchrun prune_model.py $i
    echo $i
done
