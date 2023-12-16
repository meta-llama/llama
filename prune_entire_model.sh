#!/bin/bash

for i in {5..31}; do
    torchrun prune_model.py $i
    echo $i
done
