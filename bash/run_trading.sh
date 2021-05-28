#!/bin/bash

for seed in {0..9}; do
    for crypto in BTC ETH LTC; do
        python classifier.py $crypto $classifier 3 2020-01-01 2020-12-31 --seed $seed --gpus 1
    done
done