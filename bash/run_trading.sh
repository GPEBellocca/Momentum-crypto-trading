#!/bin/bash

for classifier in LSTM; do
    for crypto in BTC ETH LTC; do
        python classifier.py $crypto $classifier 3 2020-01-01 2020-12-31
    done
done