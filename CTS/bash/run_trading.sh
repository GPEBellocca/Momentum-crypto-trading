#!/bin/bash

classifier=LSTM
start_date=2020-01-01
end_date=2020-12-31

for crypto in BTC ETH LTC; do
    for labels in 3 2; do
        for seed in {0..9}; do

            echo "Running ${classifier} ${crypto} ${labels} ${start_date} ${end_date}"

            python classifier.py \
                $crypto \
                $classifier \
                $labels \
                $start_date \
                $end_date \
                --seq_lengh 10 \
                --seed $seed \
                --max_epochs 200 \
                --early_stop 10 \
                --batch_size 4096 \
                --gpus 1 \
                --oversampling

        done
    done
done