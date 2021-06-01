#!/bin/bash

classifier=LSTM
start_date=2020-01-01
end_date=2020-12-31

for labels in 3 2; do
    for seed in {0..9}; do

        echo "Trading with ${classifier} ${labels} on interval ${start_date} ${end_date}"

        python portfolio_simulator.py \
            $classifier \
            $labels \
            $start_date \
            $end_date \
            --seed $seed \

    done
done