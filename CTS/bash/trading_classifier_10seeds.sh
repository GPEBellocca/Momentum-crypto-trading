#!/bin/bash

classifier=$1
window=$2
k=$3
in_dir=$4
start_date=2020-01-01
end_date=2020-12-31

for labels in 3 2; do
    for seed in {0..9}; do

        echo "Trading with ${classifier} ${labels} on interval ${start_date} ${end_date}"

        python portfolio_simulator.py \
            $classifier \
            $labels \
            $window \
            $k \
            $start_date \
            $end_date \
            $in_dir \
            --seed $seed

    done
done