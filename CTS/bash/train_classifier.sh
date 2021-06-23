#!/bin/bash

classifier=$1
window=$2
k=$3
start_date=2020-01-01
end_date=2020-12-31

for crypto in BTC ETH LTC; do
    for labels in 3 2; do

        echo "Running $classifier $crypto $window $k $start_date $end_date"

        python classifier.py \
            $crypto \
            $classifier \
            $labels \
            $window \
            $k \
            $start_date \
            $end_date \
            "out_${window}_${k}"

    done
done