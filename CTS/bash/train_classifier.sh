#!/bin/bash

classifier=$1
start_date=2020-01-01
end_date=2020-12-31

for crypto in BTC ETH LTC; do
    for labels in 3 2; do

        echo "Running ${classifier} ${crypto} ${labels} ${start_date} ${end_date}"

        python classifier.py \
            $crypto \
            $classifier \
            $labels \
            $start_date \
            $end_date

    done
done