#!/bin/bash

window=50
k=1

echo "Backtest all the models"

source ./bash/trading_classifier.sh KNN $window $k out_${window}_${k} &
source ./bash/trading_classifier.sh GNB $window $k out_${window}_${k} &
source ./bash/trading_classifier.sh MNB $window $k out_${window}_${k} &
source ./bash/trading_classifier.sh HE $window $k out_${window}_${k} &

source ./bash/trading_classifier_10seeds.sh MLP $window $k out_${window}_${k} &
source ./bash/trading_classifier_10seeds.sh SVC $window $k out_${window}_${k} &
source ./bash/trading_classifier_10seeds.sh LG $window $k out_${window}_${k} &
source ./bash/trading_classifier_10seeds.sh RFC $window $k out_${window}_${k} &
source ./bash/trading_classifier_10seeds.sh LSTM $window $k out_${window}_${k} &
