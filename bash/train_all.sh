#!/bin/bash

window=50
k=2

echo "Training all the models"

source ./bash/train_classifier.sh KNN $window $k &
source ./bash/train_classifier.sh GNB $window $k &
source ./bash/train_classifier.sh MNB $window $k

source ./bash/train_classifier_10seeds.sh MLP $window $k
source ./bash/train_classifier_10seeds.sh SVC $window $k
source ./bash/train_classifier_10seeds.sh LG $window $k
source ./bash/train_classifier_10seeds.sh RFC $window $k

