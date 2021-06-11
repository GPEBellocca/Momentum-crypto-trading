# CTS

## Getting Started

### Sample run

```bash
python classifier.py BTC RFC 3 2020-01-01 2020-12-31
```

## Model configuration


Models with 10 seeds:
- LSTM
- MLP
- SVC
- RFC
- LG

Models with a single run:
- KNN
- GNB
- MNB

Please find the configuration used for each classifier in [the configuration file](config.py).

### LSTM

LSTM was tested with manual tuning on the main hyper-parameters using a 10% split of
the training sequences as validation set.
We optimized the weighted f1 score (nonetheless, we should look at the trading performance,
but we suppose that better classification ~ better trading).

The final configuaration chosen is:

- sequence length 7 (days)
- max epochs 100
- early stop 10
- oversampling (balancing classes for training data with SMOTE)
- lr 2e-5 (constant, no scheduling)
- num layers 2
- hidden 512
- batch size 4096 (the whole dataset fits)

### SVC

- C 1
- class_weight balanced
- kernel poly w/ degree = 4
- single training: 10 seconds

### KNN

- weights uniform
- n neighbors 3
- algorithm ball_tree

### MLP

- activation tanh
- hidden layer sizes (10, 10)
- solver lbfgs
- learning rate constant
- learning rate init 2e-5
- tol 1e-5

### RFC

- class weight balanced_subsample
- max_depth 10
- criterion entropy
- min_samples_split 0.01
- min samples leaf 0.005

### LR

- solver liblinear
- penalty 1
- C l1

### NB (include here Gaussian and Multinomial)

MNB:
- alpha 10
