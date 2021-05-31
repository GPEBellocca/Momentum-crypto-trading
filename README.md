# CTS

## Getting Started

### Sample run

```bash
python classifier.py BTC RFC 3 2020-01-01 2020-12-31
```

## Model configuration

### LSTM

LSTM was tested with manual tuning on the main hyper-parameters using a 10% split of
the training sequences as validation set.
We optimized the weighted f1 score (nonetheless, we should look at the trading performance,
but we suppose that better classification ~ better trading).


The final configuaration chosen is:

- sequence length 7 (days)
- max epochs 200
- early stop 10
- oversampling (balancing classes for training data with SMOTE)
- lr 2e-5 (constant, no scheduling)
- num layers 2
- hidden 512
