import numpy as np
from pytorch_lightning import callbacks
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torch import nn
from collections import Counter
import os
import torchmetrics as tm


class LSTMClassifier(pl.LightningModule):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        num_classes,
        batch_size,
        learning_rate,
        reduce_lr=False,
        bidirectional=False,
        stateful=False,
        class_weights=None,
    ):
        super().__init__()

        self.num_directions = (
            2 if bidirectional else 1
        )  # TODO I don't think this can be bidirectional though

        self.save_hyperparameters()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            dropout=0.1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(0.1)

        if bidirectional:
            self.linear = nn.Linear(hidden_size * 2, num_classes)
        else:
            self.linear = nn.Linear(hidden_size, num_classes)

        if stateful:
            h_n, c_n = self.init_hidden()
            self.register_buffer("hidden_h_n", h_n)
            self.register_buffer("hidden_c_n", c_n)

        self.loss_fct = nn.CrossEntropyLoss(weight=class_weights)

        self.val_F1 = tm.F1(num_classes=num_classes, average="weighted")
        self.val_acc = tm.Accuracy(num_classes=num_classes, average="weighted")

    def save_last_hidden(self):
        self.last_hidden = (self.hidden_h_n.detach(), self.hidden_c_n.detach())

    def init_hidden(self):
        #  save last hidden states right before wiping them
        return (
            torch.zeros(
                self.hparams.num_layers * self.num_directions,
                self.hparams.batch_size,
                self.hparams.hidden_size,
            ),
            torch.zeros(
                self.hparams.num_layers * self.num_directions,
                self.hparams.batch_size,
                self.hparams.hidden_size,
            ),
        )

    def forward(self, inputs):
        if self.hparams.stateful and getattr(self, "hidden", None):
            #  TODO this won't work if the batch_size is not a dividend of the dataset length
            out, (h_n, c_n) = self.lstm(inputs, (self.hidden_h_n, self.hidden_c_n))
        else:
            out, (h_n, c_n) = self.lstm(inputs)

        # keep states between batches
        if self.hparams.stateful:
            self.hidden_h_n = h_n.detach()
            self.hidden_c_n = c_n.detach()

        batch_size = inputs.shape[0]

        if self.num_directions == 1:
            h_n = h_n.view(
                self.hparams.num_layers,
                self.num_directions,
                batch_size,
                self.hparams.hidden_size,
            )

            # use only the last output
            last_hidden = h_n[-1, :, :]
            last_hidden = last_hidden.transpose(0, 1).squeeze(1)
            last_hidden = self.dropout(last_hidden)
            out = self.linear(last_hidden)
        else:
            h_n = h_n.view(
                self.hparams.num_layers,
                batch_size,
                self.hparams.hidden_size * 2,
            )
            # use only the last output
            last_hidden = h_n[-1, :, :]
            last_hidden = self.dropout(last_hidden)
            out = self.linear(last_hidden)

        return out

    def training_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.loss_fct(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.loss_fct(out, y)

        self.val_acc(out.argmax(-1), y)
        self.val_F1(out.argmax(-1), y)

        self.log("val_loss", loss)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val_F1", self.val_F1, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        X, y = batch
        out = self(X)
        loss = self.loss_fct(out, y)
        self.log("test_loss", loss)

    def training_epoch_end(self, outputs):
        if self.hparams.stateful:
            self.save_last_hidden()
            self.hidden_h_n, self.hidden_c_n = self.init_hidden()

    def validation_epoch_end(self, outputs):
        if self.hparams.stateful:
            self.save_last_hidden()
            self.hidden_h_n, self.hidden_c_n = self.init_hidden()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        if self.hparams.reduce_lr > 0:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, patience=self.hparams.reduce_lr
                    ),
                    "monitor": "val_loss",
                },
            }
        else:
            return optimizer


def labels_to_torchlabels(labels):
    d = {-1: 0, 0: 1, 1: 2}
    return np.vectorize(d.get)(labels)


def torchlabels_to_labels(labels):
    d = {0: -1, 1: 0, 2: 1}
    return np.vectorize(d.get)(labels)


def create_examples(data: np.array, targets: np.array, seq_length):
    data_len = data.shape[0]

    seqs = list()
    for i in range(data_len - seq_length):
        X = torch.from_numpy(data[i : i + seq_length, :]).float()
        y = torch.tensor(targets[i + seq_length], dtype=torch.long)
        seqs.append((X, y))

    print(f"Created {len(seqs)} sequences")
    return seqs


def get_class_weights(labels):
    counter = Counter(labels)
    class_weights = list()
    for c in sorted(list(counter.keys())):
        class_weights.append(1 - counter[c] / len(labels))

    weights = torch.tensor(class_weights, dtype=torch.float)
    return weights


class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]


def train_and_test_lstm(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    num_classes,
    seq_length,
    batch_size,
    max_epochs,
    lr,
    reduce_lr,
    gpus,
    seed,
    early_stop,
    stateful,
):
    if not os.path.exists("dumps"):
        os.mkdir("dumps")

    pl.seed_everything(seed)

    if num_classes == 3:
        # use only labels >= 0
        y_train = labels_to_torchlabels(y_train)
        y_test = labels_to_torchlabels(y_test)

    class_weights = get_class_weights(y_train)

    # TODO can we modify class weights in a better way?
    # class_weights[0] *= 2
    # class_weights[2] *= 2

    print("Class weights computed:", class_weights)

    # create the sequences for LSTM
    sequences = create_examples(X_train, y_train, seq_length)

    # use 20% of the training set for validation
    train, val = train_test_split(
        sequences, train_size=0.9, shuffle=False, random_state=42
    )

    train_dataloader = DataLoader(
        SequenceDataset(train), batch_size=batch_size, shuffle=False
    )
    val_dataloader = DataLoader(
        SequenceDataset(val), batch_size=batch_size, shuffle=False
    )

    model = LSTMClassifier(
        input_size=X_train.shape[1],
        hidden_size=512,
        num_layers=2,
        num_classes=num_classes,
        batch_size=batch_size,
        stateful=stateful,
        class_weights=class_weights,
        learning_rate=lr,
        reduce_lr=reduce_lr,
        bidirectional=False,
    )

    model_checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_loss",
        dirpath="dumps",
        filename="PL-{epoch}-{val_loss:.3f}-{train_loss:.3f}",
    )
    callbacks = [model_checkpoint]

    if early_stop > 0:
        callbacks.append(pl.callbacks.EarlyStopping("val_loss", patience=early_stop))

    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=gpus, max_epochs=max_epochs, callbacks=callbacks, auto_lr_find=True
    )

    # Train the model ⚡
    trainer.fit(model, train_dataloader, val_dataloader)

    # Create the test sequences. TODO Is this correct?
    test_sequences = create_examples(
        np.concatenate((X_train[-seq_length:, :], X_test), axis=0),
        np.concatenate((y_train[-seq_length:], y_test), axis=0),
        seq_length,
    )

    test_dataloader = DataLoader(
        SequenceDataset(test_sequences), batch_size=batch_size, shuffle=False
    )

    # Load the model and predict the test set
    best_model = LSTMClassifier.load_from_checkpoint(
        model_checkpoint.best_model_path
    ).eval()

    with torch.no_grad():
        y_preds = list()

        for batch in test_dataloader:
            X, y = batch
            out = best_model(X)
            y_preds.append(out.argmax(-1))  # batch x 1

        y_preds = torch.cat(y_preds).squeeze(-1).numpy()

    if num_classes == 3:
        y_preds = torchlabels_to_labels(y_preds)

    return y_preds
