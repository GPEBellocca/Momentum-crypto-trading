"""
    Configuration file for every simulation.
    It contains main variables intended to be used as global references in many places.
    """

from enum import Enum


class BaseEnum(Enum):
    def __str__(self):
        return self.value


class Classifier(BaseEnum):
    L3 = "L3"
    MLP = "MLP"
    RFC = "RFC"
    SVC = "SVC"
    KNN = "KNN"
    MNB = "MNB"
    GNB = "GNB"
    LG = "LG"
    LSTM = "LSTM"


class Cryptocurrency(BaseEnum):
    BTC = "BTC"
    ETH = "ETH"
    LTC = "LTC"


LABELS = {"POS": int(1), "NEG": int(-1), "NORMAL": int(0)}


PARAM_GRIDS = {
    Classifier.RFC: {
        "criterion": ["gini", "entropy"],
        "min_samples_split": [0.01, 0.05],
        "min_samples_leaf": [0.005, 0.01],
        "max_depth": [None, 5, 10, 20],
        "class_weight": ["balanced", "balanced_subsample"],
    },
    Classifier.KNN: {
        "weights": ["uniform", "distance"],
        "n_neighbors": [3, 5, 7],
        "algorithm": ["ball_tree", "kd_tree"],
    },
    Classifier.MLP: {
        "hidden_layer_sizes": [(10,), (30,), (10, 10), (512), (512, 256)],
        "activation": ["relu", "logistic", "tanh"],
        "solver": ["lbfgs", "sgd", "adam"],
        "learning_rate": ["constant", "invscaling"],
        "learning_rate_init": [2e-5, 1e-4, 1e-5, 1e-2, 1e-1],
        "tol": [1e-4, 1e-5],
    },
    Classifier.SVC: {
        "kernel": ["linear", "poly", "rbf"],
        "degree": [3, 4, 5],
        "C": [0.001, 0.01, 1, 10, 50],
    },
    Classifier.L3: {
        "min_sup": [0.005],  # [0.005, 0.01, 0.05, 0.1],
        "min_conf": [0.5, 0.25, 0.75],
        "max_matching": [1, 3],
        "max_length": [0, 5],
    },
    Classifier.MNB: {"alpha": [0.01, 0.1, 1, 10]},
    Classifier.GNB: {},
    Classifier.LG: {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["l1", "l2"],
        "C": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
    },
}

# Hyper-params chosen after manual grid search and optimization
HPARAMS = {
    Classifier.SVC: {"kernel": "poly", "degree": 4, "C": 1},
    Classifier.RFC: {
        "criterion": "entropy",
        "min_samples_split": 0.01,
        "min_samples_leaf": 0.005,
        "max_depth": 10,
        "class_weight": "balanced_subsample",
    },
    Classifier.LG: {
        "solver": "liblinear",
        "penalty": "l1",
        "class_weight": "balanced",
        "C": 1,
    },
    Classifier.KNN: {"weights": "uniform", "n_neighbors": 3, "algorithm": "ball_tree"},
    Classifier.MLP: {
        "hidden_layer_sizes": (10, 10),
        "activation": "logistic",
        "solver": "lbfgs",
        "learning_rate": "constant",
        "learning_rate_init": 0.0001,
    },
    Classifier.MNB: {"alpha": 10},
    Classifier.GNB: {},
}

DETERMINISTIC = [Classifier.KNN, Classifier.GNB, Classifier.MNB]