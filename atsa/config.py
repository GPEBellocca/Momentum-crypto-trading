"""
Configuration file for every simulation.

It contains main variables intended to be used as global references in many places.
"""
import logging
from atsa.enum import OperationAllowed, OperationLength, AssetCount, ClassificationStrategy, Classifier

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')

# ---- PARSING AND PREPROCESSING ----

UNKNOWN_SECTOR_NAME = "UNKNOWN_SECTOR"
# Time shift in the past for lagged features
# TODO REFACTOR
W = 10


# ---- CLASSIFICATION ----

OPERATION_TYPE = OperationAllowed.LS
OPERATION_LENGTH = OperationLength.MD
ASSET_COUNT = AssetCount.SINGLE
DISCRETIZATION = True
CLASSIFICATION_STRATEGY = ClassificationStrategy.HOLDOUT
MAX_PROCESSES = 12

STATS_FOLDER = "stats"
EW_INIT_TRAIN = 0.33
FW_TRAIN = 0.66

#: Percentage variation thresholds used to assign class labels
CLASS_TH = [-1, 1]

#: Seasonal periods used in Holt-Winter's model
SEASONAL_PERIODS = 5

#: Labels assigned to each in classification stage
LABELS = {
    "UP": 1,
    "HOLD": 0,
    "DOWN": -1,
    "TRAIN": "T",
    "DISCARDED": "D",
    "UNLABELED": "U",
}

# TODO DEPRECATED
CONFIGURATION = {
    "ARIMA": {
        "standard": [3, 1, 0],
        "p": [1, 2, 3, 4, 5],
        "d": [0, 1],
        "q": [0, 1]
    },
    "EXPSMOOTH": {
        "standard": (.5, .5),
        "a": [.1, .3, .5, .7, .9],
        "b": [None, .1, .3, .5, .7, .9]
    },
    "VAR": {
        "standard": 4,
        "p": [1, 2, 3, 4, 5]
    },
    "L3": {
        "standard": [1, 50],
        "minsup": [1, 0.5, 3, 10],
        "minconf": [50, 60, 70, 80]
    },
    "MLP": {
        "layers": [(100,), (500,), (1000,)],
        "solver": ["adam", "sgd"]
    },
    "RFC": {
        "n_estimators": [10, 50, 100],
        "criterion": ["gini", "entropy"]
    },
    "MNB": {
        "a": [0.2, 0.4, 0.6, 0.8, 1]
    },
    "SVC": {
        "kernel": ["rbf", "linear", "poly"]
    },
    "KNN": {
        "n_neighbors": [5, 4, 3]
    }
}

PARAM_GRIDS = {
    Classifier.RFC: {
        'criterion': ['gini', 'entropy'],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 3, 5]
    },
    Classifier.KNN: {
        'weights': ['uniform', 'distance'],
        'n_neighbors': [3, 5, 7],
        'algorithm': ['ball_tree', 'kd_tree']
    },
    Classifier.MLP: {
        'hidden_layer_sizes': [(10,), (30,), (10, 10)],
        'activation': ['relu', 'logistic', 'tanh'],
        'solver': ['lbfgs', 'sgd', "adam"],
        'learning_rate': ['constant', 'invscaling'],
        'learning_rate_init': [0.0001, 0.001, 0.01, 0.1]
    },
    Classifier.SVC: {
        'kernel': ['linear', 'poly', 'rbf'],
        'degree': [3, 4, 5],
        'C': [0.001, 0.01, 1, 10, 50]
    },
    Classifier.L3: {
        'min_sup': [0.005], # [0.005, 0.01, 0.05, 0.1],
        'min_conf': [0.5, 0.25, 0.75],
        'max_matching': [1, 3],
        'max_length': [0, 5]
    }
}


# ---- TRADING ----

MULTIPLE_DAYS = True

SESSION_ATTRIBUTES = ['Name', 'Total profit', 'Operations',
                      'Profit per operation', 'Accuracy',
                      'Accuracy (w/ unlabeled)', 'Unlabeled ratio']
COLLECTOR_ATTRIBUTES = ['Total profit',
                        'Total operations',
                        'Average profit per operation',
                        'Average profit per stock',
                        'Average operations per stock',
                        'Average accuracy per stock',
                        'Average unlabeled ratio per stock']
