from config import Classifier, Cryptocurrency, LABELS, PARAM_GRIDS, HPARAMS
import os
from os.path import join as j
import pandas as pd
import datetime as dt
from pandas_datareader import data
import matplotlib

# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import *
# import matplotlib.ticker as mtick
import numpy as np
import statistics as stat
from statistics import stdev
from pandas_datareader import data
import datetime as dt
import argparse
from collections import Counter

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression

import lstm
from lstm import create_examples
import utils


def create_df_parameters(df):
    dfres = pd.DataFrame()
    dailyReturns = []

    for i in range(0, df.shape[0]):
        dailyReturns.append((df.iloc[i, 2] / df.iloc[i, 1]) - 1)

    dfres["Date"] = df["Date"]
    dfres["Daily return"] = dailyReturns
    return dfres


def get_y_3(fileName, days_window, k):

    y = []
    dayCounter = 0
    counter = 0
    dayReturn = 0

    df = pd.DataFrame()
    df = pd.read_csv(fileName)
    df_parameters = create_df_parameters(df)
    df = df.head(df.shape[0] - days_window)

    for i in range(df.shape[0] - 1, -1, -1):
        # update parameters
        parameters = computeTradingParameters(df_parameters, dayCounter, days_window, k)
        dayCounter = dayCounter + 1

        # compute the label
        label = 2
        dayReturn = (df.iloc[i, 2] / df.iloc[i, 1]) - 1

        if dayReturn > parameters[4]:
            # pos label
            y.append(LABELS["POS"])
        elif dayReturn < parameters[5]:
            # neg label
            y.append(LABELS["NEG"])
        else:
            # normal label
            y.append(LABELS["NORMAL"])

    y.pop(0)
    # y.append(float("nan"))
    y.append(LABELS["NORMAL"])

    return y


def get_y_2(fileName, days_window, k):

    y = []
    dayCounter = 0
    counter = 0
    dayReturn = 0

    df = pd.DataFrame()
    df = pd.read_csv(fileName)
    df_parameters = create_df_parameters(df)
    df = df.head(df.shape[0] - days_window)

    for i in range(df.shape[0] - 1, -1, -1):
        # update parameters
        parameters = computeTradingParameters(df_parameters, dayCounter, days_window, k)
        dayCounter = dayCounter + 1

        # compute the label
        label = 2
        dayReturn = (df.iloc[i, 2] / df.iloc[i, 1]) - 1

        if dayReturn > parameters[4]:
            # pos label
            y.append(LABELS["POS"])
        elif dayReturn < parameters[5]:
            # neg label
            y.append(LABELS["POS"])
        else:
            # normal label
            y.append(LABELS["NORMAL"])

    y.pop(0)
    # y.append(float("nan"))
    y.append(LABELS["NORMAL"])

    return y


def computeTradingParameters(df_parameters, dayCounter, days_window, k):
    parameters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    pos = []
    neg = []

    for i in range(
        df_parameters.shape[0] - 1 - dayCounter,
        df_parameters.shape[0] - 1 - dayCounter - days_window,
        -1,
    ):
        x = df_parameters.iloc[i, 1]
        if x > 0:
            pos.append(x)
        else:
            neg.append(x)

    parameters[0] = stat.mean(pos)  # avgDailyReturnPos
    parameters[1] = stat.mean(neg)  # avgDailyReturnPos
    parameters[2] = stat.pstdev(pos)  # dailySDPos
    parameters[3] = stat.pstdev(neg)  # dailySDNeg
    parameters[4] = parameters[0] + k * parameters[2]  # overractionThresholdPos
    parameters[5] = parameters[1] - k * parameters[3]  # overractionThresholdNeg

    return parameters


""" ------------------------------------------------------------------------------- """


def compute_correctness(df):
    x = []
    for i in range(0, df.shape[0]):
        if df.iloc[i, 0] == df.iloc[i, 1]:
            x.append(1)
        else:
            x.append(0)

    df["Correctness"] = x

    return df


"""--------------------------------------------"""


def get_classifier(classifier, seed=42, return_grid=False):
    """Instantiate one of the classifiers tested.

    Set return_grid=True to have the grid of hparams used in the grid search.
    """

    MAX_ITER = 10000
    N_JOBS = -1

    #  set fixed hyperparameters if we don't request the whole grid
    hparams = dict() if return_grid else HPARAMS[classifier]

    if classifier == Classifier.KNN:
        clf = KNeighborsClassifier(n_jobs=N_JOBS, **hparams)
    elif classifier == Classifier.RFC:
        clf = RandomForestClassifier(
            n_jobs=N_JOBS, n_estimators=300, random_state=seed, **hparams
        )
    elif classifier == Classifier.SVC:
        clf = SVC(gamma="scale", random_state=seed, class_weight="balanced", **hparams)
    elif classifier == Classifier.MLP:
        clf = MLPClassifier(
            random_state=seed,
            max_iter=MAX_ITER,
            early_stopping=True,
            n_iter_no_change=10,
            batch_size=4096,  #  the whole dataset should fits
            **hparams,
        )
    elif classifier == Classifier.MNB:
        clf = MultinomialNB(**hparams)
    elif classifier == Classifier.GNB:
        clf = GaussianNB()
    elif classifier == Classifier.LG:
        clf = LogisticRegression(
            random_state=seed, n_jobs=N_JOBS, max_iter=MAX_ITER, **hparams
        )
    else:
        raise NotImplementedError()

    if return_grid:
        return clf, PARAM_GRIDS[classifier]
    else:
        return clf


def get_cryptocurrency(cryptocurrency):
    if cryptocurrency == Cryptocurrency.BTC:
        crypto = 0
    elif cryptocurrency == Cryptocurrency.ETH:
        crypto = 1
    elif cryptocurrency == Cryptocurrency.LTC:
        crypto = 2
    return crypto


def check_trading_period(tmp, start_date, end_date):
    dates = tmp["Date"].to_numpy()
    if start_date in dates and end_date in dates:
        return 0
    return -1


def check_trading_date(last_date, start_date):

    last_date_list = last_date.split("-")
    start_date_list = start_date.split("-")

    if int(start_date_list[0]) > int(last_date_list[0]):
        return -1
    elif int(start_date_list[0]) == int(last_date_list[0]):
        if int(start_date_list[1]) > int(last_date_list[1]):
            return -1
        elif int(start_date_list[1]) == int(last_date_list[1]):
            if int(start_date_list[2]) > int(last_date_list[2]):
                return -1
    return 0


def compute_trading_days_number(x, start_date):
    trading_days_number = 0
    for i in range(x.shape[0] - 1, -1, -1):
        trading_days_number = trading_days_number + 1
        if x.iloc[i, 0] == start_date:
            break

    return trading_days_number


def compute_train_trading_days(x, start_date, end_date):
    training_days_number = 0
    trading_days_number = 0
    flag = 0  # | 0: training period 1: trading operiod
    for i in range(x.shape[0]):

        if flag == 0 and x.iloc[i, 0] != start_date:
            training_days_number = training_days_number + 1
        elif flag == 0 and x.iloc[i, 0] == start_date:
            flag = 1

        if flag == 1 and x.iloc[i, 0] != end_date:
            trading_days_number = trading_days_number + 1
        elif flag == 1 and x.iloc[i, 0] == end_date:
            trading_days_number = trading_days_number + 1
            break

    return training_days_number, trading_days_number


def main():
    parser = argparse.ArgumentParser(description="CTS")
    parser.add_argument(
        "cryptocurrency", type=Cryptocurrency, choices=list(Cryptocurrency)
    )
    parser.add_argument("classifier", type=Classifier, choices=list(Classifier))
    parser.add_argument("labels", type=int, help="Number of labels (2 or 3)")
    parser.add_argument(
        "days_window",
        type=int,
        help="Number of days used to calculate the thresholds (min = 50)",
    )
    parser.add_argument("k", type=float, help="Constant for threshold calculation")
    parser.add_argument("start_date", type=str, help="Trading start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Trading end date yyyy-mm-dd")
    parser.add_argument("out_dir", type=str, help="Output directory")

    # parameters for preprocessing
    parser.add_argument("--oversampling", action="store_true")

    # parameters for LSTM
    parser.add_argument("--seq_length", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    parser.add_argument("--early_stop", type=int, default=0)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--stateful", action="store_true")
    parser.add_argument("--reduce_lr", type=int, default=0)
    parser.add_argument("--do_grid_search", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.days_window < 50:
        print("Error: days window minimum is 50 days")
        return

    FeaturesFileNames = [
        "./data/features_datasets/BTCUSD_features.csv",
        "./data/features_datasets/ETHUSD_features.csv",
        "./data/features_datasets/LTCUSD_features.csv",
    ]
    FileNames = [
        "./data/daily_datasets/BTCUSD.csv",
        "./data/daily_datasets/ETHUSD.csv",
        "./data/daily_datasets/LTCUSD.csv",
    ]

    # dataset preparation
    crypto = get_cryptocurrency(args.cryptocurrency)

    if args.labels == 2:
        labels = get_y_2(FileNames[crypto], args.days_window, args.k)
    elif args.labels == 3:
        labels = get_y_3(FileNames[crypto], args.days_window, args.k)
    else:
        print("Error: wrong number of labels (2 or 3 admitted)")
        return
    y = pd.DataFrame()
    y["Labels"] = labels

    tmp = pd.DataFrame()
    tmp = pd.read_csv(FeaturesFileNames[crypto])

    x = tmp.tail(
        tmp.shape[0] - args.days_window
    )  # eliminate first days_window obs used to calculate thresholds

    first_date = x.iloc[0, 0]
    last_date = x.iloc[x.shape[0] - 1, 0]

    if check_trading_period(x, args.start_date, args.end_date) == -1:
        print(
            "Error: Trading start or end not available. Data availability : "
            + first_date
            + " - "
            + last_date
        )
        return
    elif check_trading_date(args.end_date, args.start_date):
        print("Error: Trading start date < then trading end date")
        return
    else:
        # trading_days_number = compute_trading_days_number(x,args.start_date)
        # training_days_number = x.shape[0] - trading_days_number
        training_days_number, trading_days_number = compute_train_trading_days(
            x, args.start_date, args.end_date
        )

    tmp = y.head(training_days_number + trading_days_number)
    y_train = tmp.head(training_days_number)["Labels"].to_numpy()
    y_test = tmp.tail(trading_days_number)["Labels"].to_numpy()

    print("Labels distribution in TRAIN:", Counter(y_train))
    print("Labels distribution in TEST:", Counter(y_test))

    tmp = x.head(training_days_number + trading_days_number)
    X_train = tmp.head(training_days_number)
    X_test = tmp.tail(trading_days_number)

    print(
        "Train period: "
        + str(X_train.iloc[0, 0])
        + "  -  "
        + str(X_train.iloc[X_train.shape[0] - 1, 0])
        + "  Number of observations: "
        + str(X_train.shape[0])
    )
    print(
        "Test period:  "
        + str(X_test.iloc[0, 0])
        + "  -  "
        + str(X_test.iloc[X_test.shape[0] - 1, 0])
        + "  Number of observations: "
        + str(X_test.shape[0])
    )

    dates = pd.DataFrame()
    dates["Date"] = X_test["Date"]
    reversed_dates = dates
    del X_train["Date"]
    del X_test["Date"]

    # return to numpy arrays
    X_train = X_train.values
    X_test = X_test.values

    # Oversampling with SMOTE/ADASYN
    if args.oversampling:
        X_train, y_train = oversample(X_train, y_train)

    if args.classifier == Classifier.LSTM:

        y_pred = lstm.train_and_test_lstm(
            X_train,
            y_train,
            X_test,
            y_test,
            args.labels,
            args.seq_length,
            args.batch_size,
            args.max_epochs,
            args.lr,
            args.reduce_lr,
            args.gpus,
            args.seed,
            args.early_stop,
            args.stateful,
        )

    else:
        if args.do_grid_search:
            print("BEGINNING GRID SEARCH")

            clf, param_grid = get_classifier(
                args.classifier, args.seed, return_grid=True
            )
            if args.classifier == Classifier.MNB:
                pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])
            else:
                pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

            param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
            gs = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                scoring="f1_macro",
                n_jobs=-1,
                cv=TimeSeriesSplit(n_splits=5),
                error_score=-1,
            )
            gs.fit(X_train, y_train)
            y_pred = gs.predict(X_test)
            estimator = gs.best_estimator_
            print("Best setup from validation:", estimator.named_steps["clf"])
            with open(j(args.out_dir, f"best_config_{args.classifier}.txt"), "a") as fp:
                fp.write(str(estimator.named_steps["clf"]) + "\n")

            if args.classifier == Classifier.KNN:
                print("KNN K:", estimator.named_steps["clf"].n_neighbors)

            print("ENDING GRID SEARCH")

        else:
            clf = get_classifier(args.classifier, args.seed)
            if args.classifier == Classifier.MNB:
                pipeline = Pipeline([("scaler", MinMaxScaler()), ("clf", clf)])
            else:
                pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

    print("SIMULATION ON:", args.cryptocurrency, args.classifier, args.labels)
    print("Classification report:")
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    if args.labels == 3:
        print(
            "Average F1 score (-1, 1):",
            (class_report["-1"]["f1-score"] + class_report["1"]["f1-score"]) / 2,
        )

    result = pd.DataFrame()
    result["Date"] = reversed_dates["Date"]
    result["Real"] = y_test
    result["Forecast"] = y_pred

    result = compute_correctness(result)

    filename = utils.get_filename(
        str(args.cryptocurrency), str(args.classifier), str(args.labels), args.seed
    )
    print("Filename", filename)

    if not os.path.exists(j(args.out_dir, "labels")):
        os.mkdir(j(args.out_dir, "labels"))

    if not os.path.exists(j(args.out_dir, "metrics")):
        os.mkdir(j(args.out_dir, "metrics"))

    result.to_csv(j(args.out_dir, "labels", filename), index=False)

    pd.DataFrame(class_report).to_csv(
        j(args.out_dir, "metrics", filename), index_label="metric"
    )


def oversample(X_train, y_train):
    from imblearn.over_sampling import ADASYN, SMOTE

    print("Running oversampling with SMOTE")
    return SMOTE().fit_resample(X_train, y_train)


if __name__ == "__main__":
    main()
