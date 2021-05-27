from lstm import create_examples
from config import Classifier, Cryptocurrency, LABELS
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
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression

import lstm


def create_df_parameters(df):
    dfres = pd.DataFrame()
    dailyReturns = []

    for i in range(0, df.shape[0]):
        dailyReturns.append((df.iloc[i, 2] / df.iloc[i, 1]) - 1)

    dfres["Date"] = df["Date"]
    dfres["Daily return"] = dailyReturns
    return dfres


def get_y_3(fileName):

    y = []
    dayCounter = 0
    counter = 0
    dayReturn = 0

    df = pd.DataFrame()
    df = pd.read_csv(fileName)
    df_parameters = create_df_parameters(df)
    df = df.head(df.shape[0] - 365)

    for i in range(df.shape[0] - 1, -1, -1):
        # update parameters
        parameters = computeTradingParameters(df_parameters, dayCounter)
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


def get_y_2(fileName):

    y = []
    dayCounter = 0
    counter = 0
    dayReturn = 0

    df = pd.DataFrame()
    df = pd.read_csv(fileName)
    df_parameters = create_df_parameters(df)
    df = df.head(df.shape[0] - 365)

    for i in range(df.shape[0] - 1, -1, -1):
        # update parameters
        parameters = computeTradingParameters(df_parameters, dayCounter)
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


def computeTradingParameters(df_parameters, dayCounter):
    parameters = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    k = 0.5
    pos = []
    neg = []

    for i in range(
        df_parameters.shape[0] - 1 - dayCounter,
        df_parameters.shape[0] - 1 - dayCounter - 365,
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


def get_classifier_and_grid(classifier):
    if classifier == Classifier.KNN:
        clf = KNeighborsClassifier(n_jobs=-1)
    elif classifier == Classifier.RFC:
        clf = RandomForestClassifier(
            n_jobs=-1, n_estimators=200, random_state=42, class_weight="balanced"
        )
    elif classifier == Classifier.SVC:
        clf = SVC(gamma="scale", random_state=42, class_weight="balanced")
    elif classifier == Classifier.MLP:
        clf = MLPClassifier(
            random_state=42, max_iter=10000, early_stopping=True, n_iter_no_change=3
        )
    elif classifier == Classifier.MNB:
        clf = MultinomialNB()
    elif classifier == Classifier.GNB:
        clf = GaussianNB()
    elif classifier == Classifier.LG:
        clf = LogisticRegression(random_state=42, class_weight="balanced", n_jobs=-1)
    else:
        raise NotImplementedError()

    return clf, PARAM_GRIDS[classifier]


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
    parser.add_argument("start_date", type=str, help="Trading start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Trading end date yyyy-mm-dd")
    parser.add_argument("--seq_length", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stateful", action="store_true")
    args = parser.parse_args()

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
        labels = get_y_2(FileNames[crypto])
    elif args.labels == 3:
        labels = get_y_3(FileNames[crypto])
    else:
        print("Error: wrong number of labels (2 or 3 admitted)")
        return
    y = pd.DataFrame()
    y["Labels"] = labels

    tmp = pd.DataFrame()
    tmp = pd.read_csv(FeaturesFileNames[crypto])

    x = tmp.tail(
        tmp.shape[0] - 365
    )  # eliminate first 365 obs used to calculate thresholds

    first_date = x.iloc[0, 0]
    last_date = x.iloc[x.shape[0] - 1, 0]

    """
    print(x.shape[0], first_date, last_date)
    print(y, y.shape[0])
    return
    """

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

    if args.classifier == Classifier.LSTM:
        y_pred = lstm.train_and_test_lstm(
            X_train.values,
            y_train,
            X_test.values,
            y_test,
            args.seq_length,
            args.batch_size,
            args.max_epochs,
            args.gpus,
            args.seed,
            args.early_stop,
            args.stateful,
        )

    else:
        # training, validation and test
        clf, param_grid = get_classifier_and_grid(args.classifier)
        pipeline = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
        param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
        gs = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring="f1_weighted",
            n_jobs=-1,
            cv=TimeSeriesSplit(n_splits=5),
        )
        gs.fit(X_train, y_train)
        y_pred = gs.predict(X_test)
        estimator = gs.best_estimator_
        print("Best setup from validation:", estimator)

    print("SIMULATION ON:", args.cryptocurrency, args.classifier, args.labels)

    acc = accuracy_score(y_test, y_pred)
    print("Weighted accuracy:", acc)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print("f1 weighted:", f1)
    print("Classiifcation report:")
    print(classification_report(y_test, y_pred))

    result = pd.DataFrame()
    result["Date"] = reversed_dates["Date"]
    result["Real"] = y_test
    result["Forecast"] = y_pred

    result = compute_correctness(result)
    path = "./data/labels_datasets/"
    path = (
        path
        + str(args.cryptocurrency)
        + "_"
        + "labels"
        + "_"
        + str(args.classifier)
        + "_"
        + str(args.labels)
        + ".csv"
    )
    # result.to_excel(path, index = False)
    result.to_csv(path, index=False)

    return


if __name__ == "__main__":
    main()
