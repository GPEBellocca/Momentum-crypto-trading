import pandas as pd
import datetime as dt
from pandas_datareader import data

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import portfolio_library as tl
from config import *
import argparse
import time
from os.path import join, exists
import os
import numpy as np
import utils
import json


def compute_trading_statistics(tradingReturn, typeOfPosition, crypto):

    totalTradingReturn = []
    profitOrLoss = []
    portfolio = 0
    nl = [0]
    ns = [0]
    sum = 0
    profitable = 0

    portfolio = tl.computeTotalTradingReturn2(
        totalTradingReturn, tradingReturn, typeOfPosition, profitOrLoss, nl, ns
    )
    print("\nSIMULATION ON ", crypto)
    print(
        "Total trading return by positions:",
        "\n - Long positions: ",
        totalTradingReturn[0] * 100,
        "%",
        "\n - Short positions: ",
        totalTradingReturn[1] * (-100),
        "%",
        "\n-> Total: ",
        (totalTradingReturn[0] - totalTradingReturn[1]) * 100,
        "%",
    )

    print("Number of long position opened: ", nl[0])
    print("Number of short position opened: ", ns[0])
    print("Total number of positions opened: ", len(typeOfPosition))
    success = 0
    for i in range(len(profitOrLoss)):
        success += profitOrLoss[i]
        if profitOrLoss[i] == 1:
            sum = sum + tradingReturn[i]
            profitable = profitable + 1

    if len(profitOrLoss) == 0:
        success_percentage = 0
    else:
        success_percentage = (success / len(profitOrLoss)) * 100
    print(
        "Succesfull trades: ",
        success,
        "Percentage on total: ",
        success_percentage,
        "%\n",
    )


def compute_portfolio_statistics(df, classifier, labels, seed=None):
    initial_equity = 1000
    fees = 0.005
    equity = initial_equity
    long_pos = 0
    short_pos = 0
    success_pos = 0
    long_return = 0.0
    short_return = 0.0
    day_counter = 0
    tot_allocations = 0

    days = []
    equities = []

    long_returns = list()
    short_returns = list()

    for i in range(df.shape[0]):
        days.append(df.iloc[i, 0])
        day_counter = day_counter + 1
        daily_allocations = 0
        for j in range(1, len(df.columns), 3):
            if df.iloc[i, j] != 0:
                # campute resource allocated for each position
                daily_allocations = daily_allocations + 1
            # compute tot number of position by type
            if df.iloc[i, j + 2] == 1:
                long_pos = long_pos + 1
            elif df.iloc[i, j + 2] == -1:
                short_pos = short_pos + 1
        tot_allocations = tot_allocations + daily_allocations
        if daily_allocations != 0:
            # print("Daily allocations: ",daily_allocations)
            resources_allocated = equity / daily_allocations
            # print("Reasources allocated each:", resources_allocated)
            equity = 0
            for j in range(1, len(df.columns), 3):

                if df.iloc[i, j] != 0 and df.iloc[i, j + 2] != 0:

                    if df.iloc[i, j + 1] > 0:
                        success_pos = success_pos + 1
                    ret = resources_allocated * (1 + df.iloc[i, j + 1] - fees)
                    # print("ret:",ret)
                    equity = equity + ret
                    if df.iloc[i, j + 2] == 1:
                        long_return = long_return + df.iloc[i, j + 1]
                        long_returns.append(df.iloc[i, j + 1])
                    elif df.iloc[i, j + 2] == -1:
                        short_return = short_return + df.iloc[i, j + 1]
                        short_returns.append(df.iloc[i, j + 1])

                elif df.iloc[i, j] != 0 and df.iloc[i, j + 2] == 0:
                    ret = resources_allocated
                    # print("ret:",ret)
                    equity = equity + ret
            # print("Equity:",equity)
        equities.append(equity)

    # final_return = ((equity/initial_equity)-1)*100
    if (long_pos + short_pos) == 0:
        succ_perc = 0.0
    else:
        succ_perc = (success_pos / (long_pos + short_pos)) * 100

    print("long pos:", long_pos)
    print("short pos:", short_pos)
    print("total pos:", long_pos + short_pos)
    print("success pos:", success_pos)
    print("success percentage:", succ_perc)
    print("long return:", long_return * 100)
    print("short return:", short_return * 100)
    print("total return", (long_return + short_return) * 100)
    print(
        "number of day:", day_counter, "avg allocations:", tot_allocations / day_counter
    )

    stats = {
        "long_pos": long_pos,
        "short_pos": short_pos,
        "total_pos": long_pos + short_pos,
        "success_pos": success_pos,
        "success_perc": succ_perc,
        "long_return": long_return,
        "long_return_mean": np.array(long_returns).mean(),
        "long_return_std": np.array(long_returns).std(),
        "short_return": short_return,
        "short_return_mean": np.array(short_returns).mean(),
        "short_return_std": np.array(short_returns).std(),
        "total_return": long_return + short_return,
        "days": day_counter,
        "avg allocations": tot_allocations / day_counter,
    }

    dfres = pd.DataFrame()
    dfres["Date"] = days
    dfres["Equity"] = equities

    return equity, stats, dfres


class Classifier(BaseEnum):
    MLP = "MLP"
    RFC = "RFC"
    SVC = "SVC"
    KNN = "KNN"
    MNB = "MNB"
    GNB = "GNB"
    LG = "LG"
    LSTM = "LSTM"
    HE = "HE"
    AVG = "AVG"


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


def main():
    parser = argparse.ArgumentParser(description="AAA")
    parser.add_argument("classifier", type=Classifier, choices=list(Classifier))
    parser.add_argument("labels", type=int, help="Number of labels (2 or 3)")
    parser.add_argument(
        "days_window",
        type=int,
        help="Number of days used to calculate the thresholds (min = 50)",
    )
    parser.add_argument(
        "k", type=float, default=0.0, help="Constant for threshold calculation"
    )
    parser.add_argument("start_date", type=str, help="Trading start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Trading end date yyyy-mm-dd")
    parser.add_argument(
        "in_dir", type=str, help="Path to folder with classification predictions"
    )
    parser.add_argument("--seed", default=None)
    args = parser.parse_args()

    if args.days_window < 50:
        print("Error: days window minimum is 50 days")
        return

    cryptos = ["BTC", "ETH", "LTC"]
    dfres = pd.DataFrame()

    print("\n\n *** Portfolio simulator *** \n")
    print("Trading strategy: " + str(args.classifier), str(args.labels))
    print("Trading period: " + args.start_date + " - " + args.end_date)
    print("\n\n")

    for crypto in cryptos:

        """ OPEN DATASETS """
        df = tl.readFiles("./data/daily_datasets/" + str(crypto) + "USD.csv")
        x = df.head(
            df.shape[0] - args.days_window
        )  # eliminate first 365 obs used to calculate thresholds
        last_date = x.iloc[0, 0]
        first_date = x.iloc[x.shape[0] - 1, 0]
        # check trading period
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

        dfDaily = tl.createDailyDataset(df, args.start_date, args.end_date, args.days_window)

        df = tl.readFiles("./data/minute_datasets/" + str(crypto) + "USDm.csv")
        dfMinute = tl.createMinuteDataset(df, args.start_date, args.end_date)

        """ TRADING SIMULATION """
        print("Simulation on " + str(crypto))
        tradingReturn = []
        typeOfPosition = []
        k = args.k
        minutes = 480

        returns = []
        positions = []
        allocations = []
        dates = []

        if args.classifier == Classifier.HE:
            # only HE
            allocations, positions, returns, dates = tl.he_trading_v3(
                dfMinute,
                dfDaily,
                k,
                tradingReturn,
                typeOfPosition,
                crypto,
                minutes,
                args.days_window,
            )
        else:
            filename = utils.get_filename(
                crypto, str(args.classifier), args.labels, args.seed
            )
            dfLabels = tl.readFiles(join(args.in_dir, "labels", filename))
            if args.labels == 3:
                # 3 labels
                allocations, positions, returns, dates = tl.heml3_trading_v3(
                    dfMinute,
                    dfDaily,
                    k,
                    tradingReturn,
                    typeOfPosition,
                    dfLabels,
                    crypto,
                    minutes,
                    args.days_window,
                )
                # allocations, positions, returns, dates = tl.ml3_trading_v3(dfMinute,dfDaily,k,tradingReturn,typeOfPosition,dfLabels,crypto,minutes)
            elif args.labels == 2:
                # 2 labels
                allocations, positions, returns, dates = tl.heml2_trading_v3(
                    dfMinute,
                    dfDaily,
                    k,
                    tradingReturn,
                    typeOfPosition,
                    dfLabels,
                    crypto,
                    minutes,
                    args.days_window,
                )

        # compute final results
        compute_trading_statistics(tradingReturn, typeOfPosition, crypto)

        if crypto == "BTC":
            dfres["Dates"] = dates
        dfres[str(crypto) + " allocations"] = allocations
        dfres[str(crypto) + " returns"] = returns
        dfres[str(crypto) + " positions"] = positions

    final_equity, stats, equity_df = compute_portfolio_statistics(
        dfres, args.classifier, args.labels, args.seed
    )
    print("Final equity:", final_equity)

    utils.create_dir(join(args.in_dir, "portfolio_simulations_equity_trend"))
    utils.create_dir(join(args.in_dir, "trading"))
    utils.create_dir(join(args.in_dir, "portfolio_simulations_results"))

    equity_filename = utils.get_equity_filename(
        str(args.classifier), str(args.labels), args.seed
    )
    equity_df.to_csv(
        join(args.in_dir, "portfolio_simulations_equity_trend", equity_filename)
    )

    stats_filename = utils.get_trading_stats_filename(
        args.classifier, args.labels, args.seed
    )
    with open(join(args.in_dir, "trading", stats_filename), "w") as fp:
        json.dump(stats, fp)

    results_filename = utils.get_results_filename(
        str(args.classifier), str(args.labels), args.seed
    )

    dfres.to_csv(
        join(args.in_dir, "portfolio_simulations_results", results_filename),
        index=False,
    )

    # save also the sum on trading positions
    dfres.drop("Dates", axis=1).sum().to_csv(
        join(args.in_dir, "trading", f"sum_{results_filename}"),
        index=True,
        index_label="metric",
    )

    print("END")


if __name__ == "__main__":
    main()
