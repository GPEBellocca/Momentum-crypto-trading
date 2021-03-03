import pandas as pd
import datetime as dt
from pandas_datareader import data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import trading_library as tl
from config import *
import argparse


def compute_trading_statistics(tradingReturn,typeOfPosition,crypto):

    totalTradingReturn = []
    profitOrLoss = []
    portfolio = 0
    nl = [0]
    ns = [0]
    sum = 0
    profitable = 0

    portfolio = tl.computeTotalTradingReturn2(totalTradingReturn, tradingReturn,typeOfPosition,profitOrLoss,nl,ns)
    print("\nSIMULATION ON ",crypto)
    print("Total trading return by positions:","\n - Long positions: ", totalTradingReturn[0]*100,"%","\n - Short positions: ",totalTradingReturn[1]*(-100),"%","\n-> Total: ",(totalTradingReturn[0]-totalTradingReturn[1])*100,"%")
    
    print("Number of long position opened: ",nl[0])
    print("Number of short position opened: ",ns[0])
    print("Total number of positions opened: ",len(typeOfPosition))
    success=0
    for i in range(len(profitOrLoss)):
        success += profitOrLoss[i]
        if(profitOrLoss[i] == 1):
            sum = sum + tradingReturn[i]
            profitable = profitable + 1

    print("Succesfull trades: ",success,"Percentage on total: ",(success/len(profitOrLoss))*100,"%\n")

class HEStrategy(BaseEnum):
    HE = "HE"
    NA = "NA"


class MLStrategy(BaseEnum):
    L3 = "L3"
    MLP = "MLP"
    RFC = "RFC"
    SVC = "SVC"
    KNN = "KNN"
    MNB = "MNB"
    GNB = "GNB"
    LG = "LG"
    NA = "NA"

def main():
    parser = argparse.ArgumentParser(description="AAA")
    parser.add_argument("cryptocurrency", type=Cryptocurrency,choices=list(Cryptocurrency))
    parser.add_argument("hestrategy", type=HEStrategy, choices=list(HEStrategy))
    parser.add_argument("mlstrategy", type=MLStrategy, choices=list(MLStrategy))
    parser.add_argument("labels", type=int, help="Number of labels (2 or 3)")
    parser.add_argument("granularity", type=str, help="Granularity level (m or h)")
    args = parser.parse_args()
    
    #open datasets
    df = tl.readFiles("./data/daily_datasets/"+str(args.cryptocurrency)+"USD.csv")
    dfDaily = tl.createDailyDataset(df)
    start_date = dfDaily.iloc[0,0]
    end_date = dfDaily.iloc[365,0]

    if args.granularity == "h":
        df = tl.readFiles("./data/hourly_datasets/"+str(args.cryptocurrency)+"USD"+str(args.granularity)+".csv")
        dfHourly = tl.createHourlyDataset(df,start_date)
    else:
        df = tl.readFiles("./data/minute_datasets/"+str(args.cryptocurrency)+"USD"+str(args.granularity)+".csv")
        dfMinute = tl.createMinuteDataset(df,start_date, end_date)



    #trading simulation

    tradingReturn = []
    typeOfPosition = []
    k = 0.5

    if args.granularity == "h":
        print("Hourly backtest not available")
        return
    else:
        if args.hestrategy != HEStrategy.NA and args.mlstrategy == MLStrategy.NA:
            # only HE
            tl.he_trading_m(dfMinute,dfDaily,k,tradingReturn,typeOfPosition)

        elif args.hestrategy == HEStrategy.NA and args.mlstrategy != MLStrategy.NA:
            # only ML
            dfLabels = tl.readFiles("./data/labels_datasets/"+str(args.cryptocurrency)+"_labels_"+str(args.mlstrategy)+"_3.csv")
            tl.ml3_trading_m(dfMinute,dfDaily,k,tradingReturn,typeOfPosition,dfLabels)
        else:
            # HE + ML
            dfLabels = tl.readFiles("./data/labels_datasets/"+str(args.cryptocurrency)+"_labels_"+str(args.mlstrategy)+"_"+str(args.labels)+".csv")
            if args.labels == 3:
                tl.heml3_trading_m(dfMinute,dfDaily,k,tradingReturn,typeOfPosition,dfLabels)
            else:
                tl.heml2_trading_m(dfMinute,dfDaily,k,tradingReturn,typeOfPosition,dfLabels)

    compute_trading_statistics(tradingReturn,typeOfPosition,args.cryptocurrency)
    print("END")

if __name__ == "__main__":
    main()

