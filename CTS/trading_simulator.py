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



class Classifier(BaseEnum):
    L3 = "L3"
    MLP = "MLP"
    RFC = "RFC"
    SVC = "SVC"
    KNN = "KNN"
    MNB = "MNB"
    GNB = "GNB"
    LG = "LG"
    HE = "HE"


def check_trading_period(tmp, start_date, end_date):
    dates = tmp['Date'].to_numpy()
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
    parser.add_argument("cryptocurrency", type=Cryptocurrency,choices=list(Cryptocurrency))
    parser.add_argument("classifier", type=Classifier, choices=list(Classifier))
    parser.add_argument("labels", type=int, help="Number of labels (2 or 3)")
    parser.add_argument("start_date", type=str, help="Trading start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Trading end date yyyy-mm-dd")
    args = parser.parse_args()
    
    """ OPEN DATASETS """
    df = tl.readFiles("./data/daily_datasets/"+str(args.cryptocurrency)+"USD.csv")
    x = df.head(df.shape[0]-365) # eliminate first 365 obs used to calculate thresholds
    last_date = x.iloc[0,0]
    first_date = x.iloc[x.shape[0]-1,0]
    #check trading period
    if check_trading_period(x, args.start_date, args.end_date) == -1:
        print("Error: Trading start or end not available. Data availability : " + first_date + " - " + last_date)
        return
    elif check_trading_date(args.end_date, args.start_date):
        print("Error: Trading start date < then trading end date")
        return

    dfDaily = tl.createDailyDataset(df, args.start_date, args.end_date)

    df = tl.readFiles("./data/minute_datasets/"+str(args.cryptocurrency)+"USDm.csv")
    dfMinute = tl.createMinuteDataset(df,args.start_date, args.end_date)



    """ TRADING SIMULATION """
    print("Trading period: " + args.start_date + " - " + args.end_date )
    tradingReturn = []
    typeOfPosition = []
    k = 0.5
    
    

    if args.classifier == Classifier.HE:
        # only HE
        tl.he_trading_m(dfMinute,dfDaily,k,tradingReturn,typeOfPosition)
    else:
        # HE + ML
        dfLabels = tl.readFiles("./data/labels_datasets/"+str(args.cryptocurrency)+"_labels_"+str(args.classifier)+"_"+str(args.labels)+".csv")
        if args.labels == 3:
            # 3 labels
            tl.heml3_trading_m(dfMinute,dfDaily,k,tradingReturn,typeOfPosition,dfLabels)
        else:
            # 2 labels
            tl.heml2_trading_m(dfMinute,dfDaily,k,tradingReturn,typeOfPosition,dfLabels)

    
    # compute final results
    compute_trading_statistics(tradingReturn,typeOfPosition,args.cryptocurrency)
    print("END")

if __name__ == "__main__":
    main()

