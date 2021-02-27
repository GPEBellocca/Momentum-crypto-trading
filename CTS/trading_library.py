import pandas as pd
import datetime as dt
from pandas_datareader import data
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import matplotlib.ticker as mtick
import numpy as np
import statistics as stat
from statistics import stdev
from pandas_datareader import data
import datetime as dt

# read a csv file given its name as parameter
def readFiles(fileName):
    dfres = pd.DataFrame()
    dfres = pd.read_csv(fileName)
    return dfres


#crete hourly dataset
def createHourlyDataset(df, start_date):
    returns = []
    open = []
    close = []
    date = []
    hours = []
    
    start = 0
    counter = 0
    
    for i in range(0,df.shape[0]):
        if start == 0:
            tmp = str(df.iloc[i,1]).split(" ")
            if start_date == tmp[0]:
                start = 1
        if start == 1:
            x = str(df.iloc[i,1]).split(" ")
            date.append(x[0])
            y = str(x[1]).split(':')
            hours.append(int(y[0]))
            #returns.append((df.iloc[i,6]/df.iloc[i,3])-1)
            close.append(df.iloc[i,6])
            open.append(df.iloc[i,3])
            counter = counter + 1
        
        if counter == 366 * 24:
            break
    
    dfres = pd.DataFrame()
    dfres['Day'] = date
    dfres['Hour'] = hours
    dfres['Open'] = open
    dfres['Close'] = close
    return dfres

#create the daily returns dataset
def createDailyDataset(df):
    
    dfres = pd.DataFrame()
    date = []
    dailyReturns = []
    for i in range (0,df.shape[0]):
        dailyReturns.append((df.iloc[i,2]/df.iloc[i,1])-1)
        x = df.iloc[i,0].split()
        date.append(x[0])
    dfres['Day'] = date
    dfres['Daily Return'] = dailyReturns
    return dfres

""" ---------------------------------------------------------------- """

#return the raw number of the starting day from the daily dataset
def computeStartingDay(df):
    day = df.shape[0] - 366
    return day


#Heuristic strategy - open positions on overreaction day checking only thresholds
def heuristic_trading(dfHourlyReturn,dfDailyReturn,k,tradingReturn,typeOfPosition):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedHourlyReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    
    
    for i in range(dfHourlyReturn.shape[0]-1,-1,-1):
        

        if(dfHourlyReturn.iloc[i,1] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dayCounter = dayCounter + 1
            dayOpenPrice = dfHourlyReturn.iloc[i,2]
            firstReturn = dfHourlyReturn.iloc[i,3]/dayOpenPrice
        
        
        cumulatedHourlyReturn = (dfHourlyReturn.iloc[i,3]/dayOpenPrice)-1
        
        
        if(cumulatedHourlyReturn >= parameters[4]  and positionFlag == 0 ):
        #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = dayOpenPrice * (1+ parameters[4])
            openHour = dfHourlyReturn.iloc[i,1]
        elif(cumulatedHourlyReturn <= parameters[5] and positionFlag == 0 ):
        #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = dayOpenPrice * (1 + parameters[5])
            openHour = dfHourlyReturn.iloc[i,1]
        

        if(dfHourlyReturn.iloc[i,1] == 23 and positionFlag != 0):
        #close the overreaction day position
            closePrice = dfHourlyReturn.iloc[i,3]
            positionReturn = (closePrice/openPrice)-1
            tradingReturn.append(positionReturn)
            typeOfPosition.append(positionFlag)
            positionFlag = 0

    return

#ML strategy with 3 labels - open positions on overreaction day checking thresholds and labels
def classifier3_trading(dfHourlyReturn,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedHourlyReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    dailyLabel = 0
    
    for i in range(dfHourlyReturn.shape[0]-1,-1,-1):
        
        if(dfHourlyReturn.iloc[i,1] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dailyLabel = getDailyLabel(dfLabels,dayCounter)
            dayCounter = dayCounter + 1
            dayOpenPrice = dfHourlyReturn.iloc[i,2]
            firstReturn = dfHourlyReturn.iloc[i,3]/dayOpenPrice
        
        cumulatedHourlyReturn = (dfHourlyReturn.iloc[i,3]/dayOpenPrice)-1
        
        if(cumulatedHourlyReturn >= parameters[4]  and positionFlag == 0 and dailyLabel == 1 ):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = dayOpenPrice * (1+ parameters[4])
            openHour = dfHourlyReturn.iloc[i,1]
        elif(cumulatedHourlyReturn <= parameters[5] and positionFlag == 0 and dailyLabel == -1):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = dayOpenPrice * (1 + parameters[5])
            openHour = dfHourlyReturn.iloc[i,1]
        
        
        if(dfHourlyReturn.iloc[i,1] == 23 and positionFlag != 0):
            #close the overreaction day position
            closePrice = dfHourlyReturn.iloc[i,3]
            positionReturn = (closePrice/openPrice)-1
            tradingReturn.append(positionReturn)
            typeOfPosition.append(positionFlag)
            positionFlag = 0

    return

#ML strategy with 2 labels - open positions on overreaction day checking thresholds and labels
def classifier2_trading(dfHourlyReturn,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedHourlyReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    dailyLabel = 0
    
    for i in range(dfHourlyReturn.shape[0]-1,-1,-1):
        
        if(dfHourlyReturn.iloc[i,1] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dailyLabel = getDailyLabel(dfLabels,dayCounter)
            dayCounter = dayCounter + 1
            dayOpenPrice = dfHourlyReturn.iloc[i,2]
            firstReturn = dfHourlyReturn.iloc[i,3]/dayOpenPrice
        
        cumulatedHourlyReturn = (dfHourlyReturn.iloc[i,3]/dayOpenPrice)-1
        
        if(cumulatedHourlyReturn >= parameters[4]  and positionFlag == 0 and dailyLabel != 0 ):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = dayOpenPrice * (1+ parameters[4])
            openHour = dfHourlyReturn.iloc[i,1]
        elif(cumulatedHourlyReturn <= parameters[5] and positionFlag == 0 and dailyLabel != 0):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = dayOpenPrice * (1 + parameters[5])
            openHour = dfHourlyReturn.iloc[i,1]
        
        
        if(dfHourlyReturn.iloc[i,1] == 23 and positionFlag != 0):
            #close the overreaction day position
            closePrice = dfHourlyReturn.iloc[i,3]
            positionReturn = (closePrice/openPrice)-1
            tradingReturn.append(positionReturn)
            typeOfPosition.append(positionFlag)
            positionFlag = 0

    return


def computeTradingParameters(dfDailyReturn,k, dayCounter):
    parameters = [0.0,0.0,0.0,0.0,0.0,0.0]
    pos = []
    neg = []
    
    
    
    for i in range(dfDailyReturn.shape[0]-1-dayCounter,dfDailyReturn.shape[0]-1-dayCounter-365,-1):
        x = dfDailyReturn.iloc[i,1]
        
        if(x > 0):
            pos.append(x)
        else:
            neg.append(x)


    parameters[0]=stat.mean(pos) #avgDailyReturnPos
    parameters[1]=stat.mean(neg) #avgDailyReturnPos
    parameters[2]=stat.pstdev(pos) #dailySDPos
    parameters[3]=stat.pstdev(neg) #dailySDNeg
    parameters[4]=parameters[0]+k*parameters[2] #overractionThresholdPos
    parameters[5]=parameters[1]-k*parameters[3] #overractionThresholdNeg
    
    return parameters

#perfrome a portfolio simulation trading with all equity
def computeTotalTradingReturn(totalTradingReturn,tradingReturn,typeOfPosition,profitOrLoss,nl,ns):
    equity = 1000
    totPos= 0; #total profit from long position
    totNeg = 0; #total profit from short position
    
    for i in range(len(typeOfPosition)):
        if(typeOfPosition[i]==1):
        #long positions
            nl[0] += 1
            totPos += tradingReturn[i]
            equity *= (1+tradingReturn[i])
            if(tradingReturn[i]>0):
            #profit
                profitOrLoss.append(1)
            else:
            #loss
                profitOrLoss.append(0)
        else:
        #short positions
            ns[0] += 1
            totNeg += tradingReturn[i]
            equity *= (1-tradingReturn[i])
            if(tradingReturn[i]<0):
            #profit
                profitOrLoss.append(1)
            else:
            #loss
                profitOrLoss.append(0)
    #append the total return of long and short position
    totalTradingReturn.append(totPos)
    totalTradingReturn.append(totNeg)
    
    return equity

#perfrome a portfolio simulation trading with a percentage of the equity
def computeTotalTradingReturn2(totalTradingReturn,tradingReturn,typeOfPosition,profitOrLoss,nl,ns):
    equity = 1000
    percentage = 1
    totPos= 0; #total profit from long position
    totNeg = 0; #total profit from short position
    
    for i in range(len(typeOfPosition)):
        if(typeOfPosition[i]==1):
            #long positions
            nl[0] += 1
            totPos += tradingReturn[i]
            amount = equity * percentage
            equity -= amount
            amount *= (1+tradingReturn[i])
            equity += amount
            if(tradingReturn[i]>0):
                #profit
                profitOrLoss.append(1)
            else:
                #loss
                profitOrLoss.append(0)
        else:
            #short positions
            ns[0] += 1
            totNeg += tradingReturn[i]
            amount = equity * percentage
            equity -= amount
            amount *= (1-tradingReturn[i])
            equity += amount
            if(tradingReturn[i]<0):
                #profit
                profitOrLoss.append(1)
            else:
                #loss
                profitOrLoss.append(0)
    #append the total return of long and short position
    totalTradingReturn.append(totPos)
    totalTradingReturn.append(totNeg)
    
    return equity



def getDailyLabel(dfLabels,dayCounter):
    return dfLabels.iloc[dayCounter,2]


