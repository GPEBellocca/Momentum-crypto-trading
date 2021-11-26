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

#crete minute dataset
def createMinuteDataset(df, start_date, end_date):
    returns = []
    open = []
    close = []
    date = []
    hours = []
    minutes = []
    
    period = 0

    for i in range(0,df.shape[0]):
        tmp = str(df.iloc[i,1]).split(" ")
        if tmp[0] == end_date:
            period = 1
        if tmp[0] == start_date:
            period = 2

        if period == 1 or (period == 2 and tmp[0] == start_date):
            #lavoro
            x = str(df.iloc[i,1]).split(" ")
            date.append(x[0])
            y = str(x[1]).split(':')
            hours.append(int(y[0]))
            minutes.append(int(y[1]))
            close.append(df.iloc[i,6])
            open.append(df.iloc[i,3])
        if period == 2 and tmp[0] != start_date:
            break

    dfres = pd.DataFrame()
    dfres['Day'] = date
    dfres['Hour'] = hours
    dfres['Minute'] = minutes
    dfres['Open'] = open
    dfres['Close'] = close
    return dfres


#create the daily returns dataset
def createDailyDataset(df, start_date, end_date):
    dfres = pd.DataFrame()
    date = []
    dailyReturns = []
    
    period = 0 # 1 trading period | 2 metrics calcultaion period (365 days) | 0 null
    counter = 0
    for i in range (0,df.shape[0]):
        if df.iloc[i,0] == end_date:
            period = 1
        if df.iloc[i,0] == start_date:
            period = 2

        if period == 1 or period == 2:
            dailyReturns.append((df.iloc[i,2]/df.iloc[i,1])-1)
            x = df.iloc[i,0].split()
            date.append(x[0])
        if period == 2:
            counter = counter + 1

        if counter == 366:
            break

    dfres['Day'] = date
    dfres['Daily Return'] = dailyReturns
    return dfres


""" ---------------------------------------------------------------- """

#return the raw number of the starting day from the daily dataset
def computeStartingDay(df):
    day = df.shape[0] - 366
    return day


#Heuristic strategy m
def he_trading_m(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            current_day = dfMinute.iloc[i,0]
        
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 ):
            #open long position if there isn't another position opened
            positionFlag = 1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 ):
            #open short position if there isn't another position opened
            positionFlag = -1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[5])

        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1
            
        
    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)

    return



#Heuristic strategy m old
def he_trading_m_old(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            day_counter = dayCounter + 1
            current_day = dfMinute.iloc[i,0]
        
        if(dfMinute.iloc[i,1] == 0 and dfMinute.iloc[i,2] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dayCounter = dayCounter + 1
            dayOpenPrice = dfMinute.iloc[i,3]
            newDay = dfMinute.iloc[i,0]
        
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 ):
            #open long position if there isn't another position opened
            positionFlag = 1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1+ parameters[4])
            openHour = dfMinute.iloc[i,1]
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 ):
            #open short position if there isn't another position opened
            positionFlag = -1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1+ parameters[5])
            openHour = dfMinute.iloc[i,1]
        
        
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59  and positionFlag != 0):
            #close the overreaction day position
            closePrice = dfMinute.iloc[i,4]
            positionReturn = (closePrice/openPrice)-1
            tradingReturn.append(positionReturn)
            typeOfPosition.append(positionFlag)
            positionFlag = 0


        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1


    print(dayCounter, end_day_counter)

    return


#HE + ML strategy with 3 labels m
def heml3_trading_m(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            current_day = dfMinute.iloc[i,0]
            
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel == 1 ):
            #open long position if there isn't another position opened
            positionFlag = 1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel == -1 ):
            #open short position if there isn't another position opened
            positionFlag = -1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[5])
        
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1


    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)

    return




#HE + ML strategy with 3 labels m old
def heml3_trading_m_old(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition, dfLabels):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0

    miss_days = 0
    day_counter = 0
    
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        
        if(dfMinute.iloc[i,1] == 0 and dfMinute.iloc[i,2] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            dayCounter = dayCounter + 1
            dayOpenPrice = dfMinute.iloc[i,3]
            newDay = dfMinute.iloc[i,0]
            day_counter = dayCounter + 1
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel == 1):
            #open long position if there isn't another position opened
            positionFlag = 1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1+ parameters[4])
            openHour = dfMinute.iloc[i,1]
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel == -1):
            #open short position if there isn't another position opened
            positionFlag = -1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1+ parameters[5])
            openHour = dfMinute.iloc[i,1]
        
        
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59  and positionFlag != 0):
            #close the overreaction day position
            closePrice = dfMinute.iloc[i,4]
            positionReturn = (closePrice/openPrice)-1
            tradingReturn.append(positionReturn)
            typeOfPosition.append(positionFlag)
            positionFlag = 0
            
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            if newDay != dfMinute.iloc[i,0]:
                miss_days = miss_days + 1
    print(miss_days, dayCounter)
    return


#HE + ML strategy with 32 labels m
def heml2_trading_m(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            current_day = dfMinute.iloc[i,0]
            
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel != 0 ):
            #open long position if there isn't another position opened
            positionFlag = 1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel != 0 ):
            #open short position if there isn't another position opened
            positionFlag = -1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[5])
        
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1


    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)

    return

#HE + ML strategy with 2 labels m old
def heml2_trading_m_old(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition, dfLabels):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        
        if(dfMinute.iloc[i,1] == 0 and dfMinute.iloc[i,2] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            dayCounter = dayCounter + 1
            dayOpenPrice = dfMinute.iloc[i,3]
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel != 0):
            #open long position if there isn't another position opened
            positionFlag = 1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1+ parameters[4])
            openHour = dfMinute.iloc[i,1]
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel != 0):
            #open short position if there isn't another position opened
            positionFlag = -1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1+ parameters[5])
            openHour = dfMinute.iloc[i,1]
        
        
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59  and positionFlag != 0):
            #close the overreaction day position
            closePrice = dfMinute.iloc[i,4]
            positionReturn = (closePrice/openPrice)-1
            tradingReturn.append(positionReturn)
            typeOfPosition.append(positionFlag)
            positionFlag = 0

    return


""" ---------------------------------------------------------------------------- """

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


def getDailyLabel(dfLabels, date):
    found = 0
    for i in range(0,dfLabels.shape[0]):
        if dfLabels.iloc[i,0] == date:
            found = 1
            res = dfLabels.iloc[i,2]
            break
    if found == 0:
        print("Not found")
        res = 0
    return res




    



