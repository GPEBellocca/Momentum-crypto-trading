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

# read a csv file given its name as parameter - return a dataframe
def readFiles(fileName):
    dfres = pd.DataFrame()
    dfres = pd.read_csv(fileName)
    return dfres

#create the hourly returns dataset - return a dataframe
def createHourlyReturnDataset(df):
    dfres = pd.DataFrame()
    computeHourlyReturns(df)
    dfres['Day'] = df['Day']
    dfres['Hour'] = df['Hour']
    dfres['Hourly Return'] = df['Return']
    return dfres

#compute hourly returns
def computeHourlyReturns(df):
    returns = []
    date = []
    hours = []
    
    for i in range(0,df.shape[0]):
        x = df.iloc[i,0].split()
        date.append(x[0])
        y = x[1].split('-')
        if(str(y[1]) == 'AM' and int(y[0]) == 12):
            hours.append(int('00'))
        elif(str(y[1]) == 'PM' and int(y[0]) != 12):
            z = int(y[0]) + 12
            hours.append(int(z))
        else:
            hours.append(int(y[0]))
        returns.append((df.iloc[i,5]/df.iloc[i,2])-1)

    df['Day'] = date
    df['Hour'] = hours
    df['Return'] = returns

#create the daily returns dataset - return a dataframe
def createDailyReturnDataset(df):
    
    dfres = pd.DataFrame()
    date = []
    dailyReturns = []
    for i in range (0,df.shape[0]):
        dailyReturns.append((df.iloc[i,5]/df.iloc[i,2])-1)
        x = df.iloc[i,0].split()
        date.append(x[0])
    dfres['Day'] = date
    dfres['Daily Return'] = dailyReturns
    return dfres

#compute initial avgDailyReturn and dailySD both for positive and negative returns and the threshold for overeaction days - return an array with all the parameters
def computeParameters(df,k):
    
    res = [0,0,0,0,0,0]
    pos = []
    neg = []
    
    for i in range (0,df.shape[0]):
        x = df.iloc[i,1]
        if(x > 0):
            pos.append(x)
        else:
            neg.append(x)

    res[0]=stat.mean(pos) #avgDailyReturnPos
    res[1]=stat.mean(neg) #avgDailyReturnPos
    res[2]=stat.pstdev(pos) #dailySDPos
    res[3]=stat.pstdev(neg) #dailySDNeg
    res[4]=res[0]+k*res[2] #overractionThresholdPos
    res[5]=res[1]-k*res[3] #overractionThresholdNeg

    return res

#return the raw number of the starting day from the daily dataset
def computeStartingDay(df):
    day = df.shape[0] - 366
    return day


#plot the avg hourly return of overreaction day vs normal day
def overReactionDayPlot(dfHourlyReturn,dfDailyReturn,parameters):
    df = pd.DataFrame()
    tmpDaily = pd.DataFrame()
    tmp2 = pd.DataFrame()
    
    title1 = 'Average hourly returns on overreaction and normal days'
    title2 = 'Average hourly returns on the day after the overreaction and normal days'
    
    dfDailyReturn = computeOverReactionFlag(dfDailyReturn,parameters)
    
    df = pd.merge(dfHourlyReturn,dfDailyReturn, on='Day', how='outer')
    df = df.drop(columns=['Day','Daily Return'])
    
    
    posOverVsNormalPlot(df,title1)
    negOverVsNormalPlot(df,title1)
    
    
    v = []
    for i in range(0,df.shape[0]-24):
        v.append(df.iloc[i+24,2])
    
    df = df.head(df.shape[0]-24)
    df['orFlag'] = v

    posOverVsNormalPlot(df,title2)
    negOverVsNormalPlot(df,title2)

    return

#compute if a day is a normal or an overreaction day
# 2: positive overreaction
# 1: positive normal day
# -1: negative normal day
# -2: negative overreaction day
def computeOverReactionFlag(df,parameters):
    orFlag = []
    df = df.copy()
    for i in range (0,df.shape[0]):
        if(df.iloc[i,1]>0):
            if(df.iloc[i,1]>parameters[4]):
                orFlag.append(2)
            else:
                orFlag.append(1)
        else:
            if(df.iloc[i,1]<parameters[5]):
                orFlag.append(-2)
            else:
                orFlag.append(-1)

    df['orFlag'] = orFlag
    return df

def posOverVsNormalPlot(input,tit):
    
    
    df = pd.DataFrame()
    df = input
    df = df.loc[(df['orFlag'].isin(['1','2']))]
    df['Hourly Return'] = df['Hourly Return']*100
    df = df.sort_values('Hour', ascending = True, na_position ='last')
    
    fig, ax = subplots()
    group = df.groupby(['Hour','orFlag']).mean().unstack().plot(kind='bar', title=tit,ax=ax, color=['green', 'orange'])
    
    ax.legend(['Normal Day', 'Overreaction Day']);
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()
    
    return

def negOverVsNormalPlot(input,tit):
    
    
    df = pd.DataFrame()
    df = input
    df = df.loc[(df['orFlag'].isin(['-1','-2']))]
    df['Hourly Return'] = df['Hourly Return']*100
    df = df.sort_values('Hour', ascending = True, na_position ='last')
    
    fig, ax = subplots()
    group = df.groupby(['Hour','orFlag']).mean().unstack().plot(kind='bar', title=tit,ax=ax, color=['orange', 'green'])
    
    ax.legend(['Overreaction Day', 'Normal Day']);
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    plt.show()
    
    return

def computeStats(dfDailyReturn,parameters):
    dfDailyReturn = computeOverReactionFlag(dfDailyReturn,parameters)
    
    posOvCounter = 0
    negOvCounter = 0
    posNormCounter = 0
    negNormCounter = 0
    posOvSum = 0
    negOvSum = 0
    posNormSum = 0
    negNormSum = 0
    
    
    for i in range(0,dfDailyReturn.shape[0]):
        x = dfDailyReturn.iloc[i,2]
        y = dfDailyReturn.iloc[i,1]
        if(x == 2):
            posOvCounter = posOvCounter + 1
            posOvSum = posOvSum + y
        elif(x == 1):
            posNormCounter = posNormCounter + 1
            posNormSum = posNormSum + y
        elif(x == -2):
            negOvCounter = negOvCounter + 1
            negOvSum = negOvSum + y
        else:
            negNormCounter = negNormCounter + 1
            negNormSum = negNormSum + y

    print("Number of positive overreaction: ",posOvCounter, "with average return of: ",posOvSum/posOvCounter*100,"%")
    print("Number of positive normal: ",posNormCounter, "with average return of: ",posNormSum/posNormCounter*100,"%")
    print("Number of negative overreaction: ",negOvCounter, "with average return of: ",negOvSum/negOvCounter*100,"%")
    print("Number of negative normal: ",negNormCounter, "with average return of: ",negNormSum/negNormCounter*100,"%")


    posOvCounter = 0
    negOvCounter = 0
    posNormCounter = 0
    negNormCounter = 0
    
    dayAfterReturn = 0

    for i in range(dfDailyReturn.shape[0]-1,1,-1):
        x = dfDailyReturn.iloc[i,2]
        if(x == 2):
            y = dfDailyReturn.iloc[i-1,2]
            z = dfDailyReturn.iloc[i-1,1]
            dayAfterReturn = dayAfterReturn + z
            if(y == 2):
                posOvCounter = posOvCounter + 1
            elif(y == 1):
                posNormCounter = posNormCounter + 1
            elif(y == -2):
                negOvCounter = negOvCounter + 1
            else:
                negNormCounter = negNormCounter + 1

    print("A positive overreaction day is followed by: ")
    print("Positive overreaction: ",posOvCounter)
    print("Positive normal: ",posNormCounter)
    print("Negative overreaction: ",negOvCounter)
    print("Negative normal: ",negNormCounter)
    print("Day after total return: ", dayAfterReturn*100,"%")

    posOvCounter = 0
    negOvCounter = 0
    posNormCounter = 0
    negNormCounter = 0
    
    dayAfterReturn = 0
    
    for i in range(dfDailyReturn.shape[0]-1,1,-1):
        x = dfDailyReturn.iloc[i,2]
        if(x == -2):
            y = dfDailyReturn.iloc[i-1,2]
            z = dfDailyReturn.iloc[i-1,1]
            dayAfterReturn = dayAfterReturn + z
            if(y == 2):
                posOvCounter = posOvCounter + 1
            elif(y == 1):
                posNormCounter = posNormCounter + 1
            elif(y == -2):
                negOvCounter = negOvCounter + 1
            else:
                negNormCounter = negNormCounter + 1

    print("A negative overreaction day is followed by: ")
    print("Positive overreaction: ",posOvCounter)
    print("Positive normal: ",posNormCounter)
    print("Negative overreaction: ",negOvCounter)
    print("Negative normal: ",negNormCounter)
    print("Day after total return: ", dayAfterReturn*100,"%")



    return


#Heuristic strategy 1 - open positions on overreaction day checking only thresholds
def trading(dfHourlyReturn,dfDailyReturn,k,tradingReturn,typeOfPosition):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedHourlyReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    
    counter = 0
    for i in range(dfHourlyReturn.shape[0]-1,-1,-1):
        
        
        if(dfHourlyReturn.iloc[i,1] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dayCounter = dayCounter + 1
            dayOpenPrice = dfHourlyReturn.iloc[i,3]
            firstReturn = dfHourlyReturn.iloc[i,4]/dayOpenPrice
        
        
        
        
        cumulatedHourlyReturn = (dfHourlyReturn.iloc[i,4]/dayOpenPrice)-1
        
        
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
            closePrice = dfHourlyReturn.iloc[i,4]
            positionReturn = (closePrice/openPrice)-1
            tradingReturn.append(positionReturn)
            typeOfPosition.append(positionFlag)
            positionFlag = 0

    return

#Heuristic strategy 2 - open a reverse position the day after an overreaction day checking only thresholds
def trading2(dfHourlyReturn,dfDailyReturn,k,tradingReturn,typeOfPosition):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedHourlyReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    
    yesterdayFlag = 0
    
    counter = 0
    for i in range(dfHourlyReturn.shape[0]-1,-1,-1):
       
        
        if(dfHourlyReturn.iloc[i,1] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dayCounter = dayCounter + 1
            dayOpenPrice = dfHourlyReturn.iloc[i,3]
            firstReturn = dfHourlyReturn.iloc[i,4]/dayOpenPrice
        
        cumulatedHourlyReturn = (dfHourlyReturn.iloc[i,4]/dayOpenPrice)-1
        
        
        
        if(cumulatedHourlyReturn >= parameters[4]  and positionFlag == 0  ):
            #open long position if there isn't another position opened
            positionFlag = 1
    
        if(yesterdayFlag !=0 and dfHourlyReturn.iloc[i,1] == 23 ):
            tradingReturn.append((dfHourlyReturn.iloc[i,4]/dayOpenPrice)-1)
            typeOfPosition.append(yesterdayFlag)
            yesterdayFlag = 0
        
        if(dfHourlyReturn.iloc[i,1] == 23 and positionFlag != 0):
            #close the overreaction day position
            yesterdayFlag = positionFlag*(-1) #reverse the position
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



#Heuristic + ML strategy 1 - open positions on overreaction day checking thresholds and labels
def trading3(dfHourlyReturn,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedHourlyReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    dailyLabel = 0
    
    counter = 0
    for i in range(dfHourlyReturn.shape[0]-1,-1,-1):
        
        
        if(dfHourlyReturn.iloc[i,1] == 0):
            # new day - update parameters and take dayOpenPrice
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dailyLabel = getDailyLabel(dfLabels,dayCounter)
            dayCounter = dayCounter + 1
            dayOpenPrice = dfHourlyReturn.iloc[i,3]
            firstReturn = dfHourlyReturn.iloc[i,4]/dayOpenPrice
        
        
        cumulatedHourlyReturn = (dfHourlyReturn.iloc[i,4]/dayOpenPrice)-1
        
        
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
            closePrice = dfHourlyReturn.iloc[i,4]
            positionReturn = (closePrice/openPrice)-1
            tradingReturn.append(positionReturn)
            typeOfPosition.append(positionFlag)
            positionFlag = 0

    return

#Heuristic + ML strategy 2 - open reverse position the day after an overreaction day checking thresholds and labels
def trading4(dfHourlyReturn,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels):
    
    parameters = []
    dayCounter = 0
    dayOpenPrice = 0
    cumulatedHourlyReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    openHour = 0
    
    yesterdayFlag = 0
    dailyLabel = 0
    
    counter = 0
    for i in range(dfHourlyReturn.shape[0]-1,-1,-1):
        
        if(dfHourlyReturn.iloc[i,1] == 0):
            # new day - update parameters and take dayOpenPrice
            #print("new day")
            parameters = computeTradingParameters(dfDailyReturn,k,dayCounter)
            dailyLabel = getDailyLabel(dfLabels,dayCounter)
            dayCounter = dayCounter + 1
            dayOpenPrice = dfHourlyReturn.iloc[i,3]
            firstReturn = dfHourlyReturn.iloc[i,4]/dayOpenPrice
        
        cumulatedHourlyReturn = (dfHourlyReturn.iloc[i,4]/dayOpenPrice)-1
        
        
        if(cumulatedHourlyReturn >= parameters[4]  and positionFlag == 0 and dailyLabel == 1 ):
            #open long position if there isn't another position opened
            positionFlag = 1
    
        if(yesterdayFlag !=0 and dfHourlyReturn.iloc[i,1] == 23 ):
            tradingReturn.append((dfHourlyReturn.iloc[i,4]/dayOpenPrice)-1)
            typeOfPosition.append(yesterdayFlag)
            yesterdayFlag = 0
        
        if(dfHourlyReturn.iloc[i,1] == 23 and positionFlag != 0):
            #close the overreaction day position
            yesterdayFlag = positionFlag*(-1) #reverse the position
            positionFlag = 0


    return




def getDailyLabel(dfLabels,dayCounter):
    return dfLabels.iloc[dayCounter,0]


