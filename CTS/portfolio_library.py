import pandas as pd
import datetime as dt
from pandas_datareader import data
#import matplotlib
#matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt
#from matplotlib.pyplot import *
#import matplotlib.ticker as mtick
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
    first_day = 0

    for i in range(0,df.shape[0]):
        tmp = str(df.iloc[i,1]).split(" ")
        if tmp[0] == end_date:
            period = 1
        if tmp[0] == start_date:
            period = 2

        if period == 1 or (period == 2 and tmp[0] == start_date):
            #lavoro
            if first_day == 0:
                first_day = 1
                x = str(df.iloc[i-1,1]).split(" ")
                date.append(x[0])
                y = str(x[1]).split(':')
                hours.append(int(y[0]))
                minutes.append(int(y[1]))
                close.append(df.iloc[i-1,6])
                open.append(df.iloc[i-1,3])
            
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
def createDailyDataset(df, start_date, end_date, days_window):
    dfres = pd.DataFrame()
    date = []
    dailyReturns = []
    
    period = 0 # 1 trading period | 2 metrics calcultaion period (window days) | 0 null
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

        if counter == days_window + 1:
            break

    dfres['Day'] = date
    dfres['Daily Return'] = dailyReturns
    return dfres


""" ---------------------------------------------------------------- """

#return the raw number of the starting day from the daily dataset
def computeStartingDay(df):
    day = df.shape[0] - 366
    return day



#HE strategy minutes v3
def he_trading_v3(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,crypto,minutes,days_window):

    dates = []
    allocations = []
    positions = []
    returns = []
    
    current_day = "9999-99-99"
    day_counter = 0
    complete_day_counter = 0
    
    hour_phase = 0
    minute_phase = 0
    stop = 999
    stop_loss_flag = 0
    
    if minutes < 60:
        minute_phase = minutes
    else:
        hour_phase = minutes / 60
        minute_phase = minutes - (hour_phase*60)
    target_hour = 0
    target_minute = 0
    end_day_flag = 0



    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            current_day = dfMinute.iloc[i,0]
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter,days_window)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            positionFlag = 0
            stop_loss_flag = 0
            target_hour = 0
            target_minute = 0


        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1

        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = current_price
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = current_price


        """ stop loss """
        if positionFlag != 0:
            position_current_return = (current_price/openPrice)-1
            if positionFlag == 1 and position_current_return <= -1*stop and stop_loss_flag == 0:
                stop_loss_flag = 1
                stop_return = position_current_return
            elif positionFlag == -1 and position_current_return >= stop and stop_loss_flag == 0:
                stop_loss_flag = 1
                stop_return = position_current_return


        if dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute:
            target_hour = target_hour + hour_phase
            target_minute = target_minute + minute_phase
            if target_minute >= 60:
                target_hour = target_hour + 1
                target_minute = target_minute - 60
            if target_hour >= 24:
                target_hour = 0

        if(dfMinute.iloc[i,0] == current_day and dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59 ):
            #end of day
            end_day_flag = 1
            complete_day_counter = complete_day_counter + 1
            if crypto == "BTC":
                dates.append(current_day)
            allocations.append(1) # heuristic always allocate resources
            ret = 0.0
            
            if positionFlag != 0:
                #close current position if there is an open position and the day is a complete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                if stop_loss_flag == 1:
                    positionReturn = stop_return
                    stop_loss_flag = 0

                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
        
            positions.append(positionFlag)
            returns.append(ret)
            positionFlag = 0
                
        
                    
    print("Total number of day:",day_counter,"Complete days",complete_day_counter)
    return allocations, positions, returns, dates

#HE + ML 3 labels strategy minutes v3
def heml3_trading_v3(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto,minutes,days_window):
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    current_day = "9999-99-99"
    day_counter = 0
    complete_day_counter = 0
    
    hour_phase = 0
    minute_phase = 0
    stop = 999
    stop_loss_flag = 0
    
    if minutes < 60:
        minute_phase = minutes
    else:
        hour_phase = minutes / 60
        minute_phase = minutes - (hour_phase*60)
    target_hour = 0
    target_minute = 0
    
    
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            current_day = dfMinute.iloc[i,0]
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter,days_window)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            positionFlag = 0
            stop_loss_flag = 0
            target_hour = 0
            target_minute = 0
        
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute and dailyLabel == 1):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = current_price
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute and dailyLabel == -1):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = current_price
        
        
        """ stop loss """
        if positionFlag != 0:
            position_current_return = (current_price/openPrice)-1
            if positionFlag == 1 and position_current_return <= -1*stop and stop_loss_flag == 0:
                stop_loss_flag = 1
                stop_return = position_current_return
            elif positionFlag == -1 and position_current_return >= stop and stop_loss_flag == 0:
                stop_loss_flag = 1
                stop_return = position_current_return
    
        
        if dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute:
            target_hour = target_hour + hour_phase
            target_minute = target_minute + minute_phase
            if target_minute >= 60:
                target_hour = target_hour + 1
                target_minute = target_minute - 60
            if target_hour >= 24:
                target_hour = 0

        if(dfMinute.iloc[i,0] == current_day and dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59 ):
            #end of day
            complete_day_counter = complete_day_counter + 1
            if crypto == "BTC":
                dates.append(current_day)
            allocations.append(dailyLabel)
            ret = 0.0
            
            if positionFlag != 0:
                #close current position if there is an open position and the day is a complete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                if stop_loss_flag == 1:
                    positionReturn = stop_return
                    stop_loss_flag = 0
                
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
                
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
            
            positions.append(positionFlag)
            returns.append(ret)
            positionFlag = 0



    print("Total number of day:",day_counter,"Complete days",complete_day_counter)
    return allocations, positions, returns, dates


#HE + ML 2 labels strategy minutes v3
def heml2_trading_v3(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto,minutes,days_window):
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    current_day = "9999-99-99"
    day_counter = 0
    complete_day_counter = 0
    
    hour_phase = 0
    minute_phase = 0
    stop = 999
    stop_loss_flag = 0
    
    if minutes < 60:
        minute_phase = minutes
    else:
        hour_phase = minutes / 60
        minute_phase = minutes - (hour_phase*60)
    target_hour = 0
    target_minute = 0
    
    
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            current_day = dfMinute.iloc[i,0]
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter,days_window)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            positionFlag = 0
            stop_loss_flag = 0
            target_hour = 0
            target_minute = 0
        
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute and dailyLabel != 0):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = current_price
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute and dailyLabel != 0):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = current_price
        
        
        """ stop loss """
        if positionFlag != 0:
            position_current_return = (current_price/openPrice)-1
            if positionFlag == 1 and position_current_return <= -1*stop and stop_loss_flag == 0:
                stop_loss_flag = 1
                stop_return = position_current_return
            elif positionFlag == -1 and position_current_return >= stop and stop_loss_flag == 0:
                stop_loss_flag = 1
                stop_return = position_current_return
    
        
        if dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute:
            target_hour = target_hour + hour_phase
            target_minute = target_minute + minute_phase
            if target_minute >= 60:
                target_hour = target_hour + 1
                target_minute = target_minute - 60
            if target_hour >= 24:
                target_hour = 0
        
        if(dfMinute.iloc[i,0] == current_day and dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59 ):
            #end of day
            complete_day_counter = complete_day_counter + 1
            if crypto == "BTC":
                dates.append(current_day)
            allocations.append(dailyLabel)
            ret = 0.0
            
            if positionFlag != 0:
                #close current position if there is an open position and the day is a complete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                if stop_loss_flag == 1:
                    positionReturn = stop_return
                    stop_loss_flag = 0
            
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
                
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
            
            positions.append(positionFlag)
            returns.append(ret)
            positionFlag = 0
                
                
                
    print("Total number of day:",day_counter,"Complete days",complete_day_counter)
    return allocations, positions, returns, dates


#ML 3 labels strategy minutes v3
def ml3_trading_v3(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto,minutes, days_window):
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    current_day = "9999-99-99"
    day_counter = 0
    complete_day_counter = 0
    
    hour_phase = 0
    minute_phase = 0
    stop = 999
    stop_loss_flag = 0
    
    if minutes < 60:
        minute_phase = minutes
    else:
        hour_phase = minutes / 60
        minute_phase = minutes - (hour_phase*60)
    target_hour = 0
    target_minute = 0
    
    
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            current_day = dfMinute.iloc[i,0]
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter,days_window)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            positionFlag = 0
            stop_loss_flag = 0
            target_hour = 0
            target_minute = 0
        
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        if(positionFlag == 0 and dailyLabel == 1):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = current_price
        elif(positionFlag == 0 and dailyLabel == -1):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = current_price
        
        
        """ stop loss """
        if positionFlag != 0:
            position_current_return = (current_price/openPrice)-1
            if positionFlag == 1 and position_current_return <= -1*stop and stop_loss_flag == 0:
                stop_loss_flag = 1
                stop_return = position_current_return
            elif positionFlag == -1 and position_current_return >= stop and stop_loss_flag == 0:
                stop_loss_flag = 1
                stop_return = position_current_return
    
        
        if dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute:
            target_hour = target_hour + hour_phase
            target_minute = target_minute + minute_phase
            if target_minute >= 60:
                target_hour = target_hour + 1
                target_minute = target_minute - 60
            if target_hour >= 24:
                target_hour = 0

        if(dfMinute.iloc[i,0] == current_day and dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59 ):
            #end of day
            complete_day_counter = complete_day_counter + 1
            if crypto == "BTC":
                dates.append(current_day)
            allocations.append(dailyLabel)
            ret = 0.0
            
            if positionFlag != 0:
                #close current position if there is an open position and the day is a complete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                if stop_loss_flag == 1:
                    positionReturn = stop_return
                    stop_loss_flag = 0
                
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
                
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
            
            positions.append(positionFlag)
            returns.append(ret)
            positionFlag = 0



    print("Total number of day:",day_counter,"Complete days",complete_day_counter)
    return allocations, positions, returns, dates
""" ---------------------------------------------------------------------------------------------  """
""" ---------------------------------------------------------------------------------------------  """
""" ---------------------------------------------------------------------------------------------  """


#HE + ML strategy with 3 labels minutes
def heml3_trading_minutes(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto, minutes):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    yesterday = current_day
    yesterday_label = 0
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    c = minutes
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            allocations.append(dailyLabel)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            yesterday = current_day
            current_day = dfMinute.iloc[i,0]
            
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
            #save day return
            if day_counter > 1:
                if crypto == "BTC":
                    dates.append(yesterday)
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
                else:
                    ret = 0.0
                positions.append(positionFlag)
                returns.append(ret)
            #allocations.append(dailyLabel)
            
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel == 1 and c == minutes):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = current_price
            #openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel == -1 and c == minutes):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = current_price
            #openPrice = dayOpenPrice * (1 + parameters[5])
        
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1

        c = c + 1
        if c == (minutes + 1):
            c = 1
    
    day_counter = day_counter - 1
    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)
    allocations.pop()
    return allocations, positions, returns, dates



#HE + ML strategy with 3 labels
def heml3_trading(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    yesterday = current_day
    yesterday_label = 0
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            allocations.append(dailyLabel)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            yesterday = current_day
            current_day = dfMinute.iloc[i,0]
            
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
            #save day return
            if day_counter > 1:
                if crypto == "BTC":
                    dates.append(yesterday)
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
                else:
                    ret = 0.0
                positions.append(positionFlag)
                returns.append(ret)
            #allocations.append(dailyLabel)
            
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel == 1):
            #open long position if there isn't another position opened
            positionFlag = 1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel == -1):
            #open short position if there isn't another position opened
            positionFlag = -1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[5])
        
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1
    
    day_counter = day_counter - 1
    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)
    allocations.pop()
    return allocations, positions, returns, dates


#HE + ML strategy with 2 labels minutes v2
def heml2_trading_minutes2(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto,minutes):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    yesterday = current_day
    yesterday_label = 0
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    hour_phase = 0
    minute_phase = 0
    
    
    if minutes < 60:
        minute_phase = minutes
    else:
        hour_phase = minutes / 60
        minute_phase = minutes - (hour_phase*60)
    
    target_hour = 0
    target_minute = 0
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            allocations.append(dailyLabel)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            yesterday = current_day
            current_day = dfMinute.iloc[i,0]
            target_hour = 0
            target_minute = 0
            
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
            #save day return
            if day_counter > 1:
                if crypto == "BTC":
                    dates.append(yesterday)
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
                else:
                    ret = 0.0
                positions.append(positionFlag)
                returns.append(ret)
            #allocations.append(dailyLabel)
            
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel != 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = current_price
            #openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel != 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = current_price
            #openPrice = dayOpenPrice * (1 + parameters[5])


        if dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute:
            target_hour = target_hour + hour_phase
            target_minute = target_minute + minute_phase
            
            if target_minute >= 60:
                target_hour = target_hour + 1
                target_minute = target_minute - 60
            if target_hour >= 24:
                target_hour = 0
                    
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1


                
    day_counter = day_counter - 1
    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)
    allocations.pop()
    return allocations, positions, returns, dates



#HE + ML strategy with 2 labels minutes
def heml2_trading_minutes(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto,minutes):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    yesterday = current_day
    yesterday_label = 0
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    c = minutes
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            allocations.append(dailyLabel)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            yesterday = current_day
            current_day = dfMinute.iloc[i,0]
            
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
            #save day return
            if day_counter > 1:
                if crypto == "BTC":
                    dates.append(yesterday)
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
                else:
                    ret = 0.0
                positions.append(positionFlag)
                returns.append(ret)
            #allocations.append(dailyLabel)
            
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel != 0 and c == minutes):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = current_price
        #openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel != 0 and c == minutes):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = current_price
        #openPrice = dayOpenPrice * (1 + parameters[5])
        
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1
        
        
        c = c + 1
        if c == (minutes + 1):
            c = 1

    day_counter = day_counter - 1
    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)
    allocations.pop()
    return allocations, positions, returns, dates


#HE + ML strategy with 2 labels minutes vf
def heml2_trading_minutes_VF(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto,minutes):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    yesterday = current_day
    yesterday_label = 0
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    hour_phase = 0
    minute_phase = 0
    
    
    if minutes < 60:
        minute_phase = minutes
    else:
        hour_phase = minutes / 60
        minute_phase = minutes - (hour_phase*60)
    
    target_hour = 0
    target_minute = 0
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            allocations.append(dailyLabel)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            yesterday = current_day
            current_day = dfMinute.iloc[i,0]
            target_hour = 0
            target_minute = 0
            
            ret = 0.0
            
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
                
            
            #save day return
            if day_counter > 1:
                if crypto == "BTC":
                    dates.append(yesterday)
            
                positions.append(positionFlag)
                returns.append(ret)
            #allocations.append(dailyLabel)
            
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel != 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute):
            #open long position if there isn't another position opened
            positionFlag = 1
            openPrice = current_price
        #openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel != 0 and dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute):
            #open short position if there isn't another position opened
            positionFlag = -1
            openPrice = current_price
        #openPrice = dayOpenPrice * (1 + parameters[5])
        
        
        if dfMinute.iloc[i,1] == target_hour and dfMinute.iloc[i,2] == target_minute:
            target_hour = target_hour + hour_phase
            target_minute = target_minute + minute_phase
            
            if target_minute >= 60:
                target_hour = target_hour + 1
                target_minute = target_minute - 60
            if target_hour >= 24:
                target_hour = 0
    
        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1



    day_counter = day_counter - 1
    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)
    allocations.pop()
    return allocations, positions, returns, dates



#HE + ML strategy with 2 labels
def heml2_trading(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    yesterday = current_day
    yesterday_label = 0
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            allocations.append(dailyLabel)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            yesterday = current_day
            current_day = dfMinute.iloc[i,0]
            
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
            #save day return
            if day_counter > 1:
                if crypto == "BTC":
                    dates.append(yesterday)
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
                else:
                    ret = 0.0
                positions.append(positionFlag)
                returns.append(ret)
                #allocations.append(dailyLabel)
            
            
            positionFlag = 0
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1
        
        
        
        if(cumulatedReturn >= parameters[4]  and positionFlag == 0 and dailyLabel != 0):
            #open long position if there isn't another position opened
            positionFlag = 1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[4])
        elif(cumulatedReturn <= parameters[5] and positionFlag == 0 and dailyLabel != 0):
            #open short position if there isn't another position opened
            positionFlag = -1
            #openPrice = current_price
            openPrice = dayOpenPrice * (1 + parameters[5])

        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1
    
    day_counter = day_counter - 1
    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)
    allocations.pop()
    return allocations, positions, returns, dates

#ML strategy with 3 labels
def ml3_trading(dfMinute,dfDailyReturn,k,tradingReturn,typeOfPosition,dfLabels,crypto):
    
    parameters = []
    
    dayOpenPrice = 0
    cumulatedReturn = 0
    positionFlag = 0 # 0: no position opened | 1: long position | -1: short position
    openPrice = 0
    
    day_counter = 0
    end_day_counter = 0
    current_day = "9999-99-99"
    yesterday = current_day
    yesterday_label = 0
    
    dates = []
    allocations = []
    positions = []
    returns = []
    
    
    for i in range(dfMinute.shape[0]-1,-1,-1):
        
        if (dfMinute.iloc[i,0] != current_day):
            
            # new day
            parameters = computeTradingParameters(dfDailyReturn,k,day_counter)
            dailyLabel = getDailyLabel(dfLabels,dfMinute.iloc[i,0])
            allocations.append(dailyLabel)
            dayOpenPrice = dfMinute.iloc[i,3]
            day_counter = day_counter + 1
            yesterday = current_day
            current_day = dfMinute.iloc[i,0]
            
            ret = 0
            if(day_counter > 1 and dfMinute.iloc[i+1,1] == 23 and dfMinute.iloc[i+1,2] == 59  and positionFlag != 0):
                #close current position if there is an popen position and past day was a cpomplete day
                closePrice = dfMinute.iloc[i,3]
                positionReturn = (closePrice/openPrice)-1
                tradingReturn.append(positionReturn)
                typeOfPosition.append(positionFlag)
            
                if positionFlag == 1:
                    ret = positionReturn
                elif positionFlag == -1:
                    ret = positionReturn * -1
                else:
                    ret = 0.0
            
            #save day return
            if day_counter > 1:
                if crypto == "BTC":
                    dates.append(yesterday)
                positions.append(positionFlag)
                returns.append(ret)
                positionFlag = 0
            
            current_price = dfMinute.iloc[i,3]
            
            if(positionFlag == 0 and dailyLabel == 1 ):
                #open long position if there isn't another position opened
                positionFlag = 1
                openPrice = current_price
            #openPrice = dayOpenPrice * (1 + parameters[4])
            elif(positionFlag == 0 and dailyLabel == -1 ):
                #open short position if there isn't another position opened
                positionFlag = -1
                openPrice = current_price
            #openPrice = dayOpenPrice * (1 + parameters[5])
            
            
            
        
        
        current_price = dfMinute.iloc[i,3]
        cumulatedReturn = (current_price/dayOpenPrice)-1

        if(dfMinute.iloc[i,1] == 23 and dfMinute.iloc[i,2] == 59):
            end_day_counter = end_day_counter + 1
        
    
    day_counter = day_counter - 1
    print("Total day counter: ",day_counter,"Complete day counter: ", end_day_counter)
    allocations.pop()
    
    return allocations, positions, returns, dates


""" ---------------------------------------------------------------------------- """

def computeTradingParameters_window(dfDailyReturn,k, dayCounter):
    parameters = [0.0,0.0,0.0,0.0,0.0,0.0]
    pos = []
    neg = []
    
    days = 75
    
    pos_counter = 0
    neg_counter = 0
    for i in range(dfDailyReturn.shape[0]-1-dayCounter,dfDailyReturn.shape[0]-1-dayCounter-365,-1):
       
       
       
        x = dfDailyReturn.iloc[i,1]
        
        if pos_counter == days and neg_counter < days and x > 0:
            continue
        if pos_counter < days and neg_counter == days and x < 0:
            continue
        
        if(x > 0):
            pos.append(x)
            pos_counter = pos_counter + 1
        else:
            neg.append(x)
            neg_counter = neg_counter + 1

        if pos_counter == days and neg_counter == days:
            break


    parameters[0]=stat.mean(pos) #avgDailyReturnPos
    parameters[1]=stat.mean(neg) #avgDailyReturnPos
    parameters[2]=stat.pstdev(pos) #dailySDPos
    parameters[3]=stat.pstdev(neg) #dailySDNeg
    parameters[4]=parameters[0]+k*parameters[2] #overractionThresholdPos
    parameters[5]=parameters[1]-k*parameters[3] #overractionThresholdNeg

    return parameters

def computeTradingParameters(dfDailyReturn,k, dayCounter, days_window):
    parameters = [0.0,0.0,0.0,0.0,0.0,0.0]
    pos = []
    neg = []
    
    
    for i in range(dfDailyReturn.shape[0]-1-dayCounter,dfDailyReturn.shape[0]-1-dayCounter-days_window,-1):
        
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




    



