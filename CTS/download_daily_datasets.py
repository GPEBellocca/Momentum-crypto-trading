# First import the libraries that we need to use
import pandas as pd
import requests
import json


def fetch_OHLC_data(symbol, timeframe):
    """This function will get Open/High/Low/Close, Volume and tradecount data for the pair passed and save to CSV"""
    pair_split = symbol.split('/')  # symbol must be in format XXX/XXX ie. BTC/USD
    symbol = pair_split[0] + pair_split[1]
    url = f'https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={timeframe}'
    response = requests.get(url)
    if response.status_code == 200:  # check to make sure the response from server is good
        j = json.loads(response.text)
        result = j['result']
        keys = []
        for item in result:
            keys.append(item)
        if keys[0] != 'last':
            data = pd.DataFrame(result[keys[0]],
                                columns=['unix', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'tradecount'])
        else:
            data = pd.DataFrame(result[keys[1]],
                                columns=['unix', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'tradecount'])
    
        data['date'] = pd.to_datetime(data['unix'], unit='s')
        data['volume_from'] = data['volume'].astype(float) * data['close'].astype(float)
        
        # if we failed to get any data, print an error...otherwise write the file
        if data is None:
            print("Did not return any data from Kraken for this symbol")
        else:
            if timeframe == '1':
                tf = 'minute'
            elif timeframe == '60':
                tf = 'hour'
            elif timeframe == '1440':
                tf = 'day'
            else:
                tf = ''
            #data.to_csv(f'Kraken_{symbol}_{tf}.csv', index=False)
            return data
    else:
        print("Did not receieve OK response from Kraken API")

def fetch_SPREAD_data(symbol):
    """This function will return the nearest bid/ask and calculate the spread for the symbol passed and save
        the results to a CSV file"""
    pair_split = symbol.split('/')  # symbol must be in format XXX/XXX ie. BTC/USD
    symbol = pair_split[0] + pair_split[1]
    url = f'https://api.kraken.com/0/public/Spread?pair={symbol}'
    response = requests.get(url)
    if response.status_code == 200:  # check to make sure the response from server is good
        j = json.loads(response.text)
        result = j['result']
        keys = []
        for item in result:
            keys.append(item)
        if keys[0] != 'last':
            data = pd.DataFrame(result[keys[0]], columns=['unix', 'bid', 'ask'])
        else:
            data = pd.DataFrame(result[keys[1]], columns=['unix', 'bid', 'ask'])
    
        data['date'] = pd.to_datetime(data['unix'], unit='s')
        data['spread'] = data['ask'].astype(float) - data['bid'].astype(float)
        
        # if we failed to get any data, print an error...otherwise write the file
        if data is None:
            print("Did not return any data from Kraken for this symbol")
        else:
            data.to_csv(f'Kraken_{symbol}_spreads.csv', index=False)
    else:
        print("Did not receieve OK response from Kraken API")

def fetch_PRINTS_data(symbol):
    """This function will return historical trade prints for the symbol passed and save the results to a CSV file"""
    pair_split = symbol.split('/')  # symbol must be in format XXX/XXX ie. BTC/USD
    symbol = pair_split[0] + pair_split[1]
    url = f'https://api.kraken.com/0/public/Trades?pair={symbol}'
    response = requests.get(url)
    if response.status_code == 200:  # check to make sure the response from server is good
        j = json.loads(response.text)
        
        result = j['result']
        keys = []
        for item in result:
            keys.append(item)
        if keys[0] != 'last':
            data = pd.DataFrame(result[keys[0]], columns=['price', 'volume', 'time', 'buysell', 'ordtype', 'misc'])
        else:
            data = pd.DataFrame(result[keys[1]], columns=['price', 'volume', 'time', 'buysell', 'ordtype', 'misc'])
    
        data['date'] = pd.to_datetime(data['time'], unit='s')
        data['buysell'] = data['buysell'].apply(lambda x: "buy" if x == 'b' else "sell")
        data['ordtype'] = data['ordtype'].apply(lambda x: "limit" if x == 'l' else "market")
        data['dollaramount'] = data['price'].astype(float) * data['volume'].astype(float)
        data.drop(columns=['misc'], inplace=True)  #drop misc column that is typically blank
        
        # if we failed to get any data, print an error...otherwise write the file
        if data is None:
            print("Did not return any data from Kraken for this symbol")
        else:
            data.to_csv(f'Kraken_{symbol}_tradeprints.csv', index=False)
    else:
        print("Did not receieve OK response from Kraken API")

def update_pair(pair,newdata):
    df = pd.read_csv("./data/daily_datasets/"+pair+".csv")
    last_day = df.iloc[0,0]

    index = 0
    found = 0
    for i in range(newdata.shape[0]-1,-1,-1):
        index = index + 1
        date = str(newdata.iloc[i,8]).split(" ")
        if date[0] == last_day:
            found = 1
            break

    if found == 1:
        print("Add data from: " + str(last_day))
        tmp = newdata.tail(index-1)
        data_portion = tmp.head(tmp.shape[0]-1)

        data_portion_cleaned = clean_new_data(data_portion)
        df.set_index('Date', inplace=True)
        dataset_updated = pd.concat([data_portion_cleaned,df],sort=False)
        dataset_updated.to_csv("./data/daily_datasets/"+pair+".csv")
        print("Data updated")
    else:
        print("Impossible to perform the update")


def clean_new_data(df):
    dfres = pd.DataFrame()
    
    #dfres['Date'] = df['date']
    #dfres['Date'] = pd.to_datetime(df['date'].dt.strftime('%Y-%m-%d'))
    #dfres['Date']=pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d')
    dfres['Date'] = pd.to_datetime(df['date']).dt.date
    dfres['Open'] = df['open']
    dfres['Close'] = df['close']
    dfres['High'] = df['high']
    dfres['Low'] = df['low']
    dfres['Volume'] = df['volume']
    dfres.set_index('Date', inplace=True)
    reversed_df = dfres.iloc[::-1]
                       
    return reversed_df



if __name__ == "__main__":
    # we set which pair we want to retrieve data for
    pairs = {'BTCUSD': 'BTC/USD', 'ETHUSD': 'ETH/USD', 'LTCUSD': 'LTC/USD'}
    for key in pairs.keys():
        # full timeframe intervals found here: https://www.kraken.com/en-us/features/api#get-ohlc-data
        #fetch_OHLC_data(symbol=pair, timeframe='1') # fetches minute data
        # fetch_OHLC_data(symbol=pair, timeframe='60')  # fetches hourly data
        df = fetch_OHLC_data(symbol=pairs[key], timeframe='1440')  # fetches daily data
        #fetch_SPREAD_data(symbol=pair) # gets bid/ask spread data
        #fetch_PRINTS_data(symbol=pair) # gets historical trade print data
        update_pair(key,df)






