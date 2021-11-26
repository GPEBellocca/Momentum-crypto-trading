import os
from os.path import join
import pandas as pd
import atsa
from atsa.preprocessing import FeaturesHandler, find_percentage_variation
from atsa.parsing import _date_parser
from atsa.assets import Stock
from atsa.enum import OperationAllowed
from joblib import Parallel, delayed, parallel_backend


in_dir = './data/daily_datasets'
files = [f for f in os.listdir(in_dir) if f != ".DS_Store"]
print(files)

stocks = list()
for file in files:
    fpath = join(in_dir, file)
    
    df = pd.read_csv(fpath, header=0, index_col=0)
    #df = pd.read_csv(fpath)
    
    
    stock_df = df.iloc[::-1]
    
    stock = Stock(file.split(".")[0],
                  stock_df.Open,
                  stock_df.Close,
                  stock_df.High,
                  stock_df.Low,
                  stock_df.Volume)
    stocks.append(stock)
print(f"Loaded {len(stocks)} stocks.")


out_dir = './data/features_datasets'

def process_stock(stock):
    fh = FeaturesHandler(OperationAllowed.LS)
    c_df, _ = fh.compute_technical_features(stock)
    c_df = pd.concat([c_df, stock.open, stock.high, stock.low, stock.close, stock.volume], axis=1)
    c_df.to_csv(join(out_dir, f"{stock.name}_features.csv"))

for stock in stocks:
    process_stock(stock)

