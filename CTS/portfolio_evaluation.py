import pandas as pd
import subprocess
from subprocess import call
import argparse

def compute_portfolio_statistics(df):
    initial_equity = 1000
    equity = initial_equity

    for i in range(df.shape[0]):
        
        daily_positions = 0
        for j in range(0,len(df.columns),2):
            if df.iloc[i,j] != 0:
                daily_positions = daily_positions + 1
        if daily_positions != 0:
            resources_allocated = equity/daily_positions
            equity = 0
            for j in range(0,len(df.columns),2):
                if df.iloc[i,j] != 0:
                    ret = resources_allocated * (1 + df.iloc[i,j+1])
                    equity = equity + ret
    #final_return = ((equity/initial_equity)-1)*100
    return equity


def main():

    classifiers = ["HE", "RFC", "KNN", "SVC", "LG", "GNB"]
    labels = ["3", "2"]


    for classifier in classifiers:
        for label in labels:
            print("\n\n *** Portfolio details: " + classifier + " " + label + " ***")
            df = pd.DataFrame()
            df = pd.read_csv("./data/portfolio_simulations_results/results_"+str(classifier)+"_"+str(label)+".csv")
            print(compute_portfolio_statistics(df))
            return



if __name__ == "__main__":
    main()


