
import subprocess
from subprocess import call
import argparse

#call(["python3", "trading_simulator.py", "LTC", "GNB", "2", "2019-07-01", "2020-06-30"])

#call(["python3", "classifier.py", "BTC", "RFC"  , "3", "2020-02-24", "2021-02-23"])



def main():

    cryptos = ["BTC", "ETH", "LTC"]
    classifiers = ["RFC", "KNN", "SVC", "LG", "GNB", "MLP"]
    labels = ["3", "2"]
    
    parser = argparse.ArgumentParser(description="AAA")
    parser.add_argument("days_window", type=int, help="Number of days used to calculate the thresholds (min = 50)")
    parser.add_argument("k", type=float, default=0.0, help="Constant for threshold calculation")
    parser.add_argument("start_date", type=str, help="Trading start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Trading end date yyyy-mm-dd")
    args = parser.parse_args()

    for crypto in cryptos:
        for classifier in classifiers:
            for label in labels:
                print("\n\n *** Classifications details: " + crypto + " " + classifier + " " + label + " ***")
                call(["python3", "classifier.py", crypto, classifier, label,str(args.days_window) ,str(args.k), args.start_date, args.end_date])


if __name__ == "__main__":
    main()


