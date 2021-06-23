
import subprocess
from subprocess import call
import argparse





def main():

    classifiers = ["HE","RFC", "KNN", "SVC", "LG", "GNB", "MLP"]
    labels = ["3", "2"]
    
    parser = argparse.ArgumentParser(description="AAA")
    parser.add_argument("days_window", type=int, help="Number of days used to calculate the thresholds (min = 50)")
    parser.add_argument("k", type=float, default=0.0, help="Constant for threshold calculation")
    parser.add_argument("start_date", type=str, help="Trading start date yyyy-mm-dd")
    parser.add_argument("end_date", type=str, help="Trading end date yyyy-mm-dd")
    args = parser.parse_args()


    for classifier in classifiers:
        for label in labels:
            if classifier == "HE" and label == "2":
                continue
            call(["python3", "portfolio_simulator.py", classifier, label,str(args.days_window) ,str(args.k), args.start_date, args.end_date])


if __name__ == "__main__":
    main()


