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
from config import *
import argparse

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.linear_model import LogisticRegression

def create_df_parameters(df):
    dfres = pd.DataFrame()
    dailyReturns = []
    
    for i in range (0,df.shape[0]):
        dailyReturns.append((df.iloc[i,2]/df.iloc[i,1])-1)
    
    dfres['Date'] = df['Date']
    dfres['Daily return'] = dailyReturns
    return dfres

def get_y_3(fileName):
    
    y = []
    dayCounter = 0
    counter = 0
    dayReturn = 0
    
    
    df = pd.DataFrame()
    df = pd.read_csv(fileName)
    df_parameters = create_df_parameters(df)
    df = df.head(df.shape[0]-365)
    
    
    for i in range(df.shape[0]-1,-1,-1):
        #update parameters
        parameters = computeTradingParameters(df_parameters,dayCounter)
        dayCounter = dayCounter + 1
        
        #compute the label
        label = 2
        dayReturn = (df.iloc[i,2]/df.iloc[i,1])-1
        
        if(dayReturn > parameters[4]):
        #pos label
            y.append(LABELS['POS'])
        elif(dayReturn < parameters[5]):
        #neg label
            y.append(LABELS['NEG'])
        else:
        #normal label
            y.append(LABELS['NORMAL'])
                
    y.pop(0)
    #y.append(float("nan"))
    y.append(LABELS['NORMAL'])
    
    return y

def get_y_2(fileName):
    
    y = []
    dayCounter = 0
    counter = 0
    dayReturn = 0
    
    
    df = pd.DataFrame()
    df = pd.read_csv(fileName)
    df_parameters = create_df_parameters(df)
    df = df.head(df.shape[0]-365)
    
    
    for i in range(df.shape[0]-1,-1,-1):
        #update parameters
        parameters = computeTradingParameters(df_parameters,dayCounter)
        dayCounter = dayCounter + 1
        
        #compute the label
        label = 2
        dayReturn = (df.iloc[i,2]/df.iloc[i,1])-1
        
        if(dayReturn > parameters[4]):
            #pos label
            y.append(LABELS['POS'])
        elif(dayReturn < parameters[5]):
            #neg label
            y.append(LABELS['POS'])
        else:
            #normal label
            y.append(LABELS['NORMAL'])

    y.pop(0)
    #y.append(float("nan"))
    y.append(LABELS['NORMAL'])
    
    return y

def computeTradingParameters(df_parameters, dayCounter):
    parameters = [0.0,0.0,0.0,0.0,0.0,0.0]
    k = 0
    pos = []
    neg = []
    
    for i in range(df_parameters.shape[0]-1-dayCounter,df_parameters.shape[0]-1-dayCounter-365,-1):
        x = df_parameters.iloc[i,1]
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

""" ------------------------------------------------------------------------------- """


def compute_correctness(df):
    x = []
    for i in range(0,df.shape[0]):
        if(df.iloc[i,0]==df.iloc[i,1]):
            x.append(1)
        else:
            x.append(0)

    df['Correctness'] = x

    return df

"""--------------------------------------------"""


def get_classifier_and_grid(classifier):
    if classifier == Classifier.KNN:
        clf = KNeighborsClassifier(n_jobs=-1)
    elif classifier == Classifier.RFC:
        clf = RandomForestClassifier(n_jobs=-1,n_estimators=200,random_state=42,class_weight="balanced")
    elif classifier == Classifier.SVC:
        clf = SVC(gamma="scale", random_state=42, class_weight="balanced")
    elif classifier == Classifier.MLP:
        clf = MLPClassifier(random_state=42,max_iter=10000,early_stopping=True,n_iter_no_change=3)
    elif classifier == Classifier.MNB:
        clf = MultinomialNB()
    elif classifier == Classifier.GNB:
        clf = GaussianNB()
    elif classifier == Classifier.LG:
        clf = LogisticRegression(random_state=42,class_weight="balanced",n_jobs=-1)
    else:
        raise NotImplementedError()

    return clf, PARAM_GRIDS[classifier]

def get_cryptocurrency(cryptocurrency):
    if cryptocurrency == Cryptocurrency.BTC:
        crypto = 0
    elif cryptocurrency == Cryptocurrency.ETH:
        crypto = 1
    elif cryptocurrency == Cryptocurrency.LTC:
        crypto = 2
    return crypto



def main():
    parser = argparse.ArgumentParser(description="AAA")
    parser.add_argument("cryptocurrency", type=Cryptocurrency,choices=list(Cryptocurrency))
    parser.add_argument("classifier", type=Classifier, choices=list(Classifier))
    parser.add_argument("labels", type=int, help="Number of labels (2 or 3)")
    args = parser.parse_args()


    FeaturesFileNames = ['./data/features_datasets/BTCUSD_features.csv','./data/features_datasets/ETHUSD_features.csv']
    FileNames = ['./data/daily_datasets/BTCUSD.csv', './data/daily_datasets/ETHUSD.csv']

    #dataset preparation
    crypto = get_cryptocurrency(args.cryptocurrency)
    
    if args.labels == 2:
        labels = get_y_2(FileNames[crypto])
    elif args.labels == 3:
        labels = get_y_3(FileNames[crypto])
    y = pd.DataFrame()
    y["Labels"] = labels


    x = pd.DataFrame()
    x = pd.read_csv(FeaturesFileNames[crypto])


    Y_train = y.head(y.shape[0]-366)['Labels'].to_numpy()
    Y_test = y.tail(366)['Labels'].to_numpy()


    tmp = x.head(x.shape[0]-366)
    X_train = tmp.tail(tmp.shape[0]-365)
    X_test = x.tail(366)


    print("Train period: " + str(X_train.iloc[0,0]) + "  -  " + str(X_train.iloc[X_train.shape[0]-1,0]) + "  Number of observations: " +  str(X_train.shape[0]))
    print("Test period:  " + str(X_test.iloc[0,0]) + "  -  " + str(X_test.iloc[X_test.shape[0]-1,0]) + "  Number of observations: " +  str(X_test.shape[0]))

    del X_train['Date']
    del X_test['Date']


    # training, validation and test
    clf, param_grid = get_classifier_and_grid(args.classifier)
    pipeline = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
    param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
    gs = GridSearchCV(pipeline,param_grid=param_grid,scoring="f1_weighted",n_jobs=-1,cv=TimeSeriesSplit(n_splits=5))
    gs.fit(X_train, Y_train)
    y_pred = gs.predict(X_test)
    estimator = gs.best_estimator_


    print("SIMULATION ON:",args.cryptocurrency,args.classifier,args.labels)
    print("Best setup from validation:",estimator)
    acc = accuracy_score(Y_test, y_pred)
    print("Weighted accuracy:",acc)
    f1 = f1_score(Y_test, y_pred,average='weighted')
    print("f1 weighted:",f1)
    print("Classiifcation report:")
    print(classification_report(Y_test, y_pred))



    result = pd.DataFrame()
    result['Real'] = Y_test
    result['Forecast'] = y_pred


    result = compute_correctness(result)
    path = './data/labels_datasets/'
    path=path+str(args.cryptocurrency)+"_"+"labels"+"_"+str(args.classifier)+"_"+str(args.labels)+'.xlsx'
    result.to_excel(path, index = False)


    return

if __name__ == "__main__":
    main()


