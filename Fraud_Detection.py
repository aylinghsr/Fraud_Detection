#!/usr/bin/python
#coding: utf-8

"""
Credit Card Fraud Detection
Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
"""

import pandas as pd
from random import sample, choice
from sklearn.preprocessing import RobustScaler as Scaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
from matplotlib import pyplot as plt
from sys import argv

#sometimes LogisticRegression fails to converge
#so, the warning message related to this issue is ignored
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter(action='ignore', category=ConvergenceWarning)

def normalizer(X, features: list):
    '''
    Normalizer function using RobustScaler
    Recieves a dataframe, namely X
    Then, normalizes the given features of this dataframe
    '''
    
    scaler = Scaler()
    X.loc[:, features] = scaler.fit_transform(X.loc[:, features])
    return X

def data_loader(path='creditcard.csv'):
    print("Loading data...")
    df = pd.read_csv(path)
    fraud = df[df['Class'] == 1] #separates the fraud instances
    non_fraud = df[df['Class'] == 0].sample(n=2*fraud.shape[0]) #samples 984 non-fruad instances
    df = pd.concat([fraud, non_fraud]).sample(frac=1) #concatinates these two and shuffles the rows
    X, y = df.drop(df.columns[-1], axis=1), df[df.columns[-1]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=42)
    X_train = normalizer(X_train, ['Time', 'Amount'])
    X_test = normalizer(X_test, ['Time', 'Amount'])
    print("Data loaded successfully!")
    return X_train, X_test, y_train, y_test

def pr_curve_plot(models):
    '''
    Precision-Recall Curve Plotter
    Gets a list of models and plots their PR Curve
    The figures should be closed manually!
    '''
    for i in range(len(models)):
        model = models[i][1]
        pred = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, pred)
        disp = PrecisionRecallDisplay(precision=precision, recall=recall)
        disp.plot()
        plt.title(models[i][0]+' PR Curve')
        plt.show()

def roc_curve_plot(models):
    '''
    ROC Curve Plotter
    Gets a list of models and plots their ROC Curve
    The figures should be closed manually!
    '''
    for i in range(len(models)):
        model = models[i][1]
        pred = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, pred)
        roc_auc = auc(fpr, tpr)
        disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=models[i][0])
        disp.plot()
        plt.title(models[i][0]+' ROC Curve\nArea Under Curve: '+str(roc_auc))
        plt.show()

def main():
    if len(argv) == 1:
        path = input('Path to the dataset: ')
    else:
        path = argv[1]
    
    global X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = data_loader(path)
    
    #Decision Tree
    print('\n---DecisionTree---')
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    dt_f1 = f1_score(y_test, dt_pred)
    print('Accuracy: %.4f' %accuracy_score(y_test, dt_pred))
    print('Recall: %.4f' %recall_score(y_test, dt_pred))
    print('Precision: %.4f' %precision_score(y_test, dt_pred))
    print('F1-score: %.4f' %dt_f1)

    #Random Forest
    print('\n---RandomForest---')
    rf = RandomForestClassifier(n_estimators=200)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred)
    print('Accuracy: %.4f' %accuracy_score(y_test, rf_pred))
    print('Recall: %.4f' %recall_score(y_test, rf_pred))
    print('Precision: %.4f' %precision_score(y_test, rf_pred))
    print('F1-score: %.4f' %rf_f1)

    #Logistic Regression
    print('\n---LogisticRegression---')
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_f1 = f1_score(y_test, lr_pred)
    print('Accuracy: %.4f' %accuracy_score(y_test, lr_pred))
    print('Recall: %.4f' %recall_score(y_test, lr_pred))
    print('Precision: %.4f' %precision_score(y_test, lr_pred))
    print('F1-score: %.4f' %lr_f1)

    #Naive Bayes
    print('\n---GaussianNaiveBayes---')
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    gnb_pred = gnb.predict(X_test)
    gnb_f1 = f1_score(y_test, gnb_pred)
    print('Accuracy: %.4f' %accuracy_score(y_test, gnb_pred))
    print('Recall: %.4f' %recall_score(y_test, gnb_pred))
    print('Precision: %.4f' %precision_score(y_test, gnb_pred))
    print('F1-score: %.4f' %gnb_f1)

    #K-Nearest Neighbors    
    print('\n---KNN---')
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    knn_f1 = f1_score(y_test, knn_pred)
    print('Accuracy: %.4f' %accuracy_score(y_test, knn_pred))
    print('Recall: %.4f' %recall_score(y_test, knn_pred))
    print('Precision: %.4f' %precision_score(y_test, knn_pred))
    print('F1-score: %.4f' %knn_f1)

    #Extreme Gradient Boosting
    print('\n---XGBoost---')
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_f1 = f1_score(y_test, xgb_pred)
    print('Accuracy: %.4f' %accuracy_score(y_test, xgb_pred))
    print('Recall: %.4f' %recall_score(y_test, xgb_pred))
    print('Precision: %.4f' %precision_score(y_test, xgb_pred))
    print('F1-score: %.4f' %xgb_f1)
    
    #Adaptive Gradient Boosting
    print('\n---AdaBoost---')
    ab = AdaBoostClassifier()
    ab.fit(X_train, y_train)
    ab_pred = ab.predict(X_test)
    ab_f1 = f1_score(y_test, ab_pred)
    print('Accuracy: %.4f' %accuracy_score(y_test, ab_pred))
    print('Recall: %.4f' %recall_score(y_test, ab_pred))
    print('Precision: %.4f' %precision_score(y_test, ab_pred))
    print('F1-score: %.4f' %ab_f1)
    
    #models are sorted by their f1-score and the voting ensemble is created randomly
    models = [('LR', lr, lr_f1), ('RF', rf, rf_f1), ('KNN', knn, knn_f1), ('XGB', xgb, xgb_f1), ('GNB', gnb, gnb_f1), ('DT', dt, dt_f1), ('AB', ab, ab_f1)]
    models = sorted(models, key=lambda x: x[-1], reverse=True)
    voting_better = False
    best_f1, best_estimators, best_pred = 0.0, [], []
    print('\nPlease wait while voting is taking place...')
    for _ in range(30):
        k = choice([2, 3, 4, 5])
        estimators = sample(models, k)
        estimators = sorted(estimators, key=lambda x: x[-1], reverse=True)
        estimators = [x[:2] for x in estimators]
        weights = [1 for _ in range(k)]
        weights[0], weights[1] = 5, 4
        voting = VotingClassifier(estimators=estimators, weights=weights, voting='soft')
        voting.fit(X_train, y_train)
        voting_pred = voting.predict(X_test)
        v_f1 = f1_score(y_test, voting_pred)
        if v_f1 > best_f1:
            #best performing ensemble is saved
            best_f1 = v_f1
            best_estimators = [x[0] for x in estimators]
            best_pred = voting_pred.copy()
        if not voting_better and best_f1 >= models[0][2]:
            voting_better = True

    print('\n---Voting---')
    print('Voting ensemble consists of ' + ', '.join(best_estimators))
    print('Accuracy: %.4f' %accuracy_score(y_test, best_pred))
    print('Recall: %.4f' %recall_score(y_test, best_pred))
    print('Precision: %.4f' %precision_score(y_test, best_pred))
    print('F1-score: %.4f' %best_f1)
    
    models = [('DT', dt), ('RF', rf), ('LR', lr), ('GNB', gnb), ('KNN', knn), ('XGB', xgb), ('AB', ab), ('Voting', voting)]
    pr_curve_plot(models)
    roc_curve_plot(models)
    
    if voting_better:
        print('\nVoting yields the best result!')    
    else:
        print(f'\nThe chosen model in terms of performance is {models[0][0]}')

if __name__ == '__main__':
    main()
