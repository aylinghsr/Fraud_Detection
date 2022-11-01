# Credit Card Fraud Detection using Machine Learning methods

## Introduction
In this project, the aim is to import and use several ML models on [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) (on Kaggle) in order to classify fraudulent and non-fraudulent transactions and compare their performances based on how well they can distinguish fraudulent transactions, rather than merely calculating accuracy-scores.

Since only 0.17% of the transaction records in this dataset are fraudulent, it is not plausible to adopt accuracy-score as the evaluation metric. Although, this metric is calculated, other metrics, such as precision and f1-score, are also used.

In this project, the following ML models are used: Decision Tree, Random Forest, Logistic Regression, Gaussian Naive Bayes, KNN, XGBoost, AdaBoost, and a model ensemble.

The ensemble adopts Majority Voting technique. When creating the ensemble, in 30 iterations, a weighted combination of the mentioned models is created and then, their majority vote is taken. The combination that is created in each iteration consists of 2 to 5 randomly chosen models from a list, sorted by f1-scores.

The evaluation metrics corresponding to each model, or combination of models, is printed.

Finally, the Precision-Recall and ROC Curves of all models are plotted for more convinience.

## Installation
First, you need to clone this repository to your local machine via the following command:
```shell
$ git clone https://github.com/aylinghsr/Fraud_Detection.git
```
In case you don't have `git` installed on your computer, you can download the zip file of this repository and then, extract it.

## Requirements
This project is written in Python3 and requires Scikit-learn, Pandas, and Numpy libraries.

All the required libraries can be installed by running the following command:
```shell
$ pip install -r requirements.txt
```
If the command above results in an error, you can also try:
```shell
$ python -m pip install -r requirements.txt
```
Also, the dataset (.csv file) should be downloaded on your computer.

Dataset: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)

## Usage
Run:
```shell
$ cd Fraud_Detection
$ python Fraud_Detection.py
```
