#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:34:18 2018

@author: bullet
"""

from pathlib import Path

import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
                          
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def PlotConfusionMatrix(y_test,y_pred,fig_name):

    cm = confusion_matrix(y_test, y_pred)
    cm_norm = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm_norm)

    fig = plt.figure(figsize=(9, 3))
    ax = fig.add_subplot(1,2,1)
    sns.heatmap(cm, cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True churn type')
    plt.xlabel('Predicted churn type')

    ax = fig.add_subplot(1,2,2)
    sns.heatmap(cm_norm ,cmap='coolwarm_r',linewidths=0.5,annot=True,ax=ax)
    plt.title('Normalized Confusion Matrix')
    plt.ylabel('True churn type')
    plt.xlabel('Predicted churn type')
    plt.show()
    
    plot_path = Path('plots')
    if not plot_path.exists():
        plot_path.mkdir()
    fig.savefig(str(plot_path.joinpath(fig_name + 'confusion_matrix.png')), bbox_inches='tight')
    
    return None

#%%
cv = 10
    
#%%

filename_train = "dataset_train.csv"
filename_test = "dataset_test.csv"

data_train = pd.read_csv(filename_train)
X_test_with_index = pd.read_csv(filename_test)

data_train = data_train.loc[:, data_train.columns != 'index']
X_test = X_test_with_index.loc[:, X_test_with_index.columns != 'index']

X_train = data_train.loc[:, data_train.columns != 'churned_within_30_days']
y_train = data_train.loc[:, data_train.columns == 'churned_within_30_days']

# one-hot encode
cols_to_transform = ['f17', 'f18']
X_train = pd.get_dummies(X_train, columns=cols_to_transform)
X_test = pd.get_dummies(X_test, columns=cols_to_transform)

#%% convert data to numpy arrays
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()
y_train = y_train.as_matrix()
y_train = y_train.ravel()

#%%
mean_monthly_churn_rate_train = (np.count_nonzero(y_train == 1)) / ( np.count_nonzero(y_train == 0) +np.count_nonzero(y_train == 1) )
print("Monthly churn rate: ", mean_monthly_churn_rate_train)

#%% Model logistic regression

steps = [
        ('scaler', StandardScaler()), 
        ('clf', LogisticRegression(max_iter=10, 
                                   random_state=0))
        ]
clf_lr = Pipeline(steps)

parameters = {
        'clf__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'clf__penalty': ['l1', 'l2'],
        'clf__class_weight': ['balanced']
        }

gs = GridSearchCV(
        clf_lr, 
        param_grid=parameters, 
        scoring='f1', 
#        scoring='roc_auc', 
        cv=cv,
        refit=True
        )

gs.fit(X_train, y_train)
print('gs.best_score_:', gs.best_score_)
print('gs.best_params_:', gs.best_params_)

best_clf = gs.best_estimator_

y_pred_lr = best_clf.predict(X_train)

acc_lr = accuracy_score(y_train, y_pred_lr)
print("accuracy: ", 100 * acc_lr)

f1_lr = f1_score(y_train, y_pred_lr)
print("F1 score: ", f1_lr)

print(classification_report(y_train, y_pred_lr))
PlotConfusionMatrix(y_train, y_pred_lr, 'logistic_regression_')

#y_prob_lr = clf_lr.predict_proba(X_train)


#%% Model random forest

steps = [
        ('clf', RandomForestClassifier(random_state=0))
        ]
clf_rf = Pipeline(steps)

parameters = {
        'clf__n_estimators': [100, 500, 1000],
        'clf__class_weight': ['balanced', 'balanced_subsample']
        }

gs = GridSearchCV(
        clf_rf, 
        param_grid=parameters, 
        scoring='f1', 
#        scoring='roc_auc', 
        cv=cv,
        refit=True
        )

gs.fit(X_train, y_train)
print('gs.best_score_:', gs.best_score_)
print('gs.best_params_:', gs.best_params_)

best_clf = gs.best_estimator_

y_pred_rf = best_clf.predict(X_train)

acc_rf = accuracy_score(y_train, y_pred_rf)
print("accuracy: ", 100 * acc_rf)

f1_rf = f1_score(y_train, y_pred_rf)
print("F1 score: ", f1_rf)

print(classification_report(y_train, y_pred_rf))
PlotConfusionMatrix(y_train, y_pred_rf, 'random_forest_')

y_test_rf = best_clf.predict(X_test)

y_test_rf_df = pd.DataFrame()
y_test_rf_df['churned_within_30_days'] = y_test_rf

#data_test_churn_0_1 = pd.concat([X_test_with_index, y_test_rf_df], axis=1)
#data_test_churn_0_1.to_csv("dataset_test.csv")
