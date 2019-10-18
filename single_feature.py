import pandas as pd
import numpy as np
import os
from shutil import copyfile
from sklearn import linear_model
import random
from sklearn.model_selection import cross_validate
import sklearn
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor

# Input CSV containing all info on files that are enrolled
import os.path

info = '2018_data.csv'

# Read in data
df = pd.read_csv(info)

l = '2018_labels.csv'

labels = pd.read_csv(l)

names = np.array(labels[['Name']].values.tolist())
names = names[:,0]
era = np.array(labels[['ERA']].values.tolist())
era = era[:, 0]

label_map = {}
for idx,name in enumerate(names):
    label_map[name] = era[idx]

data = df.values

len_data = len(data)

name_col = np.array(df[['Name']].values.tolist())[:,0]

r2_scores = {}
max_error_scores = {}
mae_scores = {}
mse_scores = {}
explained_variance_scores = {}
medae_scores = {}

models = {'Linear Regression':linear_model.LinearRegression(), 'SVM Regression':SVR(gamma="scale"), 'AdaBoost Regression':AdaBoostRegressor()}
for model in models:
    for column in df.columns.values:
        if column == 'Name' or column == 'Unnamed: 0':
            continue
        
        data_feature = np.array(df[[column]].values.tolist())[:,0]

        feat_map = {}
        for idx,name in enumerate(name_col):
            feat_map[name] = data_feature[idx]

        X = []
        y = []
        for ind in name_col:
            era = label_map[ind]
            feat = feat_map[ind]

            X.append(feat)
            y.append(era)
        X = np.array(X).reshape(-1, 1)
        y = np.array(y).ravel()

        m = models[model]
        scoring_methodologies = ['r2', 'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_median_absolute_error']
        cv_results = cross_validate(m, X, y, cv=5, scoring=scoring_methodologies)

        r2_score = -1 * np.mean(cv_results['test_r2'])
        explained_variance_score = np.mean(cv_results['test_explained_variance'])
        max_error_score = -1 * np.mean(cv_results['test_max_error'])
        mae_score = -1 * np.mean(cv_results['test_neg_mean_absolute_error'])
        mse_score = -1 * np.mean(cv_results['test_neg_mean_squared_error'])
        medae_score = -1 * np.mean(cv_results['test_neg_median_absolute_error'])

        r2_scores[column] = r2_score
        explained_variance_scores[column] = explained_variance_score
        max_error_scores[column] = max_error_score
        mae_scores[column] = mae_score
        mse_scores[column] = mse_score
        medae_scores[column] = medae_score

    print("\n\nFor " + model + " results for various metrics are as follows:\n")

    print("Mean Absolute Error")
    for count, w in enumerate(sorted(mae_scores, key=mae_scores.get, reverse=False)):

        if count < 5:
            print(w, mae_scores[w])
        else:
            break

    print("\nMean Squared Error")
    for count, w in enumerate(sorted(mse_scores, key=mse_scores.get, reverse=False)):

        if count < 5:
            print(w, mse_scores[w])
        else:
            break

    print("\nExplained Variance")
    for count, w in enumerate(sorted(explained_variance_scores, key=explained_variance_scores.get, reverse=True)):

        if count < 5:
            print(w, explained_variance_scores[w])
        else:
            break

    print("\nMax Error")
    for count, w in enumerate(sorted(max_error_scores, key=max_error_scores.get, reverse=False)):

        if count < 5:
            print(w, max_error_scores[w])
        else:
            break

    print("\nMedian Absolute Error")
    for count, w in enumerate(sorted(medae_scores, key=medae_scores.get, reverse=False)):

        if count < 5:
            print(w, medae_scores[w])
        else:
            break
