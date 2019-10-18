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
    for c1, column1 in enumerate(df.columns.values):
        if column1 == 'Name' or column1 == 'Unnamed: 0':
            continue
        for c2, column2 in enumerate(df.columns.values):
            if column2 == 'Name' or column2 == 'Unnamed: 0':
                continue
            
            if column1 == column2:
                continue
            elif c2 <= c1:
                continue
        
            data_feature1 = np.array(df[[column1]].values.tolist())[:,0]
            data_feature2 = np.array(df[[column2]].values.tolist())[:,0]

            feat_map1 = {}
            feat_map2 = {}
            for idx,name in enumerate(name_col):
                feat_map1[name] = data_feature1[idx]
                feat_map2[name] = data_feature2[idx]

            X = []
            y = []
            for ind in name_col:
                era = label_map[ind]
                feat1 = feat_map1[ind]
                feat2 = feat_map2[ind]

                X.append([feat1, feat2])
                y.append(era)
            X = np.array(X).reshape(-1, 2)
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

            r2_scores[column1 + "-" + column2] = r2_score
            explained_variance_scores[column1 + "-" + column2] = explained_variance_score
            max_error_scores[column1 + "-" + column2] = max_error_score
            mae_scores[column1 + "-" + column2] = mae_score
            mse_scores[column1 + "-" + column2] = mse_score
            medae_scores[column1 + "-" + column2] = medae_score


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
