import pandas as pd
import numpy as np
import os
from shutil import copyfile
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import random
from sklearn.model_selection import cross_validate
import sklearn
from skorch.dataset import Dataset
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
import torch.nn as nn
import torch
import torch.nn.functional as f

device = 'cuda'

net = MLPRegressor(hidden_layer_sizes=(5,),
                                       activation='relu',
                                       solver='adam',
                                       learning_rate='adaptive',
                                       max_iter=20000,
                                       learning_rate_init=0.01,
                                       alpha=0.01)
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

models = {'Linear Regression':linear_model.LinearRegression(), 'SVM Regression':SVR(gamma="scale"), 
            'AdaBoost Regression':AdaBoostRegressor(), "Multilayer Perceptron":net}
for model in models:

    data_features = df.values
    data_features = np.array(data_features[:,2:])

    feat_map = {}
    for idx,name in enumerate(name_col):
        feat_map[name] = data_features[idx]

    X = []
    y = []
    for ind in name_col:
        era = label_map[ind]
        feat = feat_map[ind]

        X.append(feat)
        y.append(era)
    X = np.array(X)
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

    print("\n\nFor " + model + " results for various metrics are as follows:\n")


    print("R2 score: " + str(r2_score))
    print("Explained Variance score: " + str(explained_variance_score))
    print("Max Error score: " + str(max_error_score))
    print("Mean Absolute Error score: " + str(mae_score))
    print("Mean Squared Error score: " + str(mse_score))
    print("Median Absolute Error score: " + str(medae_score))