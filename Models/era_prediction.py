import pandas as pd
import numpy as np
import os
from shutil import copyfile
from sklearn import linear_model
import random

from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

pitch_types = ['SL','FF','CU','FT','CH','FC','KC','SI','PO','FS']

r2_scores = {}
max_error_scores = {}
mae_scores = {}
mse_scores = {}
explained_variance_scores = {}
medae_scores = {}

# net = MLPRegressor(hidden_layer_sizes=(5,),
#                                        activation='relu',
#                                        solver='adam',
#                                        learning_rate='adaptive',
#                                        max_iter=20000,
#                                        learning_rate_init=0.01,
#                                        alpha=0.01)
net = MLPRegressor()

models = {'Linear Regression':linear_model.LinearRegression(), 'SVM Regression':SVR(gamma="scale"),
          'AdaBoost Regression':AdaBoostRegressor(), 'Multilayer Perceptron':net}
for pitch_type in pitch_types:
    pitch_data = []
    print(pitch_type)
    for column in df.columns.values:
        if column == 'Name' or column == 'Unnamed: 0':
            continue

        if pitch_type in column:
            data_feature = np.array(df[[column]].values.tolist())[:, 0]
            pitch_data.append(data_feature)
    # pitch_data.append(name_col)
    pitch_data = np.transpose(np.array(pitch_data))
    useless_indices = ~np.all(pitch_data == 0, axis=1)
    pitch_data = pitch_data[useless_indices]
    y = name_col[useless_indices]
    y_scores = []
    for n in y:
        y_scores.append(label_map[n])

    X_train, X_test, y_train, y_test = train_test_split(pitch_data, y_scores, test_size = 0.33, random_state = 42)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    print('Number of elements in test set: ' + str(len(X_test)))
    scaler = []

    pca = PCA(n_components=10)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    ninefive = 0
    num_feats = 0
    for val in pca.explained_variance_ratio_:
        ninefive = ninefive + val
        num_feats = num_feats + 1
        if ninefive >= 0.9:
            break
    X_train = X_train[:, :num_feats]
    X_test = X_test[:, :num_feats]

    for model_name in models:
        model = models[model_name]
        reg = model.fit(X_train, y_train)
        predictions = reg.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = reg.score(X_test, y_test)

        print("Model: " + model_name + "; Pitch Type: " + pitch_type + ": MAE: " + str(mae) + ", MSE: " + str(mse)
              + ", R2 score: " + str(r2))
