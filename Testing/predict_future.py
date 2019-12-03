import pandas as pd
import numpy as np
from sklearn import linear_model

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

from statistics import mode
from joblib import dump, load
from data_loader import get_data
import os

# The metrics we are interested in predicting
intersting_metrics = ['xFIP']#, 'xFIP']#, 'K/9', 'H/9', 'AVG', 'BABIP', 'GB%']

# All pitch types we are interested in
pitch_types = ['SL', 'FF', 'CU', 'FT', 'CH', 'FC', 'KC', 'SI', 'PO', 'FS']
# intersting_metrics = ['ERA']
future = True

# WHICH METRIC WE WANT TO CREATE A MODEL FOR
all_results = {}
final_results = {}
first_time = True
predictions = {}

for metric_to_estimate in intersting_metrics:

    df, labels = get_data(2019, 2020, target=metric_to_estimate, future=False)

    # Extract names of players in a given year
    names = np.array(labels[['Name']].values.tolist())
    names = names[:, 0]
    print(metric_to_estimate)
    # Extract data for that metric
    metric = np.array(labels[[metric_to_estimate]].values.tolist())
    metric = metric[:, 0]

    # Make a map where the key is the player name and the value is the metric for that player
    label_map = {}
    for idx, name in enumerate(names):
        if first_time:
            all_results[name] = {}
            final_results[name] = {}

        label_map[name] = metric[idx]
    first_time = False
    # Get data
    data = df.values

    len_data = len(data)

    # Get names in order they appear in the data csv
    name_col = np.array(df[['Name']].values.tolist())[:, 0]

    # For every pitch we are concerned with
    for pitch_type in pitch_types:

        predictions[pitch_type] = []

        # to store all useful data
        pitch_data = []
        # print("\n\n" + pitch_type)
        # For all features in the data
        for column in df.columns.values:
            # Unused columns
            if column == 'Name' or column == 'Unnamed: 0':
                continue
            # If that feature corresponds to the current pitch type
            if pitch_type in column:
                data_feature = np.array(df[[column]].values.tolist())[:, 0]
                pitch_data.append(data_feature)

        # Get it so it is a num_players x num_features array instead of num_features x num_players
        pitch_data = np.transpose(np.array(pitch_data))
        # Remove pitchers that dont throw the current pitch type (i.e. all features for pitch = 0)
        useless_indices = ~np.all(pitch_data == 0, axis=1)
        pitch_data = pitch_data[useless_indices]
        y = name_col[useless_indices]

        # Change from player names into the metric we want to estimate
        y_scores = []
        for n in y:
            all_results[n][pitch_type] = []
            y_scores.append(label_map[n])

        if future == False:
            subdir = "Present"
        else:
            subdir = "Future_Incl2019"
        model_location = "../Models/" + metric_to_estimate + "/" + subdir + "/"
        scaler_location = "../Scalers/" + metric_to_estimate + "/" + subdir + "/"
        pca_location = "../PCA_Models/" + metric_to_estimate + "/" + subdir + "/"
        for model_name in os.listdir(model_location):
            if pitch_type + "_" in model_name:

                model = load(model_location + model_name)
                scaler = load(scaler_location + model_name)
                pca = load(pca_location + model_name)

                for id_data, entry in enumerate(pitch_data):
                    label = y_scores[id_data]
                    name = y[id_data]
                    data = scaler.transform(entry.reshape(1, -1))
                    data = pca.transform(data)
                    prediction = model.predict(data)
                    all_results[name][pitch_type].append(prediction)


    for player in all_results:
        methods = all_results[player]
        votes = []
        for vote in methods:
            val = methods[vote]
            votes.append(val)
        votes = [x for x in votes if x != []]
        estimated_metric = np.mean(votes)
        final_results[player][metric_to_estimate] = estimated_metric

best_pitchers = {}
for pitcher in final_results:
    estimations = final_results[pitcher]
    diffs = []
    for metric in estimations:
        val = estimations[metric]
        diffs.append(val)

    best_pitchers[pitcher] = np.mean(diffs)

sorted_x = sorted(best_pitchers.items(), key=lambda kv: kv[1], reverse=True)

for i in range(20):
    print(sorted_x[i][0])
    print("Predicted: " + str(final_results[sorted_x[i][0]]))




