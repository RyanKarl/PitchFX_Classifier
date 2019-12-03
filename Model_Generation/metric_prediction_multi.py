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

# Input CSV containing all info on files that are enrolled
info = '../Data/2018clean.csv'

# Read in data
df = pd.read_csv(info)

# CSV file containing the metrics we want to estimate using name as a key
l = '../Labels/Players_Stats_2018.csv'

labels = pd.read_csv(l)

# Extract names of players in a given year
names = np.array(labels[['Name']].values.tolist())
names = names[:,0]

# The metrics we are interested in predicting
# intersting_metrics = ['ERA', 'xFIP', 'K/9', 'H/9', 'AVG', 'BABIP', 'GB%']
intersting_metrics = ['ERA']

# WHICH METRIC WE WANT TO CREATE A MODEL FOR
for metric_to_estimate in intersting_metrics:
    print(metric_to_estimate)
    # Extract data for that metric
    metric = np.array(labels[[metric_to_estimate]].values.tolist())
    metric = metric[:, 0]

    # Make a map where the key is the player name and the value is the metric for that player
    label_map = {}
    for idx,name in enumerate(names):
        label_map[name] = metric[idx]

    # Get data
    data = df.values

    len_data = len(data)

    # Get names in order they appear in the data csv
    name_col = np.array(df[['Name']].values.tolist())[:,0]

    # All pitch types we are interested in
    pitch_types = ['SL','FF','CU','FT','CH','FC','KC','SI','PO','FS']

    # IF YOU WANT TO PLAY AROUND WITH THE NETWORK PARAMETERS
    # net = MLPRegressor(hidden_layer_sizes=(5,),
    #                                        activation='relu',
    #                                        solver='adam',
    #                                        learning_rate='adaptive',
    #                                        max_iter=20000,
    #                                        learning_rate_init=0.01,
    #                                        alpha=0.01)
    net = MLPRegressor(max_iter=20000) # otherwise use default params

    # All models we want to test, use default parameters for all
    models = {'Linear Regression':linear_model.LinearRegression(), 'SVM Regression':SVR(gamma="scale"),
            'AdaBoost Regression':AdaBoostRegressor(), 'Multilayer Perceptron':net}

    # For every pitch we are concerned with
    for pitch_type in pitch_types:
        # to store all useful data
        pitch_data = []
        print("\n\n" + pitch_type)
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
            y_scores.append(label_map[n])

        mse_scores = {'Linear Regression':[], 'SVM Regression':[],
            'AdaBoost Regression':[], 'Multilayer Perceptron':[]}
        mae_scores = {'Linear Regression':[], 'SVM Regression':[],
            'AdaBoost Regression':[], 'Multilayer Perceptron':[]}
        r2_scores = {'Linear Regression':[], 'SVM Regression':[],
            'AdaBoost Regression':[], 'Multilayer Perceptron':[]}

        for random_split in range(10):

            # Create training/testing split
            X_train, X_test, y_train, y_test = train_test_split(pitch_data, y_scores, test_size = 0.33, random_state = 10*random_split)

            # Normalize all features between 0 and 1 independently
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            # print('Number of elements in test set: ' + str(len(X_test)))

            # Reduce dimensionality to 90% of the explained variance or 10, whichever is minimum
            pca = PCA(n_components=10)
            pca.fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)
            expl_var = 0
            num_feats = 0
            for val in pca.explained_variance_ratio_:
                expl_var = expl_var + val
                num_feats = num_feats + 1
                if expl_var >= 0.9:
                    break
            X_train = X_train[:, :num_feats]
            X_test = X_test[:, :num_feats]

            # For all models we want to test

            
            for model_name in models:
                model = models[model_name]                              # Get model
                reg = model.fit(X_train, y_train)                       # Train model
                predictions = reg.predict(X_test)                       # Predict metric

                mse = mean_squared_error(y_test, predictions)           # Mean Squared Error
                mae = mean_absolute_error(y_test, predictions)          # Mean Absolute Error
                r2 = reg.score(X_test, y_test)                          # R2 Score

                mse_scores[model_name] = mse
                mae_scores[model_name] = mae
                r2_scores[model_name] = r2
        
        min_mae = float("inf")
        mae_model = ''
        min_mse = float("inf")
        mse_model = ''
        max_r2 = float("-inf")
        r2_model = ''
        for model_name in models:
            
            mse = np.mean(mse_scores[model_name])
            mae = np.mean(mae_scores[model_name])
            r2 = np.mean(r2_scores[model_name])

            if mse < min_mse:
                min_mse = mse
                mse_model = model_name

            if mae < min_mae:
                min_mae = mae
                mae_model = model_name

            if r2 > max_r2:
                max_r2 = r2
                r2_model = model_name
                # print("Model: " + model_name + "; Pitch Type: " + pitch_type + ": MAE: " + str(mae) + ", MSE: " + str(mse)
                #       + ", R2 score: " + str(r2))                       # Print scores

        print("Best model to minimize MAE: " + mae_model + ", MAE = " + str(min_mae))
        print("Best model to minimize MSE: " + mse_model + ", MSE = " + str(min_mse))
        print("Best model to minimize R2: " + r2_model + ", R2 = " + str(max_r2))

        best_models_all = [mse_model, mae_model, r2_model]
        if len(set(best_models_all)) == 3:
            best_model = mae_model
        else:
            best_model = mode(best_models_all)

        print("\nOverall Best Model for " + pitch_type + " is " + best_model)

        model = models[model_name]                              # Get model
        reg = model.fit(pitch_data, y_scores)                   # Train model
        dump(reg, "../Models/" + pitch_type + "_" + best_model.replace(" ", "_") + '.joblib') 
