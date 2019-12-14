# PitchFX_Classifier

To run this code, you will need to use python3 with common packages such as scikit-learn, pandas, numpy and joblib.
All data used can be found in the Data folder and all labels (ERA, xFIP etc) can be found in Labels folder.
The Scalers and PCA_Models folder contained the saved feature selection models.
Models contain all pretrained models.
Cleaning scripts for the data are also included.

After you have downloaded this code packet:
---------------------

To train the models: 
- Change directory Model_Generation.
- Open file metric_prediction_multi.py.
- There is a variable future, set this to False if you want to generate models to predict current year labels from current year data or set to True if you want to train models to predict future year's labels from current year's data. This is all you need to change. This repo has our pretrained networks, if you want to train your own for the purpose of grading just delete and recreate the Models, PCA_Models and Scalers folders.
- python3 metric_prediction_multi.py.


To test these generated models: 
- Change directory to Testing.
- Open file test_models.py.
- Again edit future variable as above to switch between the two settings; present and future.
- python3 test_models.py.
- This should output all results (see table 1 in the paper).

To predict 2020 statistics:
- python3 predict_future.py.
- If you want to change the metric it outputs the best players for just open the file and edit the interesting_metrics list to one of the other available metrics.
- Note that the models used to predict the future are different from the ones used to produce the results in the paper. The reason is that to predict the 2020 season we retrained all the models that were deemed best for future prediction but now included the previous testing data such that we used as much data as possible to predict 2020. These models were not used at all in the results in the paper as this would mean the training and testing were not disjoint. This was just a fun exercise.


