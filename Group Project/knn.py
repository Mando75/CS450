from data_cleanup import load_data
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pandas as pd
from math import sqrt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor

data = load_data(use_means=True)

rmse_val = []  #to store rmse values for different k
acc_scores = []
for K in range(20):
    K = K + 1
    model = KNeighborsRegressor(n_neighbors=K)

    training_data, testing_data, training_targets, testing_targets = tts(
        data["data"], data["targets"], test_size=.20, shuffle=True)
    model.fit(training_data, training_targets)  #fit the model
    pred = model.predict(testing_data)  #make prediction on test set
    error = sqrt(mean_squared_error(testing_targets, pred))  #calculate rmse
    rmse_val.append(error)  #store rmse values
    grading_scheme_bins = [-1000.0, 60.0, 70.0, 80.0, 90.0, 500.0]
    pred_letters = pd.cut(
        pred,
        grading_scheme_bins,
        labels=False,
        right=False,
        include_lowest=True)
    testing_letters = pd.cut(
        testing_targets.values,
        grading_scheme_bins,
        labels=False,
        right=False,
        include_lowest=True)
    acc_scores.append(accuracy_score(testing_letters, pred_letters) * 100)

print("MIN RMSE Value: ", min(rmse_val))
print("MAX Classification accuracy: ", max(acc_scores))
plt.plot(rmse_val)
plt.plot(acc_scores)
plt.xlabel("K")
plt.show()

# Top Scores
# RMSE: 13.23
# Class: 67.28
