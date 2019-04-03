from data_cleanup import load_data
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

data = load_data(use_means=True)

rmse_val = []  #to store rmse values for different k
for K in np.linspace(0.1, 1, 11):
    model = MLPRegressor(max_iter=500, learning_rate='adaptive')

    training_data, testing_data, training_targets, testing_targets = tts(
        data["data"], data["targets"], test_size=.33, shuffle=True)
    model.fit(training_data, training_targets)  #fit the model
    pred = model.predict(testing_data)  #make prediction on test set
    error = sqrt(mean_squared_error(testing_targets, pred))  #calculate rmse
    rmse_val.append(error)  #store rmse values
    print('RMSE value for k= ', K, 'is:', error)

print("MIN RMSE Value: ", min(rmse_val))
plt.plot(rmse_val)
plt.xlabel("K")
plt.ylabel("RMSE")
plt.show()
