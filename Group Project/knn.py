from data_cleanup import load_data
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from math import sqrt
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier

data = load_data(use_means=True)

rmse_val = []  #to store rmse values for different k
for K in range(20):
    K = K + 1
    model = KNeighborsRegressor(n_neighbors=K)
    # model = KNeighborsClassifier(n_neighbors=K)

    training_data, testing_data, training_targets, testing_targets = tts(
        data["data"], data["targets"], test_size=.33, shuffle=True)
    model.fit(training_data, training_targets)  #fit the model
    pred = model.predict(testing_data)  #make prediction on test set
    # accuracy = accuracy_score(testing_targets, pred) * 100
    error = sqrt(mean_squared_error(testing_targets, pred))  #calculate rmse
    rmse_val.append(error)  #store rmse values
    print('RMSE value for k= ', K, 'is:', error)

print("MIN RMSE Value: ", min(rmse_val))
plt.plot(rmse_val)
plt.xlabel("K")
plt.ylabel("RMSE")
plt.show()
