from sklearn.neural_network import MLPRegressor
from model_runner import model_runner


def model_generator(k):
    return MLPRegressor(
        max_iter=2000,
        activation='logistic',
        solver='adam',
        learning_rate='adaptive',
    )


model_runner(model_generator)

# Top Scores

# no core, no teacher
# RMSE: 7.14069
# Classification : 75.083

# no teacher
# RMSE: 9.10
# Classification: 70.1

# no core
# RMSE: 5.45689
# Classification: 82.2259
