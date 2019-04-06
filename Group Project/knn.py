from model_runner import model_runner
from sklearn.neighbors import KNeighborsRegressor


def model_generator(k):
    return KNeighborsRegressor(
        n_neighbors=k, weights='uniform', algorithm='auto', p=2)


model_runner(model_generator)
