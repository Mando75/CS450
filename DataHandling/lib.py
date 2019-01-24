from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from kNN.kNNClassifier import kNNClassifier
import numpy as np


def run_model(data, targets, message, regression=False):
    if regression:
        classifiers = [KNeighborsRegressor(n_neighbors=3)]
        c_names = ["Sklearn regression"]
    else:
        classifiers = [
            KNeighborsClassifier(n_neighbors=3),
            kNNClassifier(k=3, use_tree=True, scale=False)
        ]
        c_names = ["Sklearn", "Personal kNN w/ Tree"]

    train_d, test_d, train_t, test_t = train_test_split(
        data, targets, shuffle=True)

    for index, classifier in enumerate(classifiers):
        classifier.fit(train_d, train_t)
        predict = classifier.predict(test_d)
        diff = get_diff(predict, test_t, regression)
        print("#################################")
        print(c_names[index])
        print(message)
        print("#################################")
        if not regression:
            print("Accuracy",
                  round(((test_t.size - len(diff)) / test_t.size) * 100, 2))
        else:
            print("Mean distance from actual")
            print(diff.mean(axis=0)[0])


def get_diff(predicted_t, actual_t, regression=False):
    diff = []
    if not regression:
        for index, predicted in enumerate(predicted_t):
            if predicted != actual_t[index]:
                diff.append({
                    'index': index,
                    'predicted': predicted,
                    'actual': actual_t[index]
                })
    else:
        for index, predicted in enumerate(predicted_t):
            diff.append(
                np.array(
                    [predicted - actual_t[index], predicted, actual_t[index]]))
    return np.array(diff)
