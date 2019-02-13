from NeuralNetworkClassifier.NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from common.ClassifierTesterIris import ClassifierTesterIris as cti
from sklearn.preprocessing import StandardScaler
import numpy as np


def main():
    iris = load_iris()
    training_data, testing_data, training_targets, testing_targets = tts(
        iris.data, iris.target, shuffle=True, test_size=.33)
    scalar = StandardScaler().fit(training_data)
    training_data = scalar.transform(training_data)
    testing_data = scalar.transform(testing_data)
    n = NeuralNetwork()
    n.make_net(len(np.unique(training_targets)), training_data.shape[1])
    n.fit(training_data, training_targets)
    results = n.predict(testing_data)
    diff = cti.get_diff(results, testing_targets)
    size = testing_targets.size
    accuracy = round(((size - len(diff)) / size) * 100, 2)
    print("DIFF: ", diff)
    print("ACCURACY ", accuracy)


if __name__ == '__main__':
    main()
