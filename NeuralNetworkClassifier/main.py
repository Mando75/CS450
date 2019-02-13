from NeuralNetworkClassifier.NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split as tts
from common.ClassifierTesterIris import ClassifierTesterIris as cti
from sklearn.preprocessing import StandardScaler
import numpy as np
from common.progressBar import printProgressBar


def get_params():
    num_tests = int(input("\nEnter number of tests (int): "))
    learning_rate = float(input("\nEnter learning Rate (.2-.4): "))
    momentum = float(input("\nEnter Momentum (.1-.9): "))
    error_change_per = float(input("\nEnter Err Change Percent (.1-.5): "))
    epoch_iter = int(input("\nEnter # Epoch Iterations (int): "))
    return num_tests, learning_rate, momentum, error_change_per, epoch_iter


def main():
    iris = load_iris()
    num_tests, learning_rate, momentum, error_change_per, epoch_iter = get_params(
    )
    avg_acc = 0
    print("\nStarting tests")
    printProgressBar(0, num_tests, prefix="Progress", suffix="Complete")
    for i in range(num_tests):
        printProgressBar(
            i + 1, num_tests, prefix="Progress", suffix="Complete")
        training_data, testing_data, training_targets, testing_targets = tts(
            iris.data, iris.target, shuffle=True, test_size=.33)
        scalar = StandardScaler().fit(training_data)
        training_data = scalar.transform(training_data)
        testing_data = scalar.transform(testing_data)
        n = NeuralNetwork()
        n.make_net(len(np.unique(training_targets)), training_data.shape[1])
        n.fit(training_data, training_targets, learning_rate, momentum,
              error_change_per, epoch_iter)
        results = n.predict(testing_data)
        diff = cti.get_diff(results, testing_targets)
        size = testing_targets.size
        accuracy = round(((size - len(diff)) / size) * 100, 2)
        avg_acc += accuracy
    print("ACCURACY ", avg_acc / num_tests)


if __name__ == '__main__':
    main()
