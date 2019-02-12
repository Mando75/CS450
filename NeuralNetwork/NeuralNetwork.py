import numpy as np
from NeuralNetwork.Neuron import Neuron


class NeuralNetwork(object):
    def __init__(self, num_neurons=5):
        self.num_neurons = num_neurons
        self.neurons = []

    def fit(self, training_data, training_targets, bias=True):
        num_inputs = training_data.shape[1]
        self.neurons = [Neuron(num_inputs)] * self.num_neurons

    def train(self):
        return 0

    def predict(self, test_data):
        results = [1 for x in range(len(test_data))]
        return np.array(results)
