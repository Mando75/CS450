import numpy as np
from operator import sub


class Neuron(object):
    def __init__(self, num_weights):
        self.value = 0
        self.num_weights = num_weights
        # Set the initial weights as something random
        self.weights = list(
            map(lambda _: np.random.uniform(-1.0), range(num_weights)))
        # To store the change in weights
        self.weight_deltas = [0] * (num_weights + 1)
        # to store the previous weight values after tweak
        self.previous_weights = [0] * (num_weights + 1)
        self.y_delta = 0

    def calc_value(self, inputs):
        if len(inputs) != len(self.weights):
            raise IndexError("Length mismatch: inputs vs weights")

        # Using linear activation function for now... will probably
        # switch to sigmoid or tanh
        self.value = np.sum(np.dot(self.weights, inputs))
        return self.value

    def activated(self):
        return 1 if self.value >= 0 else 0

    def tweak_weights(self, learning_rate, inputs, target):
        self.previous_weights = self.weights
        self.y_delta = learning_rate * (self.value - target)
        self.weight_deltas = list(
            map(lambda x: self.weights[x] - (self.y_delta * inputs[x]),
                range(self.num_weights)))
        self.weights = list(map(sub, self.weights, self.weight_deltas))
