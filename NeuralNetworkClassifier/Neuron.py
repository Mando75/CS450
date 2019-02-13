import numpy as np
from operator import sub


class Neuron(object):
    def __init__(self, num_weights):
        self.num_weights = num_weights
        self.weights = list(
            map(lambda _: np.random.uniform(-1.0), range(num_weights)))
        self.previous_weights = [0] * num_weights
        self.weight_deltas = [0] * num_weights
        self.value = 0
        self.previous_value = 0
        self.y_delta = 0

    def calculate_value(self, inputs):
        if len(inputs) is not self.num_weights:
            raise IndexError("Input vs weight count mismatch")

        self.value = 1 / (1 + np.e**(-np.dot(self.weights, inputs)))
        return self.value

    def tweak_weights(self,
                      learning_rate,
                      momentum,
                      previous_layer,
                      target=None,
                      next_layer=None,
                      current_index=None,
                      is_hidden=False):
        self.previous_weights = self.weights

        if is_hidden:
            layer_sum = 0
            for n in next_layer:
                layer_sum += n.y_delta * n.previous_weights[current_index]
            self.y_delta = self.value * (1 - self.value) * layer_sum
        else:
            self.y_delta = self.value * (1 - self.value) * (
                self.value - target)

        self.weight_deltas = list(
            map(
                lambda x: learning_rate * self.y_delta * previous_layer[x] + momentum * self.weight_deltas[x],
                range(self.num_weights)))
        self.weights = list(map(sub, self.weights, self.weight_deltas))

    def __repr__(self):
        return f"\n[value: {self.value}\nweights: {self.weights}]"


class Layer(object):
    def __init__(self, num_neurons, num_inputs):
        self.neurons = [Neuron(num_inputs) for _ in range(num_neurons)]
