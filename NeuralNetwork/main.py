from NeuralNetwork.Neuron import Neuron
import numpy as np


def main():
    num_inputs = 10
    inputs = list(map(lambda _: np.random.uniform(-9.0, 9), range(num_inputs)))
    n = Neuron(num_inputs)
    n.calc_value(inputs)
    print(n.weights)
    print(n.value)
    print(n.activated())
    n.tweak_weights(.2, inputs, target=1)
    n.calc_value(inputs)
    print("Tweaked")
    print(n.weights)
    print(n.value)
    print(n.activated())


if __name__ == '__main__':
    main()
