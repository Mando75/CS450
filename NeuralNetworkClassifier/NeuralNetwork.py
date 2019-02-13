from NeuralNetworkClassifier.Neuron import Layer


class NeuralNetwork(object):
    def __init__(self):
        self.layers = []

    def fit(self,
            training_data,
            training_targets,
            learning_rate=0.2,
            momentum=0.9,
            error_change_percent=0.1,
            epoch_iter=250):
        error_data = []
        first = True
        alive = True
        limit = epoch_iter
        while alive and limit:
            instance_mean_err = 0
            for i, row in enumerate(training_data):
                self.compute_results(row)
                targets = [
                    1 if x == training_targets[i] else 0
                    for x in range(len(training_targets))
                ]
                instance_mean_err += self.update_neurons(
                    row, targets, learning_rate, momentum)
            instance_mean_err /= len(training_data)
            error_data.append(abs(instance_mean_err))

            if first:
                first = False
                continue
            else:
                per = error_data[len(error_data) -
                                 1] / error_data[len(error_data) - 2]
                if per < error_change_percent:
                    alive = False
            limit -= 1
        return error_data

    def predict(self, test_data):
        results = []
        predictions = []

        for row in test_data:
            self.compute_results(row)

            for n in self.layers[-1].neurons:
                results.append(n.value)

            predictions.append(results.index(max(results)))
            results.clear()
        return predictions

    def make_net(self, target_count, input_count, hidden_layers=[]):
        if len(hidden_layers) is 0:
            self.layers.append(Layer(target_count, input_count + 1))
            return

        for x in range(len(hidden_layers)):
            if x is 0:
                self.layers.append(Layer(hidden_layers[x], input_count + 1))
            else:
                self.layers.append(
                    Layer(hidden_layers[x],
                          len(self.layers[x - 1].neurons) + 1))

        self.layers.append(
            Layer(target_count,
                  len(self.layers[-1].neurons) + 1))

    def compute_results(self, inputs):
        working_inputs = []
        for layer in self.layers:
            for neuron in layer.neurons:
                if layer is self.layers[0]:
                    neuron.calculate_value([-1] + inputs.tolist())
                else:
                    neuron.calculate_value([-1] + working_inputs)
            working_inputs.clear()
            for neuron in layer.neurons:
                working_inputs.append(neuron.value)

    def update_neurons(self, inputs, target, learning_rate, momentum):
        first = True
        avg_layer_err = 0
        for layer in range(len(self.layers) - 1, -1, -1):
            layer_o = self.layers[layer]
            avg_err = 0
            if first and layer is 0:
                for i, n in enumerate(layer_o.neurons):
                    n.tweak_weights(
                        learning_rate,
                        momentum, [-1] + inputs.tolist(),
                        target=target[i])
                    avg_err += n.y_delta
                first = False
                avg_err /= len(layer_o.neurons)
            elif first:
                for i, n in enumerate(layer_o.neurons):
                    n.tweak_weights(
                        learning_rate,
                        momentum, [-1] +
                        [n.value for n in self.layers[layer - 1].neurons],
                        target=target[i])
                    first = False
                    avg_err += n.y_delta
            elif layer is 0:
                for i, n in enumerate(layer_o.neurons):
                    n.tweak_weights(
                        learning_rate,
                        momentum, [-1] + inputs.tolist(),
                        next_layer=self.layers[layer + 1].neurons,
                        current_index=i + 1,
                        is_hidden=True)
            else:
                for i, n in enumerate(layer_o.neurons):
                    n.tweak_weights(
                        learning_rate,
                        momentum, [-1] +
                        [n.value for n in self.layers[layer - 1].neurons],
                        next_layer=self.layers[layer + 1].neurons,
                        current_index=i + 1,
                        is_hidden=True)
            avg_layer_err += avg_err
        avg_layer_err /= len(self.layers)
        return avg_layer_err
