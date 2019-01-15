import numpy as np


class kNNClassifier:
    def __init__(self, k=1):
        self.k = k
        self.data = np.array([])
        self.targets = np.array([])

    def fit(self, training_data, training_targets):
        self.data = training_data
        self.targets = training_targets

    def predict(self, testing_data):
        n_inputs = np.shape(testing_data)[0]
        closest = np.zeros(n_inputs)

        for n in range(n_inputs):
            distances = np.sum((self.data - testing_data[n, :])**2, axis=1)

            indices = np.argsort(distances, axis=0)

            classes = np.unique(self.targets[indices[:self.k]])
            if len(classes) == 1:
                closest[n] = np.unique(classes)
            else:
                counts = np.zeros(max(classes) + 1)

                counts = list(
                    map(lambda index: counts[self.targets[index]] + 1,
                        indices))

                closest[n] = np.max(counts)

        return closest
