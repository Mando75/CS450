import numpy as np


class Dataset(object):
    """ All your hopes and dreams come true with Dataset! """

    def __init__(self):
        self.DESCR = ""
        self.data = np.array([])
        self.target = np.array([])
        self.input_count = 0
        self.target_count = 0
        self.training_data = np.array([])
        self.test_data = np.array([])
        self.training_targets = np.array([])
        self.test_targets = np.array([])
        self.predicted_targets = np.array([])
        self.means = np.array([])
        self.standard_devs = np.array([])

    def randomize_data(self):
        reorder = np.random.permutation(len(self.data))
        self.data = self.data[reorder]
        self.target = self.target[reorder]

    def split_data(self, training_percent=70):
        # Default 70/30, can change
        training_size = round(len(self.data) * (training_percent / 100))
        self.training_data, self.test_data = np.split(self.data,
                                                      [training_size])
        self.training_targets, self.test_targets = np.split(
            self.target, [training_size])

    def standardize_data(self):
        standardized_data = self.training_data.T
        standardized_test_data = self.test_data.T
        col_means = []
        col_stds = []

        for x in range(len(standardized_data)):
            col_means.append(np.mean(standardized_data[x]))
            col_stds.append(np.std(standardized_data[x]))
            standardized_data[x] = [(el - col_means[x]) / col_stds[x]
                                    for el in standardized_data[x]]

        for x in range(len(standardized_test_data)):
            standardized_test_data[x] = [(el - col_means[x]) / col_stds[x]
                                         for el in standardized_test_data[x]]

        self.training_data = standardized_data.T
        self.test_data = standardized_test_data.T
        self.means = np.array(col_means)
        self.standard_devs = np.array(col_stds)
        self.input_count = self.training_data.shape[1]
        self.target_count = len(np.unique(self.training_targets))

    def discretize_data(self, sections=30):
        # inefficient but... ¯\_(ツ)_/¯ Discretize!
        self.rules = []

        for col in range(len(self.training_data[0])):
            low = min(self.training_data[:, col])
            high = max(self.training_data[:, col])
            rule = np.arange(low, high, (high - low) / sections)
            self.rules.append(rule)

            for i, item in enumerate(self.training_data[:, col]):
                if item < rule[0]:
                    self.training_data[:, col][i] = -1
                    continue
                for x, bound in enumerate(rule):
                    if x == 0:
                        continue
                    if item < bound:
                        self.training_data[:, col][i] = x - 1
                        break
                    else:
                        self.training_data[:, col][i] = x
                else:
                    continue

            # Never copy and paste in code, they say
            for i, item in enumerate(self.test_data[:, col]):
                if item < rule[0]:
                    self.test_data[:, col][i] = -1
                    continue
                for x, bound in enumerate(rule):
                    if x == 0:
                        continue
                    if item < bound:
                        self.test_data[:, col][i] = x - 1
                        break
                    else:
                        self.test_data[:, col][i] = x
                else:
                    continue

    def report_accuracy(self):
        correct = 0
        for i in range(len(self.test_targets)):
            if self.test_targets[i] == self.predicted_targets[i]:
                correct += 1
        percentage = round(correct / len(self.test_targets), 2) * 100
        print("Predicting targets at {}% accuracy".format(percentage))

    # All the loads will be here. Hardcoded in so that Experiment.py is cleaner
    # Discretization is done in the load file if needed

    def load_iris(self):
        with open("../datasets/iris.names.txt") as f:
            self.DESCR = f.readlines()

        raw_data = np.genfromtxt(
            "../datasets/iris.data.txt", dtype=str, delimiter=',')

        self.data = raw_data[:, :len(raw_data[0]) - 1].astype(np.float)
        self.target = np.array([
            0 if el == "Iris-setosa" else 2 if el == "Iris-virginica" else 1
            for el in raw_data[:, len(raw_data[0]) - 1:].flatten()
        ])
        self.randomize_data()
        self.split_data()
        self.standardize_data()
        self.discretize_data()

    def load_lenses(self):
        with open("../datasets/lenses.names.txt") as f:
            self.DESCR = f.readlines()

        raw_data = np.genfromtxt(
            "../datasets/lenses.data.txt", dtype=str)[:, 1:]

        self.data = raw_data[:, :len(raw_data[0]) - 1].astype(np.float)
        self.target = raw_data[:, len(raw_data[0]) - 1:].astype(
            np.float).flatten()

        self.randomize_data()
        self.split_data()

    def load_voting(self):
        # I've hit some stupid wall with this one too. I just don't know
        with open("../datasets/house-votes-84.names.txt") as f:
            self.DESCR = f.readlines()

        raw_data = np.genfromtxt(
            "../datasets/house-votes-84.data.txt", dtype=str, delimiter=',')

        self.data = raw_data[:, 1:]
        self.target = np.array([
            0 if el == "republican" else 1 for el in raw_data[:, :1].flatten()
        ])

        self.randomize_data()
        self.split_data()

    def load_chess(self):
        # This one don't work with the things
        with open("../datasets/krkopt.info.txt") as f:
            self.DESCR = f.readlines()

        raw_data = np.genfromtxt(
            "../datasets/krkopt.data.txt", dtype=str, delimiter=',')

        self.data = raw_data[:, :len(raw_data[0]) - 1]
        self.target = raw_data[:, len(raw_data[0]) - 1:].flatten()

        # Maybe we're fine leaving things the way they are
        self.randomize_data()
        self.split_data()
