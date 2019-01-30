from DecisionTree.dTree import dTree


class DTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, training_data, training_targets):
        self.tree = dTree(training_data, training_targets)

    def predict(self, testing_data):
        if self.tree is None:
            return "ERROR"

        predictions = []
        for i in testing_data:
            predictions.append(self.tree.search(i))
        return predictions
