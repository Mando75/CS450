class kNNClassifier:
    def fit(self, training_data, training_targets):
        return training_data, training_targets

    def predict(self, testing_data):
        return list(map(lambda data: 0, testing_data))
