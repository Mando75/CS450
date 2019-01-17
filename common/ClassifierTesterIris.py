from sklearn import datasets
from sklearn.model_selection import train_test_split


class ClassifierTesterIris:
    def __init__(self, classifier):
        self.classifier = classifier
        self.accuracies = []
        self.avg_accuracy = 0
        self.num_tests = 0

    def train(self, training_data, training_target):
        self.classifier.fit(training_data, training_target)
        return self.classifier

    def compare(self, predicted_targets, actual_targets, show):
        diff = self.get_diff(predicted_targets, actual_targets)
        size = actual_targets.size
        accuracy = round(((size - len(diff)) / size) * 100, 2)
        self.accuracies.append(accuracy)
        if show:
            print("DIFF: ", diff)
            print("ACCURACY: ", accuracy)

    def test(self, num_tests=100, show=False, average=False):
        self.num_tests = num_tests
        iris = datasets.load_iris()
        for i in range(0, num_tests):
            training_data, testing_data, training_targets, testing_targets = train_test_split(
                iris.data, iris.target, shuffle=True)

            model = self.train(training_data, training_targets)
            predicted_targets = model.predict(testing_data, average)
            if show:
                print("\nTest ", i + 1)
            self.compare(predicted_targets, testing_targets, show)
        self.summary()

    def summary(self):
        print("-----------------------")
        print("        SUMMARY        ")
        print("-----------------------")
        print("Number of Tests: ", self.num_tests)
        self.avg_accuracy = round(
            sum(self.accuracies) / len(self.accuracies), 2)
        print("Total Average Accuracy: ", self.avg_accuracy)
        print("Max Accuracy", max(self.accuracies))
        print("Min Accuracy", min(self.accuracies))

    @staticmethod
    def get_diff(predicted_targets, actual_targets):
        diff = []
        for index, predicted in enumerate(predicted_targets):
            if predicted != actual_targets[index]:
                diff.append({
                    'index': index,
                    'predicted': predicted,
                    'actual': actual_targets[index]
                })
        return diff
