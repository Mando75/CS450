from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


def train(training_data, training_target):
    classifier = GaussianNB()
    classifier.fit(training_data, training_target)
    return classifier


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


def compare(predicted_targets, actual_targets):
    diff = get_diff(predicted_targets, actual_targets)
    accuracy = (len(diff) / actual_targets.size) * 100
    print("DIFF: ", diff)
    print("ACCURACY: ", accuracy)


def main():
    iris = datasets.load_iris()
    for i in range(0, 100):
        training_data, testing_data, training_target, testing_target = train_test_split(
            iris.data, iris.target, shuffle=True)

        model = train(training_data, training_target)

        predicted_targets = model.predict(testing_data)
        compare(predicted_targets, testing_target)


if __name__ == '__main__':
    main()
