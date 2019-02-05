from DecisionTree.dTree2 import dTreeClassifier
from common.ClassifierTesterIris import ClassifierTesterIris
from sklearn.model_selection import train_test_split as tts
import pandas as pd
from numpy import NaN
from common.progressBar import printProgressBar


def main():
    classifier = dTreeClassifier()
    tester = ClassifierTesterIris(classifier)
    data = load_voting_records()
    print("Starting tests")
    printProgressBar(0, 1, prefix="Progress", suffix="Complete")
    for i in range(0, 1):
        printProgressBar(i + 1, 1, prefix="Progress", suffix="Complete")
        training_data, testing_data, training_targets, testing_targets = tts(
            data["data"], data["targets"], shuffle=True, test_size=.33)
        classifier.fit(training_data, training_targets)
        predicted_targets = classifier.predict(testing_data)
        tester.compare(predicted_targets, testing_targets.values, False)
        classifier.visualize_tree()
    tester.summary()


def load_voting_records():
    columns = [
        "class", "handicapped", "water", "adoption", "physician", "salvador",
        "religious", "anti", "aid", "missile", "immigration", "synfuels",
        "education", "superfund", "crime", "duty", "export"
    ]
    voting_data = pd.read_csv(
        "../datasets/house-votes-84.data.txt", header=None)
    voting_data.columns = columns
    voting_data.replace("?", NaN, inplace=True)
    voting_data.fillna(inplace=True, method="pad")
    voting_data.replace({"y": 1, "n": 0, "?": -1}, inplace=True)
    targets = voting_data["class"]
    voting_data.drop(labels=["class"], axis=1, inplace=True)
    split = {"data": voting_data, "targets": targets}
    return split


if __name__ == '__main__':
    main()
