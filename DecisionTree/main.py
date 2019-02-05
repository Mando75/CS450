from DecisionTree.dTree2 import dTreeClassifier
from common.ClassifierTesterIris import ClassifierTesterIris
from sklearn.model_selection import train_test_split as tts
import pandas as pd
from numpy import NaN
from common.progressBar import printProgressBar
import sys


def main():
    """
    Expects 2 args, the number of tests to run (integer) and whether or not to
    generate a tree visualization as a png image using graphviz (y/n)
    :return:
    """
    num_tests = int(float(sys.argv[1]))
    if num_tests is None:
        num_tests = 1
    if sys.argv[2] is 'y':
        show_viz = True
    else:
        show_viz = False
    classifier = dTreeClassifier()
    tester = ClassifierTesterIris(classifier)
    data = load_voting_records()
    print("Starting tests")
    printProgressBar(0, num_tests, prefix="Progress", suffix="Complete")
    for i in range(0, num_tests):
        printProgressBar(
            i + 1, num_tests, prefix="Progress", suffix="Complete")
        training_data, testing_data, training_targets, testing_targets = tts(
            data["data"], data["targets"], shuffle=True, test_size=.33)
        classifier.fit(training_data, training_targets)
        predicted_targets = classifier.predict(testing_data)
        tester.compare(predicted_targets, testing_targets.values, False)
        if show_viz:
            classifier.visualize_tree(i=i)
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
