from DecisionTree.dTree2 import dTreeClassifier
from common.ClassifierTesterIris import ClassifierTesterIris
from sklearn.model_selection import train_test_split as tts
from sklearn import tree
import pandas as pd
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
    classifier2 = tree.DecisionTreeClassifier()
    tester = ClassifierTesterIris(classifier)
    tester2 = ClassifierTesterIris(classifier2)
    data = load_voting_records()
    sklearn_format = get_sklearn_format()
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

        ntrd, nttd, ntrt, nttt = tts(
            sklearn_format["data"],
            sklearn_format["targets"],
            shuffle=True,
            test_size=.33)
        classifier2.fit(ntrd, ntrt)
        predicted_targets2 = classifier2.predict(nttd)
        tester2.compare(predicted_targets2, nttt, False)
        if show_viz:
            classifier.visualize_tree(i=i)

    print("My implementation")
    tester.num_tests = num_tests
    tester.summary()
    print("\nSKLearn DTree")
    tester2.num_tests = num_tests
    tester2.summary()


def load_voting_records():
    columns = [
        "class", "handicapped", "water", "adoption", "physician", "salvador",
        "religious", "anti", "aid", "missile", "immigration", "synfuels",
        "education", "superfund", "crime", "duty", "export"
    ]
    voting_data = pd.read_csv(
        "../datasets/house-votes-84.data.txt", header=None)
    voting_data.columns = columns
    voting_data.fillna(inplace=True, method="pad")
    voting_data.replace({"y": 1, "n": 0, "?": -1}, inplace=True)
    targets = voting_data["class"]
    voting_data.drop(labels=["class"], axis=1, inplace=True)
    split = {"data": voting_data, "targets": targets}
    return split


def get_sklearn_format():
    """
    For some reason the SKLearn tree doesn't like the way I formatted
    my data for my tree implementation. Instead of modifying my tree again,
    I decided to be lazy and just reload all the data in a way that sklearn
    likes. This version doesn't work in my implementation so yeah. Loading
    the same data but twice. Can't be bothered to redo a bunch of stuff
    when this is easier...
    :return:
    """
    columns = [
        "class", "handicapped", "water", "adoption", "physician", "salvador",
        "religious", "anti", "aid", "missile", "immigration", "synfuels",
        "education", "superfund", "crime", "duty", "export"
    ]
    voting_data = pd.read_csv(
        "../datasets/house-votes-84.data.txt", header=None)
    voting_data.columns = columns
    voting_data.fillna(inplace=True, method="pad")
    voting_data.replace({"y": 1, "n": 0, "?": -1}, inplace=True)
    return {
        "targets": voting_data.pop("class").values,
        "data": voting_data.values
    }


if __name__ == '__main__':
    main()
