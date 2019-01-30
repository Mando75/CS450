from DecisionTree.DTreeClassifier import DTreeClassifier
from common.ClassifierTesterIris import ClassifierTesterIris


def main():
    classifier = DTreeClassifier()
    tester = ClassifierTesterIris(classifier)
    tester.test(100, False, discretize=True)


if __name__ == '__main__':
    main()
