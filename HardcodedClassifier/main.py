from sklearn.naive_bayes import GaussianNB
from ClassifierTesterIris import ClassifierTesterIris
from HardCodedClassifier import HardCodedClassifier


def test_gaussian_nb():
    print("----------------------------")
    print("    TESTING GAUSSIAN NB     ")
    print("----------------------------")
    classifier = GaussianNB()
    tester = ClassifierTesterIris(classifier)
    tester.test(200)
    print('\n\n')


def test_hardcoded():
    print("----------------------------")
    print("     TESTING HARDCODED      ")
    print("----------------------------")
    classifier = HardCodedClassifier()
    tester = ClassifierTesterIris(classifier)
    tester.test(200)
    print('\n\n')


def main():
    test_gaussian_nb()
    test_hardcoded()


if __name__ == '__main__':
    main()
