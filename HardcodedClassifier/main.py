import argparse
from sklearn.naive_bayes import GaussianNB
from common.ClassifierTesterIris import ClassifierTesterIris
from HardCodedClassifier import HardCodedClassifier


def test_gaussian_nb(num_tests=200, verbose=False):
    print("----------------------------")
    print("    TESTING GAUSSIAN NB     ")
    print("----------------------------")
    classifier = GaussianNB()
    tester = ClassifierTesterIris(classifier)
    tester.test(num_tests, verbose)
    print('\n\n')


def test_hardcoded(num_tests=200, verbose=False):
    print("----------------------------")
    print("     TESTING HARDCODED      ")
    print("----------------------------")
    classifier = HardCodedClassifier()
    tester = ClassifierTesterIris(classifier)
    tester.test(num_tests, verbose)
    print('\n\n')


def processArgs():
    parser = argparse.ArgumentParser(description="Run train/test iris model")
    parser.add_argument(
        '--verbose', help='show verbose accuracy output', action="store_true")
    parser.add_argument(
        '--numTests',
        type=int,
        help="specify the number times to retrain and test the model")
    parser.add_argument(
        '--gnb', help='run gaussian naive bayes tests', action="store_true")
    parser.add_argument(
        '--hcc', help='run hard coded classifier tests', action="store_true")
    return parser.parse_args()


def main():
    args = processArgs()
    if args.gnb:
        test_gaussian_nb(args.numTests, args.verbose)
    if args.hcc:
        test_hardcoded(args.numTests, args.verbose)


if __name__ == '__main__':
    main()
