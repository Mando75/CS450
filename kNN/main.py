from common.ClassifierTesterIris import ClassifierTesterIris
from kNN.kNNClassifier import kNNClassifier
from sklearn.neighbors import KNeighborsClassifier
import argparse


def test_knn(num_tests=200, verbose=False, use_tree=False, k=7, average=False):
    print("----------------------------")
    print("    TESTING skLearn KNN     ")
    print("----------------------------")
    classifier = KNeighborsClassifier(n_neighbors=k)
    tester = ClassifierTesterIris(classifier)
    tester.test(num_tests, verbose)
    print("----------------------------")
    print("        TESTING kNN         ")
    print("----------------------------")
    classifier = kNNClassifier(k, use_tree)
    tester = ClassifierTesterIris(classifier)
    tester.test(num_tests, verbose, average=average)
    print('\n\n')


def process_args():
    parser = argparse.ArgumentParser(description="Run train/test iris model")
    parser.add_argument(
        '--verbose', help='show verbose accuracy output', action="store_true")
    parser.add_argument(
        '--numTests',
        type=int,
        default=200,
        help="specify the number times to retrain and test the model")
    parser.add_argument(
        '--useTree',
        action="store_true",
        help="specify whether the classifier should use a k-d tree")
    parser.add_argument(
        '--k',
        type=int,
        default=7,
        help="specify the k nearest neighbors to return")
    parser.add_argument(
        '--average',
        action="store_true",
        help="the classifier will use a calculated average to classify the data "
        + "instead of the most common target in the k-nearest neighbors")
    return parser.parse_args()


def main():
    args = process_args()
    test_knn(args.numTests, args.verbose, args.useTree, args.k, args.average)


if __name__ == '__main__':
    main()
