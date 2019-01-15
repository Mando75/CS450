from common.ClassifierTesterIris import ClassifierTesterIris
from kNN.kNNClassifier import kNNClassifier
from kNN.kdTree import kdTree
import numpy as np
import argparse


def test_knn(num_tests=200, verbose=False):
    print("----------------------------")
    print("        TESTING kNN         ")
    print("----------------------------")
    classifier = kNNClassifier()
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
        default=200,
        help="specify the number times to retrain and test the model")
    return parser.parse_args()


def main():
    points = [[2, 3, 5], [1, 5, 4], [5, 9, 6], [1, 4, 7], [8, 8, 1], [4, 7, 2]]
    kd = kdTree(points)
    print(kd.return_nearest(kd.tree, [4, 2, 1]))


if __name__ == '__main__':
    main()
