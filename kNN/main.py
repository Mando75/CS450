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
    # args = processArgs()
    # test_knn(args.numTests, args.verbose)
    # test_data = np.array(
    #     [[0, 1, 2], [5, 3, 9], [9, 0, 4], [4, 2, 7], [8, 6, 2], [3, 6, 1],
    #      [6, 8, 2], [1, 6, 3], [1, 5, 9], [1, 3, 7], [6, 8, 3], [5, 7, 2],
    #      [1, 3, 2], [1, 2, 7], [7, 4, 6], [9, 3, 1]])
    #
    # tuple_list = [(point, i) for i, point in enumerate(test_data)]
    # k_tree = kdTree(tuple_list)
    # k_tree.return_nearest(k_tree.tree, [4, 5, 1])
    test_knn(1, False)


if __name__ == '__main__':
    main()
