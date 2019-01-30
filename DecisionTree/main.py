import common.DatasetLoader as dl
from DecisionTree.dTree import DTree


def main():
    dataset = dl.Dataset()
    dataset.load_iris()

    ID3 = DTree(dataset)
    ID3.make_tree()
    dataset.predicted_targets = ID3.predict_nominal()

    dataset.report_accuracy()
    ID3.print_tree()

    return 0


if __name__ == '__main__':
    main()
