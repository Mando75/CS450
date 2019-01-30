import math

import numpy as np
from sklearn import datasets
from sklearn import tree as dt
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, data, targets, feature_key=None, split_value=None):
        # Remove used column
        if feature_key is not None:
            np.delete(data, feature_key, axis=1)

        self.data = np.vstack(data)
        self.targets = targets

        self.feature_key = feature_key  # What column value is on
        self.split_value = split_value  # What value was split on

        self.leaf = None  # The answer

        # If dataSet is out of features or if there is only one answer left
        if not self.data.shape[0] == 2:
            if not len(set(self.targets)) == 1:
                self.childrenList = split_tree(self.data, self.targets)
                return

        self.leaf = max(self.targets)


# Find the entropy of a 1-dimensional list
def find_entropy(data):
    denominator = len(data)
    entropy_list = []

    # For value in set of values
    for value in set(data):
        count = data.count(value)

        # -PA * log2PA
        entropy_list.append(
            (-count / denominator) * math.log2(count / denominator))
    return sum(entropy_list)


# Find the best column to split on
def find_best_column(data, targets):
    answer_map = {}
    for col in range(len(data[0])):
        target_key = {}
        for index in range(len(data)):
            if data[:, col][index] not in target_key:
                target_key[data[:, col][index]] = []
            target_key[data[:, col][index]].append(targets[index])

        entropy_list = []
        size = 0
        for value in target_key:
            entropy_list.append(
                find_entropy(target_key[value]) * len(target_key[value]))
            size += len(target_key[value])

        answer_map[sum(entropy_list) / size] = col

    return answer_map[min(answer_map.keys())]


# Take in data and return a list of Nodes
def split_tree(data, targets):
    col = find_best_column(data, targets)

    feature_key = {}
    for index in range(len(data)):
        if data[index][col] not in feature_key:
            feature_key[data[index][col]] = {'data': [], 'targets': []}
        feature_key[data[index][col]]['data'].append(data[index])
        feature_key[data[index][col]]['targets'].append(targets[index])

    # Create a list of Nodes
    children_list = []
    for key in feature_key.keys():
        children_list.append(
            Node(feature_key[key]['data'], feature_key[key]['targets'], col,
                 key))

    # # Return list of Nodes to its parent Node
    return children_list


# Find answer using the tree
def find(root, feature_list):
    it = root
    while not it.leaf:
        looping = True
        for node in it.childrenList:
            if feature_list[node.feature_key] == node.split_value:
                it = node
                return it.leaf
        if looping:
            break
    return it.leaf


def answer_set(root, data):
    answer_list = []
    for i in data:
        answer_list.append(find(root, i))
    return answer_list


# Retrieve iris data
iris = datasets.load_iris()
dataSet = iris.data
targetSet = iris.target

data_train, data_test, targets_train, targets_test = train_test_split(
    dataSet, targetSet, test_size=0.3)

tree = Node(data_train, targets_train)
score = 0
for index, value in enumerate(answer_set(tree, data_test)):
    if value == targets_test[index]:
        score += 1
print("My decision tree: ", round(score / len(data_test) * 100, 2), "%")

score = 0
clf = dt.DecisionTreeClassifier()
clf = clf.fit(data_train, targets_train)
for index, value in enumerate(clf.predict(data_test)):
    if value == targets_test[index]:
        score += 1
print("skLearn decision tree: ", round(score / len(data_test) * 100, 2), "%")
