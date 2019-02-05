import pandas as pd
import numpy as np
from functools import reduce


class dNode:
    def __init__(self, label, leaf=False):
        self.label, self.leaf = label, leaf
        self.children = {}

    def append_child(self, child):
        if self.leaf:
            self.children = child
        else:
            self.children[child[0]] = child[1]
        return self.children[child[0]]

    def __repr__(self):
        if self.leaf:
            return self.label
        else:
            return f"{self.label} --> {self.children}"

class dTree:
    def __init__(self, data, labels, default_class=None):
        self.root = self.makeTree(data, "class")
        self.labels, self.default_class = labels, default_class

    def __repr__(self):
        return str(self.root)

    def makeTree(self, data, class_name):
        size = data[class_name].size
        split_options = data[class_name].unique()
        # Base Case 1:
        # We don't have any more ways to split the data
        if len(split_options) == 1:
            return dNode(split_options[0], leaf=True)
        # Base Case 2:
        if size == 1:
            return data[class_name].value_counts().index[0]

        # Recursive tree construction
        entropies = []
        for column in data.columns:
            if column is not class_name:
                average_entropy = 0
                for option in data[column].unique():
                    q_set = data[data[column] == option][class_name]
                    entropy = self.get_entropy(q_set)
                    average_entropy += q_set.size / size * entropy
                entropies.append([average_entropy, column])
            if len(entropies) is 0:
                return None
            else:
                entropies.sort()
                best_split = entropies[0][1]

            node = dNode(best_split)
            for option in data[best_split].unique():
                split_data = data[data[best_split] == option].drop(
                    best_split, axis=1)
                node.append_child(
                    [option, self.makeTree(split_data, class_name)])
            return node

    def get_entropy(self, q_set):
        probabilities = self.get_series_probabilities(q_set)
        return reduce(lambda acc, p: acc - (p * np.log2(p)) if p != 0 else 0,
                      probabilities)

    def get_series_probabilities(self, q_set):
        size = q_set.size
        return list(map(lambda c: c / size, q_set.value_counts()))


class dTreeClassifier:
    def __init__(self):
        self.tree = None
        self.labels = []
        self.default_class = None

    def fit(self, data, target):
        combined = pd.DataFrame(data)
        self.labels = combined.columns
        combined["class"] = pd.Series(target)
        self.default_class = combined["class"].value_counts().index[0]
        self.tree = dTree(combined, self.labels, self.default_class)
        return self.tree

    def predict(self, data):
        predictions = []
        for index, row in data.iterrows():
            predictions.append(self.find_one(row))
        return predictions

    def find_one(self, row):
        return self.traverse_tree(self.tree.root, row)

    def traverse_tree(self, node, data):
        if node is None:
            return self.default_class
        if node.leaf:
            return node.label
        key = int(data[node.label])
        if key in node.children:
            return self.traverse_tree(node.children[key], data)
        return self.default_class

