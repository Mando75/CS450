import numpy as np


class dTree(object):
    def split_tree(self, data, targets):
        col = self.get_best_column(data, targets)

        feature_k = {}
        for i in range(len(data)):
            if data[i][col] not in feature_k:
                feature_k[data[i][col]] = {'data': [], 'targets': []}
            feature_k[data[i][col]]['data'].append(data[i])
            feature_k[data[i][col]]['targets'].append(targets[i])

        child_list = []
        for k in feature_k.keys():
            child_list.append(
                dTreeNode(feature_k[k]['data'], feature_k[k]['targets'], k,
                          col))

        return child_list

    def get_best_column(self, data, targets):
        ans = {}
        for col in range(len(data[0])):
            target_k = {}
            for i in range(len(data)):
                if data[:, col][i] not in target_k:
                    target_k[data[:, col][i]] = []
                target_k[data[:, col][i]].append(targets[i])

            entropies = []
            size = 0
            for val in target_k:
                entropies.append(
                    self.get_entropy(target_k[val]) * len(target_k[val]))
                size += len(target_k[val])

            ans[sum(entropies) / size] = col

        return ans[min(ans.keys())]

    def get_entropy(self, data):
        divisor = len(data)
        entropies = []

        for val in set(data):
            count = data.count(val)
            entropies.append((-count / divisor) * np.log2(count / divisor))
        return sum(entropies)


class dTreeNode(dTree):
    def __init__(self, data, targets, split_val=None, feature_k=None):
        if feature_k is not None:
            np.delete(data, feature_k, axis=1)

        self.data = np.vstack(data)
        self.targets = targets
        self.feature_k = feature_k
        self.split_val = split_val

        self.leaf = None

        if not self.data.shape[0] == 2:
            if not len(set(self.targets)) == 1:
                self.childrenList = super(dTreeNode, self).split_tree(
                    self.data, self.targets)
                return

        self.leaf = max(self.targets)
