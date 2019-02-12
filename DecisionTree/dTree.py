import numpy as np


class dTree:
    def __init__(self, training_data, target_data):
        self.root = dTreeNode(training_data, target_data)

    def search(self, features):
        return self.root.search(features)


class dTreeNode:
    def __init__(self, data, targets, split_val=None, feature_k=None):
        if feature_k is not None:
            np.delete(data, feature_k, axis=1)

        self.dataset = np.vstack(data)
        self.targets = targets
        self.feature = feature_k
        self.split_value = split_val
        self.has_children = False

        uniq_targets = np.unique(targets)

        self.value = None
        if len(uniq_targets) == 1:
            self.value = uniq_targets[0]
            return

        if self.dataset.shape[0] == 2:
            self.value = max(targets)
            return

        self.children = self.make_tree(self.dataset, self.targets)
        self.has_children = True

    def make_tree(self, data, targets):
        col = self.get_best_column(data, targets)

        feature_k = {}
        for i in range(len(data)):
            if data[i][col] not in feature_k:
                feature_k[data[i][col]] = {'data': [], 'targets': []}
            feature_k[data[i][col]]['data'].append(data[i])
            feature_k[data[i][col]]['targets'].append(targets[i])

        children = []
        for k in feature_k.keys():
            children.append(
                dTreeNode(feature_k[k]['data'], feature_k[k]['targets'], k,
                          col))

        return children

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

    def print_tree(self):
        if self.has_children:
            for child in self.children:
                child.print_tree()
        else:
            print(self.value)

    def search(self, features, parent=None):
        if not self.has_children:
            return self.value
        else:
            # closest = None
            # closest_distance = None
            for child in self.children:
                #     if features[child.feature] == child.split_value:
                #         return child.search(features, self)
                #     elif closest is None:
                #         closest = child
                #         closest_distance = np.abs(child.split_value -
                #                                   features[child.feature])
                #     else:
                #         child_distance = np.abs(child.split_value -
                #                                 features[child.feature])
                #         if child_distance < closest_distance:
                #             closest = child
                #             closest_distance = child_distance
                # return closest.search(features, self)

                if features[child.feature] == child.split_value:
                    return child.search(features, self)

            if parent is not None:
                return np.argmax(np.bincount(parent.targets))

        return None
