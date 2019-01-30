import numpy as np


class DNode(object):
    def __init__(self, data, targets, columns, root=False):
        self.children = []
        self.feature = self.value = None
        self.data, self.targets, self.columns = data, targets, columns
        self.entropy = self.get_entropy()

    def get_entropy(self):
        total = 0
        for i in np.unique(self.targets):
            prob = self.targets.tolist().count(i) / len(self.targets)
            total += -1 * prob * np.log2(prob)
        return total

    def make_children(self):
        uniq_targets = np.unique(self.targets)
        """ 
        Base Case 1: We have split the tree so there is
        only one target value left in our branch. Set it
        as our node value. Stop building any more branches
        """
        if len(uniq_targets) == 1:
            self.value = uniq_targets[0]
            return
        """
        Base Case 2: We've run out of columns to split on.
        Set value to most common target. 
        """
        if len(self.columns) == 0:
            frequencies = []
            for t in uniq_targets:
                frequencies.append(self.targets.tolist().count(t))
            self.value = uniq_targets[frequencies.index(max(frequencies))]
            return
        """Split and build the tree"""
        row_count = len(self.data[:, 0])
        info_gain = []
        category_nodes = []
        for feat in self.columns:
            gain = self.entropy
            temp_nodes = []
            for vals in np.unique(self.data[:, feat]):
                prob = self.data[:, feat].tolist().count(vals) / row_count

                temp_data = np.append(self.data,
                                      np.reshape(self.targets, (-1, 1)), 1).T
                temp_data = temp_data[:, temp_data[feat] == vals].T
                temp_columns = self.columns[:]
                temp_columns.remove(feat)

                temp_node = DNode(
                    temp_data[:, :len(temp_data[0]) - 1],
                    temp_data[:, len(temp_data[0]) - 1:].astype(np.float),
                    temp_columns)

                gain -= prob * temp_node.entropy
                temp_nodes.append(temp_node)
            category_nodes.append(temp_nodes)
            info_gain.append(gain)

        i_next_feat = info_gain.index(max(info_gain))
        self.feature = self.columns[i_next_feat]
        self.children = category_nodes[i_next_feat]

        for node in self.children:
            node.make_children()
        return


class DTree(object):
    def __init__(self, data):
        self.data = data
        self.classes = data.target_count
        self.columns = list(range(0, len(data.data[0])))
        self.tree = DNode(
            self.data.training_data,
            self.data.training_targets,
            self.columns,
            root=True)

    def make_tree(self):
        self.tree.make_children()

    def predict_numeric(self):
        predicted_targets = []
        for row in self.data.test_data:
            predicted_targets.append(
                self.evaluate_numeric_node(row, self.tree))

        return predicted_targets

    def evaluate_numeric_node(self, row, node):
        if node.value:
            return node.value

        else:
            upper_bound = 0
            for i in reversed(self.data.rules[node.feature]):
                if row[node.feature] > 1:
                    break
                elif upper_bound == len(node.children) - 1:
                    break
                upper_bound += 1
            return self.evaluate_numeric_node(
                row, node.children[len(node.children) - (1 + upper_bound)])

    def print_tree(self):
        print(f"root split on feature {self.tree.feature}")
        for node in self.tree.children:
            self.node_to_string(node)

    def node_to_string(self, node):
        print("    ", end='')
        if node.value is not None:
            print(f"|__{node.value}")
        else:
            print(f"Node split on feature {node.feature}")
            for node in node.children:
                self.node_to_string(node)

    def predict_nominal(self):
        predicted_targets = []
        for row in self.data.test_data:
            predicted_targets.append(
                self.evaluate_nominal_node(row, self.tree))

        return predicted_targets

    def evaluate_nominal_node(self, row, node):
        if node.value is not None:
            return node.value
        else:
            value_set = np.unique(
                self.data.training_data[:, node.feature]).tolist()
            index_of_node = value_set.index(row[node.feature])
            return self.evaluate_nominal_node(row,
                                              node.children[index_of_node])
