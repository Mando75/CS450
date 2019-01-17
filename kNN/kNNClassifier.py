import numpy as np
from kNN.kdTree import kdTree


class kNNClassifier:
    """
        Defines a k-Nearest Neighbor Classifier for
        predicting classifications. To work the classifier, 
        provide training data with actual targets to the fit method.
        You can then run predictions using the predict method.
    """

    def __init__(self, k=1, use_tree=True):
        """
        Initialize the class setting the k neighbors to be searched for
        :param k: 
        :param use_tree: Indicate whether to use the k-dimensional tree algorithm to predict targets, or the
        Euclidean distance algorithm. Default is True.
        """
        self.k = k
        self.data = np.array([])
        self.targets = np.array([])
        self.kd_tree = None
        self.use_tree = use_tree

    def fit(self, training_data, training_targets):
        """
        Takes training data and targets as parameters. 
        Will use the training data to construct a k-dimensional 
        tree for future predictions
        :param training_data: A numpy 2-d array of training data to create a model with
        :param training_targets: A numpy 2-d array of training targets to create a model with
        :return: None
        """

        self.data = [(point, i)
                     for i, point in enumerate(np.array(training_data))]
        self.targets = np.array(training_targets)
        if self.use_tree:
            self.kd_tree = kdTree(self.data)

    def predict(self, testing_data, average=False):
        """
        Attempts to predict the targets for the provided testing data using either
        a Euclidean distance algorithm, or a k-dimensional tree
        :param testing_data: A numpy 2-d array of testing data to predict
        :param average: Indicate whether the algorithm should return the mean of the k neighbor targets 
        or the most common k neighbor target. Default is false (meaning most common k neighbor target)
        :return: Array of predictions with corresponding indexes to the provided test data
        """
        if self.use_tree:
            return self.predict_tree(testing_data, average)
        else:
            return self.predict_euclid(testing_data)

    def predict_tree(self, testing_data, average=False):
        """
        Attempts to predict the target for the provided testing data using
        a k-dimensional tree
        :param testing_data: A numpy 2-d array of testing data to predict 
        :param average: Indicate whether or the algorithm should return the mean of the k neighbor targets
        or the most common neighbor target. Default is false (meaning most common k neighbor target)
        :return: Array of predictions with corresponding indexes to the provided test data
        """
        predictions = []
        for point in testing_data:
            # Loop over each point and find it's k-nearest neighbors
            k_nearest = self.kd_tree.return_nearest_k(point, self.k)
            targets = [self.targets[n.node[1]] for n in k_nearest]
            if average:
                predictions.append(round(np.average(targets)))
            else:
                unique, counts = np.unique(targets, return_counts=True)
                max_index = np.argmax(counts)
                predictions.append(unique[max_index])
        return predictions

    def predict_euclid(self, testing_data):
        """
        Attempts to predict the target for the provided test data using the
        Euclidian distance algorithm
        :param testing_data: A 2-d numpy array of testing data to predict
        :return: Array of predictions with corresponding indexes to the provided test data
        """

        # get number of inputs from numpy.shape
        n_inputs = np.shape(testing_data)[0]
        # create array of "closest" values the same length as the inputs
        closest = np.zeros(n_inputs)

        for n in range(n_inputs):
            # Use numpy array sum to perform the Euclidean distance calculation
            # on each input
            distances = np.sum((self.data - testing_data[n, :])**2, axis=1)

            # get the sorted indices of the distance data
            indices = np.argsort(distances, axis=0)

            # retrieve the classes from the k nearest neighbors
            classes = np.unique(self.targets[indices[:self.k]])

            if len(classes) == 1:
                # if there was only one class, that is our prediction
                closest[n] = np.unique(classes)
            else:
                # Otherwise return the most common class
                counts = np.zeros(max(classes) + 1)

                counts = list(
                    map(lambda index: counts[self.targets[index]] + 1,
                        indices))

                closest[n] = np.max(counts)

        return closest
