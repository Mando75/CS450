import numpy as np


class kdNode:
    def __init__(self, point, left, right):
        self.point = point
        self.left = left
        self.right = right


class kdTree:
    def __init__(self, points):
        self.tree = self.build_tree(points)

    def build_tree(self, points, depth=0):
        n = len(points)
        if n <= 0:
            return None

        # Pick axis to split
        axis = np.mod(depth, np.shape(points)[1])

        # Find the median point
        indices = np.argsort(points[:, axis])
        points = points[indices, :]
        median = int(np.ceil(float(np.shape(points)[0] - 1) / 2))

        # Separate the remaining points into the sides of the tree
        left_points = points[:median, :]
        right_points = points[:median + 1:, :]

        # Create new branch of nodes recursively
        return kdNode(points[median, :], self.build_tree(
            left_points, depth + 1), self.build_tree(right_points, depth + 1))

    """
        Find the nearest value to a given point
    """

    def return_nearest(self, point, depth=0, tree=None):
        if tree is None:
            tree = self.tree

        if tree.left is None:
            # Leaf
            distance = np.sum((tree.point - point)**2)
            return tree.point, distance, 0
        else:
            # Pick axis to split
            axis = np.mod(depth, np.shape(point)[0])

            # traverse the tree
            if point[axis] < tree.point[axis]:
                best_guess, distance, height = self.return_nearest(
                    point, depth + 1, tree.left)
            else:
                best_guess, distance, height = self.return_nearest(
                    point, depth + 1, tree.right)

        if height <= 2:
            # Check sibling
            if point[axis] < tree.point[axis]:
                best_guess_2, distance_2, height_2 = self.return_nearest(
                    point, depth + 1, tree.right)
            else:
                best_guess_2, distance_2, height_2 = self.return_nearest(
                    point, depth + 1, tree.left)

        distance_3 = np.sum((tree.point - point)**2)
        if distance_3 < distance_2:
            distance_2 = distance_3
            best_guess = tree.point
        if distance_2 < distance:
            distance = distance_2
            best_guess = best_guess_2

        return best_guess, distance, height + 1
