import numpy as np


class kdNode:
    def __init__(self, point, left, right, depth):
        self.depth = depth
        self.point = point
        self.left = left
        self.right = right

    def __repr__(self):
        tab = "\t" * self.depth
        return f"depth: {self.depth} point{self.point}\n{tab} left: {self.left} \n{tab} right: {self.right}"


class kdTree:
    def __init__(self, points):
        self.tree = self.build_tree(points)

    def build_tree(self, points, depth=0):
        try:
            k = len(points[0])
        except IndexError as e:
            return None

        axis = depth % k

        sorted_points = sorted(points, key=lambda point: point[axis])
        median = len(sorted_points) // 2

        return kdNode(sorted_points[median],
                      self.build_tree(sorted_points[:median], depth + 1),
                      self.build_tree(sorted_points[median + 1:], depth + 1),
                      depth)

    """
        Find the nearest value to a given point
    """

    def return_nearest(self, root, point, depth=0):

        if root.right is None:
            distance = np.sum((root.point - point)**2)
            return root.point, distance, 0
        elif root.left is None:
            distance = np.sum((root.point - point)**2)
            return root.point, distance, 0
        else:

            axis = np.mod(depth, np.shape(point)[0])

            if point[axis] < root.point[axis]:
                best, distance, height = self.return_nearest(
                    root.left, point, depth + 1)
            else:
                best, distance, height = self.return_nearest(
                    root.right, point, depth + 1)

        if height <= 2:
            if point[axis] < root.point[axis]:
                best2, distance2, height2 = self.return_nearest(
                    root.right, point, depth + 1)
            else:
                best2, distance2, height2 = self.return_nearest(
                    root.left, point, depth + 1)

        distance3 = np.sum((root.point - point)**2)
        if distance3 < distance2:
            distance2 = distance3
            best2 = root.point
        if distance2 < distance:
            distance = distance2
            best = best2

        return best, distance, height + 1
