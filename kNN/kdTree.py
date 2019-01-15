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
        if root is None:
            return None

        axis = depth % len(point)

        if point[axis] < root.point[axis]:
            next_branch = root.left
            opposite_branch = root.right
        else:
            next_branch = root.right
            opposite_branch = root.left

        best = self.closer_distance(
            point, self.return_nearest(next_branch, point, depth + 1),
            root.point)

        if self.distance(point, best) > abs(point[axis] - root.point[axis]):
            best = self.closer_distance(
                point, self.return_nearest(opposite_branch, point, depth + 1),
                best)

        return best

    def closer_distance(self, pivot, p1, p2):
        if p1 is None:
            return p2

        if p2 is None:
            return p1

        d1 = self.distance(pivot, p1)
        d2 = self.distance(pivot, p2)

        if d1 < d2:
            return p1
        else:
            return p2

    @staticmethod
    def distance(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        dx = x1 - x2
        dy = y1 - y2

        return np.sqrt(dx * dx + dy * dy)
