import numpy as np
import heapq

class kdNode:
    def __init__(self, point, left, right, depth):
        self.depth = depth
        self.point = point
        self.left = left
        self.right = right

    def __repr__(self):
        tab = "\t" * self.depth
        return f"depth: {self.depth} point{self.point}\n{tab} left: {self.left} \n{tab} right: {self.right}"

class kdNeighbor:
    def __init__(self, node, distance, height):
        self.node = node
        self.distance = distance
        self.height = height

    def __lt__(self, other):
        return self.distance < other.distance

    def __le__(self, other):
        return self.distance <= other.distance

    def __gt__(self, other):
        return self.distance > other.distance

    def __ge__(self, other):
        return self.distance >= other.distance

    def __eq__(self, other):
        return self.distance == other.distance

class kdPrioritySet(object):
    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, neighbor):
        # Use a tuple so we can track the unique sets
        if tuple(neighbor.node[0]) not in self.set:
            heapq.heappush(self.heap, neighbor)
            self.set.add(tuple(neighbor.node[0]))

    def pop(self):
        neighbor = heapq.heappop(self.heap)
        self.set.remove(tuple(neighbor.node[0]))
        return neighbor

    def clear(self):
        self.heap = []
        self.set = set()



class kdTree:
    def __init__(self, points):
        self.tree = self.build_tree(points)
        self.heap = kdPrioritySet()

    def build_tree(self, points, depth=0):
        try:
            k = len(points[0][0])
        except IndexError as e:
            return None

        axis = depth % k

        sorted_points = sorted(points, key=lambda point: point[0][axis])
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
            distance = np.sum((root.point[0] - point)**2)
            return root.point, distance, 0
        elif root.left is None:
            distance = np.sum((root.point[0] - point)**2)
            return root.point, distance, 0
        else:

            axis = np.mod(depth, np.shape(point)[0])

            if point[axis] < root.point[0][axis]:
                best, distance, height = self.return_nearest(
                    root.left, point, depth + 1)
            else:
                best, distance, height = self.return_nearest(
                    root.right, point, depth + 1)

        if height <= 2:
            if point[axis] < root.point[0][axis]:
                best2, distance2, height2 = self.return_nearest(
                    root.right, point, depth + 1)
            else:
                best2, distance2, height2 = self.return_nearest(
                    root.left, point, depth + 1)

            distance3 = np.sum((root.point[0] - point)**2)
            if distance3 < distance2:
                distance2 = distance3
                best2 = root.point
            if distance2 < distance:
                distance = distance2
                best = best2
            else:
                neighbor = kdNeighbor(best2, distance, height + 1)
                self.heap.add(neighbor)

        neighbor = kdNeighbor(best, distance, height + 1)
        self.heap.add(neighbor)
        return best, distance, height + 1

    def return_nearest_k(self, point, k):
        self.return_nearest(self.tree, point)
        nearest = []
        for i in range(0, k):
            nearest.append(self.heap.pop())
        return nearest
