import numpy as np
import heapq


class kdTree:
    def __init__(self, points):
        """
        Initialize a new k-dimensional tree.
        The k will be automatically determined by the shape of the points
        :param points: Expects a numpy array of tuples of the form ([coordinates], index)
        """
        self.tree = self.build_tree(points)
        self.kdSetHeap = kdTree.kdPrioritySet()

    def build_tree(self, points, depth=0):
        """
        Recursively creates a new tree from a list of points
        :param points: Expects a numpy array of tuples of the form ([coordinates], index)
        :param depth: Indicates the tree level the function is recursively being called on
        :return: A the root kdNode of a k-dimensional tree
        """
        try:
            k = len(points[0][0])
        except IndexError as e:
            return None

        # Determine which axis to split on.
        # The axis will change on every level we go down
        axis = depth % k

        # sort the points by the coordinate on the chosen axis
        sorted_points = sorted(points, key=lambda point: point[0][axis])
        # find the middle item of the sorted points (will be used to split down the tree)
        median = len(sorted_points) // 2

        # Create a new kdNode with the middle point set as it's point
        # and recursively building the left and right branches with the
        # split sorted points
        return kdTree.kdNode(
            sorted_points[median],
            self.build_tree(sorted_points[:median], depth + 1),
            self.build_tree(sorted_points[median + 1:], depth + 1), depth)

    def return_nearest(self, root, point, depth=0):
        """
        Recursively find the nearest neighbor to a given point
        :param root: The root node of the (sub)tree to traverse
        :param point: The point we are finding the distance to
        :param depth: The current depth of the tree
        :return: Tuple containing: 
                    Neighbor point: ([coordinates], original_index)
                    Distance from target point: number
                    height of the neighbor point: integer
        """

        # If we are at the bottom of the tree, return the current (root) node
        # with its distance from the target point
        if root.right is None:
            distance = np.sum((root.point[0] - point)**2)
            return root.point, distance, 0
        elif root.left is None:
            distance = np.sum((root.point[0] - point)**2)
            return root.point, distance, 0
        else:

            # determine the axis to pivot on based on current
            # tree depth and size of the coordinates (number of axis)
            axis = np.mod(depth, np.shape(point)[0])

            # Determine whether we need to recursively traverse
            # left or right down the tree
            if point[axis] < root.point[0][axis]:
                best, distance, height = self.return_nearest(
                    root.left, point, depth + 1)
            else:
                best, distance, height = self.return_nearest(
                    root.right, point, depth + 1)

        if height <= 2:
            # Check the sibling nodes for possible better options
            if point[axis] < root.point[0][axis]:
                best2, distance2, height2 = self.return_nearest(
                    root.right, point, depth + 1)
                # add this possible better option to the k-neighbors.
                # if it is not a good option, it will be put in the back
                # of the PrioritySet. This is called to prevent only one
                # neighbor ever being added to the set of neighbors
                self.add_neighbor(best2, distance2, height2)
            else:
                best2, distance2, height2 = self.return_nearest(
                    root.left, point, depth + 1)
                self.add_neighbor(best2, distance2, height2)

            # Add current node to PrioritySet of neighbors
            distance3 = np.sum((root.point[0] - point)**2)
            self.add_neighbor(root.point, distance3, height + 1)

            # Compare current node to sibling nodes,
            # if they are better, set them as best option
            if distance3 < distance2:
                distance2 = distance3
                best2 = root.point
            if distance2 < distance:
                distance = distance2
                best = best2

        # add best option to the to set of neighbors.
        # Repeats will not be duplicated
        self.add_neighbor(best, distance, height + 1)

        # This will recursively return a tuple with the
        # closest node
        return best, distance, height + 1

    def return_nearest_k(self, point, k):
        """
        Return the nearest k neighbors
        :param point: the target point
        :param k: the k nearest neighbors to look for
        :return: array of k nearest kdNodes
        """

        # Reset the PrioritySet
        self.kdSetHeap.clear()
        # This method will rebuild the priority set as it traverses
        # the tree
        self.return_nearest(self.tree, point)
        nearest = []
        for i in range(0, k):
            if i < len(self.kdSetHeap.heap):
                # check that we aren't asking for more k
                # than there are neighbors. (e.g. k could be 30, but there
                # are only 15 items in the tree)
                nearest.append(self.kdSetHeap.pop())
        return nearest

    def add_neighbor(self, node, dist, height):
        """
        Creates and pushes a neighbor onto the PrioritySet
        :param node: the kdNode to add as a neighbor
        :param dist: the distance from the neighbor to the target point
        :param height: the height of the neighbor in the tree
        :return: 
        """
        self.kdSetHeap.add(kdTree.kdNeighbor(node, dist, height))

    """INNER CLASSES"""

    class kdNode:
        """
        Represent a node in the k-dimensional tree. Each node has a
        depth(int), 
        point ([coordinates(number)], index(int))
        left node (kdNode)
        right node (kdNode)
        """

        def __init__(self, point, left, right, depth):
            """
            Initialize a new kdNode
            :param point: Expects a tuple in the form ([coordinates], original_index)
            :param left: The kdNode to the left of self in the tree
            :param right: The kdNode to the right of self in the tree
            :param depth: The depth of the kdNode in the tree
            """

            self.depth = depth
            self.point = point
            self.left = left
            self.right = right

        def __repr__(self):
            """
            :return: A string listing the attributes of the kdNode
            """
            tab = "\t" * self.depth
            return f"depth: {self.depth} point{self.point}\n{tab} left: {self.left} \n{tab} right: {self.right}"

    class kdNeighbor:
        """
        A class for representing a k-dimensional neighbor of a sought point.
        Contains a 
        node (kdNode)
        distance (float)
        height (int)
        """

        def __init__(self, node, distance, height):
            """
            Initialize new kdNeighbor
            :param node: the neighbor kdNode from the tree
            :param distance: the distance the neighbor node is from the target node
            :param height: the height of the neighbor in the tree
            """
            self.node = node
            self.distance = distance
            self.height = height

        """ Comparisons for heap insertion. Use distance """

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
        """
        A custom PrioritySet which maintains a priority queue, but
        with the set property of unique elements. Uniqueness is
        based off of the kdNode coordinates. 
        """

        def __init__(self):
            """
            Initialize a new heap and set
            """
            self.heap = []
            self.set = set()

        def add(self, neighbor):
            """
            Add a new kdNode to the PrioritySet.
            :param neighbor: 
            :return: 
            """
            if tuple(neighbor.node[0]) not in self.set:
                # if a tuplized form of the kdNode's coordinates
                # are not in the heap, add them
                heapq.heappush(self.heap, neighbor)
                self.set.add(tuple(neighbor.node[0]))

        def pop(self):
            """
            Pop the prioritized item off of the PrioritySet
            :return: kdNeighbor
            """
            neighbor = heapq.heappop(self.heap)
            self.set.remove(tuple(neighbor.node[0]))
            return neighbor

        def clear(self):
            """
            Empty the PrioritySet
            :return: 
            """
            self.heap = []
            self.set = set()
