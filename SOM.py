"""
    This is an implementation of Kohonen's self organising map or SOM for short.
    This algorithm disperses centroids to be equ-probably distributed
    between data-instances. All that while maintaining a topology for the centroids
    that effect the process and result of the algorithm and in the end
    usually create high-quality spase filling curves.
"""

from argparse import ArgumentError
from cmath import inf
from matplotlib.collections import LineCollection
import numpy as np
import matplotlib.pyplot as plt

# product of an iterable container
def MUL(con):
    size = 1
    for i in con:
        size *= i
    return size


class centroid:

    def __init__(self, id: int, start: np.ndarray):

        if id < 0:
            raise ArgumentError('positive dimension and non-negative id required')

        self.id = id
        self.loc = start
        self.neighbours = set()

    def add_neighbour(self, id: int):
        self.neighbours.add(id)

    def distance(self, vector: np.ndarray):
        # euclidean distance
        return np.linalg.norm(vector - self.loc)

    def advance(self, factor, point):

        old_loc = self.loc.copy()  # before update
        self.loc += factor * (point - self.loc)
        return np.linalg.norm(old_loc - self.loc)  # euclidean distance / how much did 'self' move

    def get_loc(self) -> np.ndarray:
        return self.loc

    def __repr__(self) -> str:

        return "id: " + str(self.id) + ", loc: " + str(self.loc) + ", neighbours" + str(self.neighbours) + "\n"


class SOM:

    def __init__(self, shape: tuple, learning_rate: float, dim: int, start: np.ndarray):

        if len(shape) == 0:
            raise ArgumentError('no topology, argument `shape` is empty')
        for d in shape:
            if d <= 0:
                raise ArgumentError('positive dimension')
        if dim <= 0:
            raise ArgumentError('positive dimension')
        if len(start) != dim:
            raise ArgumentError('start is the initial location of the centroids must match in length `dim`')

        self.learning_rate = learning_rate
        self.shape = shape
        self.start = start
        self.cen = dict() # id -> centroid

        # constructing the centroid data structure
        # self.shape dictates topology
        self.clear(dim, start)

    # resetting map
    def clear(self, dim: int, start: np.ndarray):

        size = MUL(self.shape)

        for i in range(size):

            self.cen[i] = centroid(i, start.copy())

            cor = [0] * len(self.shape)    # centroid topology-cordinates
            d = len(self.shape) -1
            id = i
            while d >= 0:

                size //= self.shape[d]
                val = id // size
                cor[d] = val
                id -= (size*val)
                d -= 1
            size = MUL(self.shape)

            def cor_to_id(cor):
                id = 0
                size = 1
                for d in range(len(self.shape)):
                    id += size*cor[d]
                    size *= self.shape[d]
                return id

            for j in range(len(cor)):   # connecting to neighbours

                if cor[j] +1 < self.shape[j]:   # asserting in-bounds
                    cor[j] +=1
                    self.cen[i].add_neighbour(cor_to_id(cor))
                    cor[j] -=1
                if cor[j] -1 >= 0:
                    cor[j] -=1
                    self.cen[i].add_neighbour(cor_to_id(cor))
                    cor[j] +=1


    def __repr__(self) -> str:

        ans = ""
        for c in self.cen.values():
            ans += c.__repr__()
        return ans


    # get centroid by id, not obvious if the shape isn't 1 dimensional
    def get_centroid(self, id) -> centroid:
        return self.cen[id]

    def get_neighbours(self, id: int, depth: int):

        neighbourhood = set()
        neighbourhood.add(id)

        for i in range(depth):
            tmp = set()
            for j in neighbourhood:
                tmp = tmp.union(self.get_centroid(j).neighbours.copy())

        return neighbourhood

    # finding closest centroid to a data instance
    def closest(self, wins: np.ndarray, vector: np.ndarray, itr: int):

        min_score = inf
        ind = 0
        for c in self.cen.values():

            score = c.distance(vector) - (wins[c.id] - itr / len(wins))  # conscious formula

            if score < min_score:
                min_score = score
                ind = c.id

        wins[ind] += 1
        return ind

    def train(self, data: np.ndarray, halt: float, depth: int):

        # halt is a relatively small value
        # depth states how far of a neighbour is effected

        if len(data.shape) < 2:
            raise ArgumentError('expecting some instances of whatever vector-spase')

        # executing actuall algorithm

        wins = np.zeros(len(self.cen))  # taking into account past winners

        itr = 0
        diff = inf                      # a measure of change that happened in the iteration
        momentum = 1                    # factor for diminishing the learning rate each epoc

        while diff > halt:

            for instance in data:

                winner = self.get_centroid(self.closest(wins, instance, itr))

                # advance closest centroid towards the data instance
                diff = winner.advance(self.learning_rate / momentum, instance)
                # and its neighbours too but less so
                for id in self.get_neighbours(winner.id, depth):
                    self.get_centroid(id).advance(self.learning_rate / 3*momentum, instance)

                itr += 1

            momentum *= 2

        return self.cen.copy(), itr

    # this method should display the centroids together with their topology (lines between neighbours)
    def display(self, ax=None):

        if ax is None:
            ax = plt.gca()

        dots = np.array([c.get_loc() for c in self.cen.flatten()])
        ax.scatter(dots[:, 0], dots[:, 1])
        dots = dots.reshape(list(self.shape) + list(self.cen.flatten()[0].get_loc().shape))
        ax.add_collection(LineCollection(dots))
        if dots.shape[-1] == 2:
            ax.add_collection(LineCollection(dots.transpose(1, 0, 2), cmap = plt.cm.brg))
        plt.show()

    # TODO: display, tests, prephormance avaluation, jupiter notebook
    # and of course monkey hands
