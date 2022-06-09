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


class centroid:

    def __init__(self, id: int, dimention: int, scope: tuple):

        if id < 0 or dimention <= 0:
            raise ArgumentError('only non negative input')
        if len(scope) != 2:
            raise ArgumentError('scope should specify exactly max and min values')
        if scope[0] > scope[1]:
            raise ArgumentError('min > max, scope')

        self.id = id
        self.loc = np.random.uniform(scope, dimention)

    def distance(self, vector: np.ndarray):
        # euclidean distance
        return np.linalg.norm(vector - self.loc)

    def advance(self, factor, point):

        old_loc = self.loc.copy()  # before update
        self.loc = factor * (point + self.loc)
        return np.linalg.norm(old_loc - self.loc)  # euclidean distance / how much did 'self' move

    def get_loc(self) -> np.ndarray:
        return self.loc


class SOM:

    def __init__(self, shape: tuple, learning_rate: float):

        if len(shape) > 2:
            raise ArgumentError('can only deal with up to 2 dimensional topology')
        if len(shape) <= 0:
            raise ArgumentError('no topology, argument `shape` is empty')
        for dim in shape:
            if dim <= 0:
                raise ArgumentError('only positive input')

        self.learning_rate = learning_rate
        self.shape = shape

        # allocating memory for centroids the shape of this strucute
        # also defines each centroid's neighbourhood
        self.clear()

    # resetting map
    def clear(self):
        self.cen = np.zeros(self.shape).flatten().astype(centroid)

    # get centroid by id, not obvious if the shape isn't 1 dimentional
    def get_centroid(self, id) -> centroid:

        if len(self.shape) == 1:
            return self.cen[id]
        return self.cen[id // self.shape[1]][id % self.shape[1]]  # matrix like structure

    def get_neighbours(self, id: int):

        neighbours = []

        # asserting in-bounds
        if len(self.shape) == 1:
            if id - 1 >= 0:
                neighbours.append(self.get_centroid(id - 1))
            if id + 1 < self.shape[0]:
                neighbours.append(self.get_centroid(id + 1))
        else:
            if (id - 1) % self.shape[1] != self.shape[1] - 1:
                neighbours.append(self.get_centroid(id - 1))
            if (id + 1) % self.shape[1] != 0:
                neighbours.append(self.get_centroid(id + 1))
            if (id + self.shape[1]) < self.shape[0] * self.shape[1]:
                neighbours.append(self.get_centroid(id + self.shape[1]))
            if (id - self.shape[1]) >= 0:
                neighbours.append(self.get_centroid(id - self.shape[1]))

        return neighbours

    # finding closest centroid to a data instance
    def closest(self, wins: np.ndarray, vector: np.ndarray, itr: int):

        min_score = inf
        ind = 0
        for c in self.cen:

            score = c.distance(vector) - (wins[c.id] - itr / len(wins))  # conscious formula

            if score < min_score:
                min_score = score
                ind = c.id

        wins[ind] += 1
        return ind

    def train(self, data: np.ndarray, scope: tuple):

        if len(data.shape) < 2:
            raise ArgumentError('expecting some instances of whatever vector-spase')

        # initiating centroids
        for i in range(len(self.cen)):
            self.cen[i] = centroid(i, data.shape[-1], scope)

        # executing actuall algorithm

        wins = np.zeros(len(self.cen))  # taking into account past winners
        self.cen.reshape(self.shape)    # reshaping to define the topology

        itr = 0
        diff = inf                      # a measure of change that happened in the iteration
        halt = np.abs(scope[1]) / 10    # small diff relative to the max scalar in the data
        momentum = 1                    # factor for diminishing the learning rate each epoc

        while diff > halt:

            for instance in data:

                winner = self.get_centroid(self.closest(wins, instance, itr))

                # advance closest centroid towards the data instance
                diff = winner.advance(self.learning_rate / momentum, instance)
                # and its neigbours too but less so
                for c in self.get_neighbours(winner.id):
                    c.advance(self.learning_rate / 2 * momentum, instance)

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
