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
from sympy import true

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

        return "id: " + str(self.id) + ", loc: " + str(self.loc) + ", neighbours: " + str(self.neighbours) + "\n"



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
        self.clear(start)

    # resetting map
    def clear(self, start: np.ndarray):

        SIZE = MUL(self.shape)

        def id_to_cor(id):          # cor = centroid's topology-cordinates

            size = MUL(self.shape)
            cor = [0] * len(self.shape)
            dim = len(self.shape) -1

            while dim >= 0:
                size //= self.shape[dim]
                val = id // size
                cor[dim] = val
                id -= (size*val)
                dim -= 1
            return cor

        def cor_to_id(cor):

            id = 0
            size = 1

            for dim in range(len(self.shape)):
                id += size*cor[dim]
                size *= self.shape[dim]
            return id


        for id in range(SIZE):

            cor = id_to_cor(id)
            self.cen[id] = centroid(id, start.copy())

            for dim in range(len(cor)):   # connecting to neighbours

                if cor[dim] +1 < self.shape[dim]:   # asserting in-bounds
                    cor[dim] +=1
                    self.cen[id].add_neighbour(cor_to_id(cor))
                    cor[dim] -=1
                if cor[dim] -1 >= 0:
                    cor[dim] -=1
                    self.cen[id].add_neighbour(cor_to_id(cor))
                    cor[dim] +=1


    def __repr__(self) -> str:

        ans = ""
        for c in self.cen.values():
            ans += c.__repr__()
        return ans


    # get centroid by id, not obvious if the shape isn't 1 dimensional
    def get_centroid(self, id) -> centroid:
        return self.cen[id]

    def get_neighbours(self, src: int, depth: int):

        neighbourhood = set()
        neighbourhood.add(src)

        for i in range(depth):
            tmp = neighbourhood
            for id in neighbourhood:
                tmp = tmp.union(self.get_centroid(id).neighbours.copy())
            neighbourhood = tmp

        neighbourhood.remove(src)
        return neighbourhood

    # finding closest centroid to a data instance
    def closest(self, wins: np.ndarray, vector: np.ndarray, itr: int):

        min_score = inf
        ind = 0
        for c in self.cen.values():

            score = c.distance(vector) + (wins[c.id] / len(wins))  # conscious formula

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
            depth -= 1

        return self.cen.copy(), itr, wins

    # this method should display the centroids together with their topology (lines between neighbours)
    def display(self, ax=None, show_lines=True):

        if ax is None:
            ax = plt.gca()

        dots = np.array([c.get_loc() for c in self.cen.values()])
        ax.scatter(dots[:, 0], dots[:, 1])
        
        if show_lines:
            for c1 in self.cen.values():
                #get all the points that have an id that is bigger than our current point (to avoid drawing a line over a line) 
                lines = [[c1.get_loc(),self.cen[i].get_loc()] for i in self.get_neighbours(c1.id,1) if i>c1.id]
                #add all lines found to axes
                ax.add_collection(LineCollection(lines))
        plt.show()

    # TODO: display, prephormance avaluation
    # and of course monkey hands