#!python
#cython: language_level=3


import numpy as np
import HelperFunctions

class Issue():

    def __init__(self, fname, dim):
        self.f = None
        if (fname == 'rastrigin'):
            self.f = Rastrigin_function
        elif (fname == 'sphere'):
            self.f = Sphere_function
        elif (fname == 'rosenbrock'):
            self.f = Rosenbrock_function
        else:
            raise NotImplementedError

        self.dim = int(dim)


def Rastrigin_function(X):
    ret = list()

    for i in X:
        ret.append(10 + np.sum(np.square(np.array(i)) - np.cos(2*np.pi*np.array(i))))

    return ret

def Sphere_function(X):
    ret = list()

    for i in X:
        ret.append(np.sum(np.square(i)))

    return ret

def Rosenbrock_function(X):
    ret = list()
    for point in X:
        fun = list()
        for i in range(len(point) - 1):
            fun.append(100*np.square(point[i + 1] - np.square(point[i])) + np.square(1 - point[i]))
        ret.append(np.sum(fun))
    return ret