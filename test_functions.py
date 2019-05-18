import numpy as np
import HelperFunctions
import ray


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
            self.f = fname

        self.dim = int(dim)


@ray.remote
def Rastrigin_function(X):
    a = np.square(np.array(X))
    b = -10 * np.cos(2 * np.pi * np.array(X))
    sum = np.sum(a + b)
    ret = (10 * 2 + sum)
    assert ret > -0.5, "Negative Value"
    return ret


@ray.remote
def Sphere_function(X):
    ret = np.sum(np.square(X))

    return ret


@ray.remote
def Rosenbrock_function(X):
    fun = list()
    for i in range(len(X) - 1):
        fun.append(100 * np.square(X[i + 1] - np.square(X[i])) + np.square(1 - X[i]))
    ret = np.sum(fun)
    return ret
