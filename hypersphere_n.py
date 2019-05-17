import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

from rs_optimize import *
from test_functions import *




def plot_surface(f):

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3,3, 0.5)
    X, Y = np.meshgrid(x, y)
    XY = [list(a) for a in zip(np.ravel(X), np.ravel(Y))]
    zs = np.array(f(XY))
    #zs[zs > 50] = np.nan
    Z = zs.reshape(X.shape)

    #surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, vmin=0, vmax=500)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    #fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
dim = 2e5
noiseTableSize = 1000
candidateSize = 10

print(candidateSize*dim , "points need to be calculate at every step")



issue = Issue('sphere',dim=dim)


optimize(issue, local_stepSize=.1, max_episodes=5000, N=candidateSize, noiseTable=None)
#plot_surface(issue.f)