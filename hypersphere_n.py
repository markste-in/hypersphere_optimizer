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
dim = 1e4
noiseTableSize = 1000
candidateSize = 20

print(candidateSize*dim , "points need to be calculate at every step")

assert noiseTableSize > candidateSize, 'You create too many candidates for your noiseTable'
assert (noiseTableSize/candidateSize)>20, ' Your noise tabe is too small'

issue = Issue('rastrigin',dim=dim)

# print('creating noise table...')
# noiseTable = points_on_sphere(dim=dim,N=noiseTableSize)
# print('noise table created with shape',noiseTable.shape)
optimize(issue, local_stepSize=.1, max_episodes=50000, N=candidateSize, noiseTable=None)
#plot_surface(issue.f)