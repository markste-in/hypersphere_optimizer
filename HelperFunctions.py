from timeit import default_timer as timer
import timeit
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm

def t_decorator(func):
    def func_wrapper(*args, **kwargs):
        start = timer()
        result = func(*args, **kwargs)
        end = timer()
        print(func.__name__, 'took', end - start, 's')
        return result

    return func_wrapper


class CodeTimer:
    def __init__(self, name=None):
        self.name = " '" + name + "'" if name else ''

    def __enter__(self):
        self.start = timeit.default_timer()

    def __exit__(self, exc_type, exc_value, traceback):
        self.took = (timeit.default_timer() - self.start) * 1000.0
        print('Code block' + self.name + ' took: ' + str(self.took) + ' ms')


def plot_surface(f):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.arange(-3, 3, 0.5)
    X, Y = np.meshgrid(x, y)
    XY = [list(a) for a in zip(np.ravel(X), np.ravel(Y))]
    zs = np.array(f(XY))
    # zs[zs > 50] = np.nan
    Z = zs.reshape(X.shape)

    # surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, vmin=0, vmax=500)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()