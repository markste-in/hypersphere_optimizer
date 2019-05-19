from timeit import default_timer as timer
import timeit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from matplotlib import cm
import ray


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
    x = y = np.arange(-3, 3, 0.05)
    X, Y = np.meshgrid(x, y)
    XY = [list(a) for a in zip(np.ravel(X), np.ravel(Y))]
    zs = np.array([f(i) for i in XY ])
    # zs[zs > 50] = np.nan
    Z = zs.reshape(X.shape)

    # surf = ax.plot_surface(X, Y, Z, cmap=cm.jet, vmin=0, vmax=500)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.jet)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

#Code below partially by https://github.com/pablormier/yabox/ :)

# Default configuration
contourParams = dict(
    zdir='z',
    alpha=0.5,
    zorder=1,
    antialiased=True,
    cmap=cm.PuRd_r
)

surfaceParams = dict(
    rstride=1,
    cstride=1,
    linewidth=0.1,
    edgecolors='k',
    alpha=0.5,
    antialiased=True,
    cmap=cm.PuRd_r
)
bounds = [-1,1]

def plot3d(f,points=100, contour_levels=20, ax3d=None, figsize=(12, 8),
           view_init=None, surface_kwds=None, contour_kwds=None):

    contour_settings = dict(contourParams)
    surface_settings = dict(surfaceParams)
    if contour_kwds is not None:
        contour_settings.update(contour_kwds)
    if surface_kwds is not None:
        surface_settings.update(surface_kwds)
    xbounds, ybounds = bounds[0], bounds[1]
    x = y = np.arange(-3, 3, 0.05)
    X, Y = np.meshgrid(x, y)
    XY = [list(a) for a in zip(np.ravel(X), np.ravel(Y))]
    zs = np.array(ray.get([f.remote(i) for i in XY ]))
    Z = zs.reshape(X.shape)
    if ax3d is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)
        if view_init is not None:
            ax.view_init(*view_init)
    else:
        ax = ax3d
    # Make the background transparent
    ax.patch.set_alpha(0.0)
    # Make each axis pane transparent as well
    ax.w_xaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_yaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    ax.w_zaxis.set_pane_color((0.0, 0.0, 0.0, 0.0))
    surf = ax.plot_surface(X, Y, Z, **surface_settings)
    contour_settings['offset'] = np.min(Z)
    cont = ax.contourf(X, Y, Z, contour_levels, **contour_settings)
    if ax3d is None:
        plt.show()
    return ax
