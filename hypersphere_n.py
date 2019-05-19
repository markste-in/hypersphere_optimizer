from rs_optimize import *
from test_functions import *
import multiprocessing
import HelperFunctions
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


cpu_count = multiprocessing.cpu_count()

dim = 2

candidateSize = 5

print('starting rays with', cpu_count, 'CPUs')
ray.init(num_cpus=cpu_count, local_mode=False, include_webui=True)

issue = Issue('rosenbrock', dim=dim)

_, all_points = optimize(issue, local_stepSize=.001, max_episodes=300, N=candidateSize)
print(all_points)


ax = plt.axes(projection='3d')
ax = HelperFunctions.plot3d(Rosenbrock_function,ax3d=ax)
x,y,z = zip(*all_points)
ax.plot3D(x, y, z, 'black',  linewidth=5)
plt.show()
