import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
np.set_printoptions(precision=3)
N = 500
r = 5
r = np.ones(N)*r
noise = np.random.uniform(0,1,N)
noise2 = np.random.uniform(0,1,N)
theta = 2 * np.pi * noise

phi = np.arccos(1 - 2 * noise2)
x = r* np.sin(phi) * np.cos(theta)
y = r* np.sin(phi) * np.sin(theta)
z = r* np.cos(phi)



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x,y,z, marker='o')


fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(theta, r)

plt.show()