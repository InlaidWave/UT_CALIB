import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Example measurement data
x, y, z = measurements[:,0], measurements[:,1], measurements[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=5, c='blue', alpha=0.6)

# Optional: draw unit sphere for reference
u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
xs = np.cos(u)*np.sin(v)
ys = np.sin(u)*np.sin(v)
zs = np.cos(v)
ax.plot_wireframe(xs, ys, zs, color='r', alpha=0.2)

plt.show()  