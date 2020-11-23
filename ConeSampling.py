import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

INCIDENCE_ANGLE = np.pi / 6  # 30deg

# Random points within a sphere
rng = np.random.default_rng()
n = 50
R = 2

phi = rng.uniform(0, 2 * np.pi, n)
costheta = rng.uniform(-np.cos(INCIDENCE_ANGLE), np.cos(INCIDENCE_ANGLE), n)
u = rng.uniform(0, 1, n)

theta = np.arccos(costheta)
r = R * np.cbrt(u)

x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)

an = np.linspace(0, 2 * np.pi, 100)
fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect': 'auto'})
ax.scatter(x, y, z, s=100, c='r', zorder=10);
plt.show()