#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import math

# Same as before, but modified for three dimensions
N = 10000
L = 35

spins = np.loadtxt('spins.txt')
spins = spins.reshape((N, L, L))


# First, make sure all values in the spin array are between 0 and 2*pi
assert np.all(np.logical_and(spins>=0, spins<=2 * np.pi))

def angle_mod(theta):
    if theta > math.pi:
        return theta - 2 * math.pi
    if theta < -math.pi:
        return theta + 2 * math.pi
    return theta

# Compute the vorticity here too so we can plot...
for n in xrange(N):
    vortices = []
    antivortices = []
    spin_arr = spins[n]
    for x in xrange(len(spin_arr)):
        for y in xrange(len(spin_arr[x])):
            delta = 0
            # Now we want to consider the vorticity of the point (i, j)
            # We can do so by considering differing perimeters...
            perimeter = [(x, y), (x + 1, y), (x + 2, y), (x + 3, y),
                         (x + 3, y - 1), (x + 3, y - 2), (x + 3, y - 3),
                         (x + 2, y - 3), (x + 1, y - 3), (x, y - 3),
                         (x, y - 2), (x, y - 1), (x, y)];

            old = spin_arr[perimeter[0][0] % L, perimeter[0][1] % L]
            for idx in perimeter[1:]:
                new = spin_arr[idx[0] % L, idx[1] % L]
                delta += angle_mod(new - old)
                old = new
            if 2 * 3.14 - 0.005 < delta < 2 * 3.14 + 0.005:
                vortices.append(((x + 1.5) % L, (y - 1.5) % L))
                # print (x + 1.5) % L, (y - 1.5) % L
            if -2 * 3.14 - 0.005 < delta < -2 * 3.14 + 0.005:
                antivortices.append(((x + 1.5) % L, (y - 1.5) % L))
                # print (x + 1.5) % L, (y - 1.5) % L
                # print n
    print len(vortices), len(antivortices)
x_spins = np.cos(spins[95])
y_spins = np.sin(spins[95])
X, Y = np.mgrid[0:L, 0:L]

plt.quiver(X, Y, x_spins, y_spins, spins, pivot='mid',cmap=plt.cm.hsv, clim=[0, 2* 3.15])
plt.axis('equal')
plt.plot(*zip(*vortices), linestyle='None', marker='o', color='b', markersize=10)
plt.plot(*zip(*antivortices), linestyle='None', marker='o', color='r', markersize=10)
cbar = plt.colorbar(ticks=[0, 2 * 3.14])
cbar.ax.set_yticklabels(["0", "pi", "2pi"])

plt.show()
