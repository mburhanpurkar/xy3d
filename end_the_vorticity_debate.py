#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import math


# Important note: the modulo function in python ALWAYS RETURNS POSITIVE VALUES! That is, the spins are all in [0, 2pi]
# This is not true in C++ so please be careful!

def angle_mod(theta):
    if theta > math.pi:
        return theta - 2 * math.pi
    if theta < -math.pi:
        return theta + 2 * math.pi
    return theta

def make_vortex(charge, sign, start=5*math.pi):
    assert(sign == -1 or sign == 1)
    
    # We will construct examples on a 4x4 lattice
    L = 4
    outer_loop = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (2, 3), (1, 3), (0, 3), (0, 2), (0, 1), (0, 0)]
    inner_loop = [(1, 1), (2, 1), (2, 2), (1, 2), (1, 1)]

    # Construct array
    spins = np.empty((4, 4))
    spins[outer_loop[0]] = start / L
    spins[inner_loop[0]] = start / L

    # Outer loop over outer_loop
    outer_spacing = charge * 2 * math.pi / 12
    inner_spacing = charge * 2 * math.pi / 4
    for i in xrange(len(outer_loop[2:])):
        spins[outer_loop[i + 1]] = (spins[outer_loop[i]] + sign  * outer_spacing) % (2 * math.pi)

    for i in xrange(len(inner_loop[2:])):
        spins[inner_loop[i + 1]] = (spins[inner_loop[i]] + sign * inner_spacing) % (2 * math.pi)

    # Try computing the vorticity! For some reason, doing [-pi, pi] seems to work int the C++ implementation
    # (I'll figure this out tonight?) so let's give that a shot:
    delta = 0.0
    for i in xrange(len(outer_loop[1:])):
        delta += angle_mod(spins[outer_loop[i + 1]] - spins[outer_loop[i]])

    # Try plotting!
    x_spins = np.cos(spins)
    y_spins = np.sin(spins)
    X, Y = np.mgrid[0:L, 0:L]

    plt.quiver(X, Y, x_spins, y_spins, spins)
    plt.axis('equal')
    plt.title("Example charge " + str(round(delta / 2.0 / math.pi, 2)) +  " vortex")
    plt.show()

make_vortex(1, 1)
make_vortex(1, -1)
make_vortex(2, 1)
make_vortex(2, -1)
