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


def print_for_cpp(arr):
    print "{",
    for i in xrange(len(arr)):
        print "{",
        for j in xrange(len(arr[0])):
            if j == (len(arr[0]) - 1):
                print str(arr[i, j]),
            else:
                print str(arr[i, j]), ", ",
        if i == (len(arr) - 1):
            print "}",
        else:
            print "},",
    print "}"


def make_loop(size, start_x, start_y):
    # Makes a size * size with bottom left corner at (start_x, start_y)
    x = start_x
    y = start_y

    ll = [(start_x, start_y)]

    for plus_x in range(0, size - 1):
        x += 1
        ll.append((x, y))

    for plus_y in range(0, size - 1):
        y += 1
        ll.append((x, y))
        
    for minus_x in range(0, size - 1):
        x -= 1
        ll.append((x, y))
        
    for minus_y in range(0, size - 1):
        y -= 1
        ll.append((x, y))
        
    return ll
    

def make_vortex(charge, sign, start=5*math.pi / 4, verbose=False):
    assert(sign == -1 or sign == 1)
    
    # We will construct examples on a 4x4 lattice
    L = 6
    outer_outer_loop = make_loop(6, 0, 0)
    outer_loop = make_loop(4, 1, 1)
    inner_loop = make_loop(2, 2, 2)
    
    # Construct array
    spins = np.empty((L, L))
    spins[outer_loop[0]] = start
    spins[inner_loop[0]] = start

    # Outer loop over outer_loop
    outer_spacing = charge * 2 * math.pi / 12.0
    inner_spacing = charge * 2 * math.pi / 4.0
    for i in xrange(len(outer_loop[2:])):
        spins[outer_loop[i + 1]] = (spins[outer_loop[i]] + sign  * outer_spacing) % (2 * math.pi)

    for i in xrange(len(inner_loop[2:])):
        spins[inner_loop[i + 1]] = (spins[inner_loop[i]] + sign * inner_spacing) % (2 * math.pi)

    for i in xrange(len(outer_outer_loop)):
        spins[outer_outer_loop[i]] = np.random.rand() * 2 * np.pi

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

    if verbose:
        print_for_cpp(spins)

make_vortex(1, 1, verbose=True)
make_vortex(1, -1, verbose=True)
make_vortex(2, 1, verbose=True)
make_vortex(2, -1, verbose=True)
