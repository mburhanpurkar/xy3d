#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import math


def angle_mod(theta):
    if theta > math.pi:
        return angle_mod(theta - 2 * math.pi)
    if theta < -math.pi:
        return angle_mod(theta + 2 * math.pi)
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
    
def pbc(L, n):
    return int(n % L)

def another_constrain(x):
     if x < -np.pi:
         return another_constrain(x + 2 * np.pi)
     if x > np.pi:
         return another_constrain(x - 2 * np.pi)
     return x

def make_vortex(charge, sign, start=5*math.pi / 4, verbose=False, test=False):
    assert(sign == -1 or sign == 1)
    
    # We will construct examples on a 8x8 lattice
    L = 8

    oouter_loop = make_loop(8, 0, 0)
    outer_outer_loop = make_loop(6, 1, 1)
    outer_loop = make_loop(4, 2, 2)
    inner_loop = make_loop(2, 3, 3)
    
    # Construct array
    spins = np.empty((L, L))
    spins[outer_loop[0]] = start
    spins[inner_loop[0]] = start

    # Outer loop over outer_loop
    outer_spacing = charge * 2 * math.pi / 12.0
    inner_spacing = charge * 2 * math.pi / 4.0
    for i in xrange(len(outer_loop[2:])):
        spins[outer_loop[i + 1]] = (spins[outer_loop[i]] + sign  * outer_spacing) % (2 * np.pi) * ((np.random.rand() - 0.5) * 0.05 + 1.0)

    for i in xrange(len(inner_loop[2:])):
        spins[inner_loop[i + 1]] = (spins[inner_loop[i]] + sign * inner_spacing) % (2 * np.pi) * ((np.random.rand() - 0.5) * 0.05 + 1.0)

    for i in xrange(len(outer_outer_loop)):
        spins[outer_outer_loop[i]] = np.random.rand() * 2 * np.pi

    for i in xrange(len(oouter_loop)):
        spins[oouter_loop[i]] = np.random.rand() * 2 * np.pi

    # Try computing the vorticity! For some reason, doing [-pi, pi] seems to work int the C++ implementation
    # (I'll figure this out tonight?) so let's give that a shot:
    if not test:
        delta = 0.0
        for i in xrange(len(outer_loop[1:])):
            delta += angle_mod(spins[outer_loop[i + 1]] - spins[outer_loop[i]])

        # Try plotting
        x_spins = np.cos(spins)
        y_spins = np.sin(spins)
        X, Y = np.mgrid[0:L, 0:L]

        plt.quiver(X, Y, x_spins, y_spins, spins)
        plt.axis('equal')
        plt.title("Example charge " + str(round(delta / 2.0 / math.pi, 2)) +  " vortex")
        plt.show()
    else:
        vort = np.zeros((4))
        # This needs to happen properly--loop over all possible plaquettes with periodic boundary conditions
        for x in xrange(L):
            for y in xrange(L):
                boundary = [(x, y), (x + 1, y), (x + 2, y), (x + 3, y),
                            (x + 3, y + 1), (x + 3, y + 2), (x + 3, y + 3),
                            (x + 2, y + 3), (x + 1, y + 3), (x, y + 3),
                            (x, y + 2), (x, y + 1), (x, y)]

                delta = 0.0

                i = 0
                while i < len(boundary) - 1:
                    x1 = pbc(L, boundary[i + 1][0])
                    y1 = pbc(L, boundary[i + 1][1])
                    x2 = pbc(L, boundary[i][0])
                    y2 = pbc(L, boundary[i][1])

                    delta += another_constrain(spins[x1, y1] - spins[x2, y2])
                    test = abs(another_constrain(spins[x1, y1] - spins[x2, y2])) - np.pi

                    if abs(test) < (np.pi / 1.85):
                        delta = 0.0
                        i = len(boundary)
                    i += 1
                if abs(delta + 4 * np.pi) < 0.01:
                    vort[0] += 1
                elif abs(delta + 2 * np.pi) < 0.01:
                    vort[1] += 1
                elif abs(delta - 2 * np.pi) < 0.01:
                    vort[2] += 1
                elif abs(delta - 4 * np.pi) < 0.01:
                    vort[3] += 1
        return vort
        
    if verbose:
        print_for_cpp(spins)

    return spins

visual_test = False
id_test = True
if visual_test:
    # This does a visual test to make vortices are being generated as expected
    make_vortex(1, 1, verbose=True)
    make_vortex(1, -1, verbose=True)
    make_vortex(2, 1, verbose=True)
    make_vortex(2, -1, verbose=True)
    

if id_test:
    # Unit test vortex identification. Some steps:
    # First, try randomizing over the spins on the perimenter
    # Then, try adding small perturbations to the spins inside
    # Also, check that we get only ONE for vortices of all charges!
    # Also, check that it is in the right place
    # Also randomize over the starting angle
    N = 1000
    count = 0
    for i in xrange(1000):
        if i < N / 4.0:
            print "*******************1"
            v = make_vortex(1, 1, start=0 * 2 * np.pi, test=True)
            if not np.array_equal(v, [0, 0, 1, 0]):
                print v
                count += 1
        elif i < 2 * N / 4.0:
            print "*******************-1"
            v = make_vortex(1, -1, start=0 * 2 * np.pi, test=True)
            if not np.array_equal(v, [0, 1, 0, 0]):
                print v
                count += 1
        elif i < 3 * N / 4.0:
            print "*******************2"
            v = make_vortex(2, 1, start=0 * 2 * np.pi, test=True)
            if not np.array_equal(v, [0, 0, 0, 1]):
                print v
                count += 1
        else:
            print "*******************-2"
            v = make_vortex(2, -1, start=0 * 2 * np.pi, test=True)
            if not np.array_equal(v, [1, 0, 0, 0]):
                print v
                count += 1

    print "False positves/negatives:", count

        
