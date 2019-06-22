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
    # This function makes a vortex in 2d of radius 2 lattice spaces and fills the rest
    # with random spins
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
                    # test = abs(another_constrain(spins[x1, y1] - spins[x2, y2])) - np.pi
                    test = abs(another_constrain(spins[x1, y1] - spins[x2, y2]))

                    if abs(test) > (np.pi / 2.1):
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


def make_vortex_visual_test():
    # This does a visual test to make vortices are being generated as expected
    make_vortex(1, 1, verbose=True)
    make_vortex(1, -1, verbose=True)
    make_vortex(2, 1, verbose=True)
    make_vortex(2, -1, verbose=True)
    

def vortex_id_unit_test():
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
        
def make_2d_vortex(charge, sign, L, plot=False):
    # Makes an L*L sized isolated vortex
    spinlist = []
    for i in xrange(L / 2):
        start = i
        size = L - 2 * i
        spinlist.append(make_loop(size, start, start))

    spins = np.empty((L, L))
    for i in xrange(len(spinlist)):
        spins[spinlist[i][0]] = 3.0
        perimeter = (L - 2 * i - 1) * 4
        spacing = charge * 2 * math.pi / perimeter
        for j in xrange(len(spinlist[i][2:])):
            spins[spinlist[i][j + 1]] = (spins[spinlist[i][j]] + sign * spacing) % (2 * np.pi)

    if plot:
        x_spins = np.cos(spins)
        y_spins = np.sin(spins)
        X, Y = np.mgrid[0:L, 0:L]
        plt.quiver(X, Y, x_spins, y_spins, spins)
        plt.axis('equal')
        plt.grid(True)
        plt.show()

    return spins

def get_2d_energy(spins):
    # Computes the K term energy (J is commented out)
    energy = 0.0
    for i in xrange(len(spins) - 1):
        prod = 1.0
        for j in xrange(len(spins[i]) - 1):
            # Compute a product over a plaquette
            prod *= math.cos((spins[i, j] - spins[i + 1, j]) / 2.0)
            prod *= math.cos((spins[i + 1, j] - spins[i + 1, j - 1]) / 2.0)
            prod *= math.cos((spins[i + 1, j - 1] - spins[i, j - 1]) / 2.0)
            prod *= math.cos((spins[i, j - 1] - spins[i, j]) / 2.0)
        energy += prod

    # Let's also compute the J term for good measure--it's probably easiest to first
    # do all the horizontal terms and then do all the vertical ones
    # energy1 = 0.0
    # for i in xrange(len(spins)):
    #     for j in xrange(len(spins) - 1):
    #         energy1 += math.cos(spins[i, j] - spins[i, j + 1])
    # for i in xrange(len(spins) - 1):
    #     for j in xrange(len(spins)):
    #         energy1 += math.cos(spins[i, j] - spins[i + 1, j])
    
    return energy#, energy1

def make_3d_vortex(charge, sign, L):
    spins = np.empty((L, L, L))
    for i in xrange(len(spins)):
        spins[i] = make_2d_vortex(charge, sign, L)
    return spins

def get_3d_energy(spins):
    # This won't quite work because get_2d_energy currently returns two values haha
    x_energy = sum([get_2d_energy(spins[i]) for i in xrange(len(spins))])
    y_energy = sum([get_2d_energy(spins[:, i]) for i in xrange(len(spins[0]))])
    z_energy = sum([get_2d_energy(spins[:, :, i]) for i in xrange(len(spins[0][0]))])
    return x_energy + y_energy + z_energy

def k_term_tests():
    # ***** First check results in 2d *****
    # We expect the following energies
    # 2pi: 0.25*K(L - 1)*(L - 1)
    # 4pi: 0
    # Aligned: -K(L - 1)*(L - 1)
    assert (-get_2d_energy(make_2d_vortex(1, 1, 2, plot=True)) - 0.25 < 0.0001)
    assert (-get_2d_energy(make_2d_vortex(1, -1, 2, plot=True)) - 0.25 < 0.0001)
    assert (abs(get_2d_energy(make_2d_vortex(2, 1, 2, plot=True))) < 0.0001)
    assert (abs(get_2d_energy(make_2d_vortex(2, -1, 2, plot=True))) < 0.0001)

    # Now let's check the scaling--note: it doesn't make sense to have
    # an odd value of L since the vortex will not be centered and the function
    # looks pretty wonky
    x_axis = range(2, 30, 2)
    energies_2pi = []
    energies_4pi = []
    for L in x_axis:
        energies_2pi.append(-get_2d_energy(make_2d_vortex(1, 1, L)))
        energies_4pi.append(-get_2d_energy(make_2d_vortex(2, 1, L)))
    plt.plot(x_axis, energies_2pi, 'o', label='2pi K')
    plt.plot(x_axis, energies_4pi, 'o', label='4pi K')
    plt.ylabel('Energy')
    plt.xlabel('L')
    plt.title('Energy K term')
    plt.legend()
    plt.show()

    # Now repeat the same exercise in 3D, sans plotting
    x_axis = range(2, 30, 2)
    energies_2pi = []
    energies_4pi = []
    for L in x_axis:
        energies_2pi.append(-get_3d_energy(make_3d_vortex(1, 1, L)))
        energies_4pi.append(-get_3d_energy(make_3d_vortex(2, 1, L)))
    plt.plot(x_axis, energies_2pi, 'o', label='2pi K')
    plt.plot(x_axis, energies_4pi, 'o', label='4pi K')
    plt.ylabel('Energy')
    plt.xlabel('L')
    plt.title('Energy K term')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    k_term_tests()
