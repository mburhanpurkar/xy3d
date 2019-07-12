#!/usr/bin/env python

import math
import time
import random
#import mpi_fanout
import numpy as np
import matplotlib.pyplot as plt


class Expansion():
    def __init__(self, L, J, K):
        self.L = L
        self.J = J
        self.K = K
        self.Jtilde = self.J  # self.J**2 / 4.0
        self.Ktilde = self.K  # self.J**4 * self.K
        self.state = np.random.rand(self.L, self.L, self.L) * np.pi * 2
        self.get_neighs()
        self.get_energy()

    def get_neighs(self):
        self.plaqs = np.empty((self.L, self.L, self.L, 12, 4), dtype=(int, 3))
        for x in xrange(self.L):
            for y in xrange(self.L):
                for z in xrange(self.L):
                    # First get the neighbours
                    # First, consider the four plaquettes in the xy plane
                    self.plaqs[x, y, z, 0, 0] = [x, y, z]
                    self.plaqs[x, y, z, 0, 1] = [(x + 1) % self.L, y, z]
                    self.plaqs[x, y, z, 0, 2] = [(x + 1) % self.L, (y + 1) % self.L, z]
                    self.plaqs[x, y, z, 0, 3] = [x, (y + 1) % self.L, z]

                    self.plaqs[x, y, z, 1, 0] = [x, y, z]
                    self.plaqs[x, y, z, 1, 1] = [x, (y + 1) % self.L, z]
                    self.plaqs[x, y, z, 1, 2] = [(x - 1) % self.L, (y + 1) % self.L, z]
                    self.plaqs[x, y, z, 1, 3] = [(x - 1) % self.L, y, z]

                    self.plaqs[x, y, z, 2, 0] = [x, y, z]
                    self.plaqs[x, y, z, 2, 1] = [(x - 1) % self.L, y, z]
                    self.plaqs[x, y, z, 2, 2] = [(x - 1) % self.L, (y - 1) % self.L, z]
                    self.plaqs[x, y, z, 2, 3] = [x, (y - 1) % self.L, z]

                    self.plaqs[x, y, z, 3, 0] = [x, y, z]
                    self.plaqs[x, y, z, 3, 1] = [x, (y - 1) % self.L, z]
                    self.plaqs[x, y, z, 3, 2] = [(x + 1) % self.L, (y - 1) % self.L, z]
                    self.plaqs[x, y, z, 3, 3] = [(x + 1) % self.L, y, z]

                    # One plane down, two more to go...
                    self.plaqs[x, y, z, 4, 0] = [x, y, z]
                    self.plaqs[x, y, z, 4, 1] = [x, (y + 1) % self.L, z]
                    self.plaqs[x, y, z, 4, 2] = [x, (y + 1) % self.L, (z + 1) % self.L]
                    self.plaqs[x, y, z, 4, 3] = [x, y, (z + 1) % self.L]

                    self.plaqs[x, y, z, 5, 0] = [x, y, z]
                    self.plaqs[x, y, z, 5, 1] = [x, y, (z + 1) % self.L]
                    self.plaqs[x, y, z, 5, 2] = [x, (y - 1) % self.L, (z + 1) % self.L]
                    self.plaqs[x, y, z, 5, 3] = [x, (y - 1) % self.L, z]

                    self.plaqs[x, y, z, 6, 0] = [x, y, z]
                    self.plaqs[x, y, z, 6, 1] = [x, (y - 1) % self.L, z]
                    self.plaqs[x, y, z, 6, 2] = [x, (y - 1) % self.L, (z - 1) % self.L]
                    self.plaqs[x, y, z, 6, 3] = [x, y, (z - 1) % self.L]

                    self.plaqs[x, y, z, 7, 0] = [x, y, z]
                    self.plaqs[x, y, z, 7, 1] = [x, y, (z - 1) % self.L]
                    self.plaqs[x, y, z, 7, 2] = [x, (y + 1) % self.L, (z - 1) % self.L]
                    self.plaqs[x, y, z, 7, 3] = [x, (y + 1) % self.L, z]

                    # And finally the XZ plane...
                    self.plaqs[x, y, z, 8, 0] = [x, y, z]
                    self.plaqs[x, y, z, 8, 1] = [(x + 1) % self.L, y, z]
                    self.plaqs[x, y, z, 8, 2] = [(x + 1) % self.L, y, (z + 1) % self.L]
                    self.plaqs[x, y, z, 8, 3] = [x, y, (z + 1) % self.L]

                    self.plaqs[x, y, z, 9, 0] = [x, y, z]
                    self.plaqs[x, y, z, 9, 1] = [x, y, (z + 1) % self.L]
                    self.plaqs[x, y, z, 9, 2] = [(x - 1) % self.L, y, (z + 1) % self.L]
                    self.plaqs[x, y, z, 9, 3] = [(x - 1) % self.L, y, z]

                    self.plaqs[x, y, z, 10, 0] = [x, y, z]
                    self.plaqs[x, y, z, 10, 1] = [(x - 1) % self.L, y, z]
                    self.plaqs[x, y, z, 10, 2] = [(x - 1) % self.L, y, (z - 1) % self.L]
                    self.plaqs[x, y, z, 10, 3] = [x, y, (z - 1) % self.L]

                    self.plaqs[x, y, z, 11, 0] = [x, y, z]
                    self.plaqs[x, y, z, 11, 1] = [x, y, (z - 1) % self.L]
                    self.plaqs[x, y, z, 11, 2] = [(x + 1) % self.L, y, (z - 1) % self.L]
                    self.plaqs[x, y, z, 11, 3] = [(x + 1) % self.L, y, z]

    def get_energy(self):
        penergy = 0
        spin_energy = -self.Jtilde * np.sum([np.cos(self.state - np.roll(self.state, 1, axis=i)) for i in xrange(3)])
        # Energy due to plaquettes
        for x in xrange(self.L):
            for y in xrange(self.L):
                for z in xrange(self.L):
                    for idx in xrange(12):
                        prod = 1.0
                        for jdx in xrange(4):
                            i, j, k = self.plaqs[x, y, z, idx, jdx]
                            l, m, n = self.plaqs[x, y, z, idx, (jdx + 1) % 4]
                            prod *= math.cos((self.state[i, j, k] - self.state[l, m, n]) / 2.0)
                        penergy += prod

        self.energy = spin_energy - self.Ktilde * penergy

    def bond_energy(self, x, y, z, central):
        return -self.Jtilde * (math.cos((self.state[(x - 1) % self.L, y, z] - central)) +
                               math.cos((self.state[(x + 1) % self.L, y, z] - central)) +
                               math.cos((self.state[x, (y - 1) % self.L, z] - central)) +
                               math.cos((self.state[x, (y + 1) % self.L, z] - central)) +
                               math.cos((self.state[x, y, (z - 1) % self.L] - central)) +
                               math.cos((self.state[x, y, (z + 1) % self.L] - central)))

    def plaq_energy(self, x, y, z, central_angle):
        # Compute the energy from the plaquettes at (x, y, z), if the spin at that point had value central_angle
        old_angle = self.state[x, y, z]
        self.state[x, y, z] = central_angle
        penergy = 0
        for idx in xrange(12):
            prod = 1.0
            for jdx in xrange(4):
                i, j, k = self.plaqs[x, y, z, idx, jdx]
                l, m, n = self.plaqs[x, y, z, idx, (jdx + 1) % 4]
                prod *= math.cos((self.state[i, j, k] - self.state[l, m, n]) / 2.0)
            penergy += prod
        penergy *= -1.0 * self.Ktilde
        self.state[x, y, z] = old_angle
        return penergy

    def constrain(self, alpha):
        # Return alpha in [0, 2pi]
        return alpha % (2 * math.pi)

    def flip(self):
        x = random.randint(0, self.L - 1)
        y = random.randint(0, self.L - 1)
        z = random.randint(0, self.L - 1)
        flip_axis = random.random() * math.pi
        newangle = self.constrain(2.0 * flip_axis - self.state[x, y, z])

        E1 = self.bond_energy(x, y, z, self.state[x, y, z]) + self.plaq_energy(x, y, z, self.state[x, y, z])
        E2 = self.bond_energy(x, y, z, newangle) + self.plaq_energy(x, y, z, newangle)

        if E2 < E1 or random.random() < min(1, math.exp(-(E2 - E1))):
            self.state[x, y, z] = newangle
            self.energy += E2 - E1

    def magnetization(self):
        return math.sqrt(np.mean(np.cos(self.state))**2 + np.mean(np.sin(self.state))**2)


################################################################################


# def f(L, J, K, neighs, plaqs, ntherm, nmc, nmeas):
#     test = Expansion(L, J, K, neighs, plaqs)
#     ene = 0
#     ene2 = 0
#     mag = 0
#     mag2 = 0
#     flux = 0
#     flux2 = 0
#     # Thermalize
#     for i in xrange(test.L**3 * ntherm):
#         test.flip()
#     # Take measurements every 35 flips
#     for i in xrange(nmc):
#         for j in xrange(test.L**3 * nmeas):
#             test.flip()
#         # loop.append(np.average(test.poly_loop()))
#         magnetization = test.magnetization()
#         mag += magnetization
#         mag2 += magnetization**2
#         ene += test.energy
#         ene2 += test.energy**2
#         flux += test.bond_energy / test.K
#         flux2 += test.bond_energy**2 / test.K**2
#     mag /= nmc
#     mag2 /= nmc
#     ene /= nmc
#     ene2 /= nmc
#     flux /= nmc
#     flux2 /= nmc
#     return mag, (mag2 - mag**2), ene, (ene2 - ene**2), (flux2 - flux**2)
#
#
# def simulate_parallel(L, vary_J, const, start, stop, delta, ntherm, nmc, nmeas):
#     # Uses mpi_fanout.py to execute tasks in mpi_fanout.task in parallel
#     neighs, plaqs = get_neighs(L)
#     if vary_J:
#         print L, "K =", const, "J =", np.arange(start, stop, delta)
#         task_list = [mpi_fanout.task(f, L, i, const, neighs, plaqs, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
#     else:
#         print L, "J =", const, "K =", np.arange(start, stop, delta)
#         task_list = [mpi_fanout.task(f, L, const, i, neighs, plaqs, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
#     print mpi_fanout.run_tasks(task_list)


# It looks like this is working quite well!
# TODO add parallelization
def simulate(L, J, Kmin, Kmax, delta, ntherm, nmc, nmeas):
    magl = []
    susl = []
    sphl = []
    enel = []
    test = Expansion(L, J, Kmin)

    # Start looping over different temperatures
    for Kstar in np.arange(Kmin, Kmax, delta):
        ene = 0
        ene2 = 0
        mag = 0
        mag2 = 0

        print "K =", Kstar
        test.Ktilde = Kstar

        # Thermalize
        print "Thermalizing..."
        for i in xrange(ntherm * L**3):
            test.flip()

        print "Taking measurements..."
        # Make a measurement
        for i in xrange(nmc):
            print i
            for j in xrange(nmeas * L**3):
                test.flip()
            magnetization = test.magnetization()
            mag += magnetization
            mag2 += magnetization**2
            ene += test.energy
            ene2 += test.energy**2

        # Normalize
        mag /= 1.0 * nmc
        mag2 /= 1.0 * nmc
        ene /= 1.0 * nmc
        ene2 /= 1.0 * nmc
        magl.append(mag)
        susl.append(mag2 - mag**2)
        enel.append(ene)
        sphl.append(ene2 - ene**2)

    print np.arange(Kmin, Kmax, delta)
    print magl
    print enel
    print susl
    print sphl

    plt.plot(np.arange(Kmin, Kmax, delta), sphl)
    plt.xlabel("K")
    plt.ylabel("Specific Heat")
    plt.show()


################################################################################


if __name__ == "__main__":
    # We want J=0 and scan in K--check if we get the C++ result of no divergence
    # of any quantities!

    # NOTE: the specific result is that doing J, K directly doesn't give a peak
    # but doing Jtilde and Ktilde does
    simulate(8, 0, 0, 1.5, 0.2, 200, 200, 30)
