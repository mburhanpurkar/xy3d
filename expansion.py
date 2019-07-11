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
        self.Jtilde = self.J**2 / 4.0
        self.Ktilde = self.J**4 * self.K
        self.state = np.random.rand(self.L, self.L, self.L) * np.pi * 2
        self.get_neighs()
        self.get_energy()

    def get_neighs(self):
        self.neighs = np.empty((self.L, self.L, self.L, 6), dtype=(int, 3))
        self.plaqs = np.empty((self.L, self.L, self.L, 12, 4), dtype=(int, 3))
        for x in xrange(self.L):
            for y in xrange(self.L):
                for z in xrange(self.L):
                    # First get the neighbours
                    self.neighs[x, y, z, 0] = [x, y, (z + 1) % self.L]
                    self.neighs[x, y, z, 1] = [x, y, (z - 1) % self.L]
                    self.neighs[x, y, z, 2] = [x, (y + 1) % self.L, z]
                    self.neighs[x, y, z, 3] = [x, (y - 1) % self.L, z]
                    self.neighs[x, y, z, 4] = [(x + 1) % self.L, y, z]
                    self.neighs[x, y, z, 5] = [(x - 1) % self.L, y, z]

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
        nenergy = 0
        penergy = 0

        for x in xrange(self.L):
            for y in xrange(self.L):
                for z in xrange(self.L):
                    # Energy due to neighbours
                    spin = self.state[x, y, z]
                    for idx in xrange(6):
                        i, j, k = self.neighs[x, y, z, idx]
                        nenergy += math.cos(self.state[i, j, k] - spin)

                    # Energy due to plaquettes
                    for idx in xrange(12):
                        prod = 1.0
                        for jdx in xrange(4):
                            i, j, k = self.plaqs[x, y, z, idx, jdx]
                            l, m, n = self.plaqs[x, y, z, idx, (jdx + 1) % 4]
                            prod *= math.cos((self.state[i, j, k] - self.state[l, m, n]) / 2.0)
                        penergy += prod

        self.energy = -self.Jtilde * nenergy - self.Ktilde * penergy

    def bond_energy(self, x, y, z, central_angle):
        # Compute the energy from the neigbours of (x, y, z), if the spin at that point had value central_angle
        sum = 0
        for idx in xrange(6):
            i, j, k = self.neighs[x, y, z, idx]
            sum += math.cos(self.state[i, j, k] - central_angle)
        return -1.0 * self.Jtilde * sum

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
        flip_axis = random.random() * math.pi * 2
        newangle = self.constrain(2.0 * flip_axis - self.state[x, y, z])

        E1 = self.bond_energy(x, y, z, self.state[x, y, z]) + self.plaq_energy(x, y, z, self.state[x, y, z])
        E2 = self.bond_energy(x, y, z, newangle) + self.plaq_energy(x, y, z, newangle)

        if random.random() < min(1, math.exp(-(E2 - E1))):
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
        test.K = Kstar

        if Kstar == Kmin:
            t1 = time.time()
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

        if Kstar == Kmin:
            t2 = time.time()
            print "Time for one cross section:", (t2 - t1) / 60.0, "(mins)"

        # Normalize
        mag /= nmc
        mag2 /= nmc
        ene /= nmc
        ene2 /= nmc
        magl.append(mag)
        susl.append(mag2 - mag**2)
        enel.append(ene)
        sphl.append(ene2 - ene**2)

    print np.arange(Kmin, Kmax, delta)
    print magl
    print enel
    print susl
    print sphl

    plt.plot(np.arange(Kmin, Kmax, delta), susl)
    plt.xlabel("K")
    plt.ylabel("Susceptibility")
    plt.show()


################################################################################


if __name__ == "__main__":
    simulate(8, 1e-10, 0.75, 1.05, 0.025, 200, 200, 15)
