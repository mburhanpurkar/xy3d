#!/usr/bin/env python

import math
import random
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
        self.plaqs = np.empty((self.L, self.L, self.L, 12, 4, dtype=(int, 3)))
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
        for x in xrange(self.L):
            for y in xrange(self.L):
                for z in xrange(self.L):
                    # Energy due to neighbours
                    nenergy = 0
                    spin = self.state[x, y, z]
                    for i in xrange(6):
                        nenergy += math.cos(self.state[neighs[x, y, z, i]] - spin)
                    nenergy *= -1.0 * self.Jtilde

                    # Energy due to plaquettes
                    penergy = 0
                    for i in xrange(12):
                        prod = 1.0
                        for j in xrange(4):
                            prod *= math.cos((self.state[self.plaqs[x, y, z, i, j]] -
                                              self.state[self.plaqs[x, y, z, i, (j + 1) % 4]]) / 2.0)
                        penergy += prod
                    penergy *= -1.0 * self.Ktilde
        self.energy = penergy + nenergy

    def bond_energy(self, x, y, z, central_angle):
        # Compute the energy from the neigbours of (x, y, z), if the spin at that point had value central_angle
        return -1.0 * self.Jtilde * sum([math.cos(self.state[neighs[x, y, z, i]] - central_angle) for i in xrange(6)])

    def plaq_energy(self, x, y, z, central_angle):
        # Compute the energy from the plaquettes at (x, y, z), if the spin at that point had value central_angle
        old_angle = self.state[x, y, z]
        self.state[x, y, z] = central_angle
        penergy = 0
        for i in xrange(12):
            prod = 1.0
            for j in xrange(4):
                prod *= math.cos((self.state[self.plaqs[x, y, z, i, j]] -
                                  self.state[self.plaqs[x, y, z, i, (j + 1) % 4]]) / 2.0)
            penergy += prod
        penergy *= -1.0 * self.Ktilde
        self.state[x, y, z] = old_angle
        return penergy

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


################################################################################


def simulate(L, J, Kmin, Kmax, delta, ntherm, nmeas):
    sus = []
    test = Expansion(L, J, Kmin)

    # Start looping over different temperatures
    for Kstar in np.arange(Kmin, Kmax, delta):
        m = np.zeros(2)
        test.K = Kstar
        test.get_energy()

        # Thermalize
        for i in xrange(ntherm * L**3):
            test.flip()

        # Make a measurement
        for i in xrange(nmc):
            for j in xrange(nmeas * L**3):
                test.flip()

            # Update flux Susceptibility
            m[0] += test.bond_energy
            m[1] += test.bond_energy**2

        # Normalize
        m /= nmc
        sus.append(m[1] - m[0]**2)

    print sus
    plt.plot(np.arange(Kmin, Kmax, delta), sus)
    plt.xlabel("K")
    plt.ylabel("Flux Susceptibility")
    plt.show()
