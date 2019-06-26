#!/usr/bin/env python

import math
import random
import numpy as np
import matplotlib.pyplot as plt


class IsingGauge3d():
    def __init__(self, L, J, K, rand):
        self.L = L
        self.K = K
        self.J = J
        if rand:
            self.random_init()
        else:
            self.uniform_init()
        self.get_plaqs()
        self.energy = self.get_energy()

    def uniform_init(self):
        self.spins = np.zeros((self.L, self.L, self.L))
        self.dual = np.ones((3, self.L, self.L, self.L))

    def random_init(self):
        self.spins = np.random.rand(self.L, self.L, self.L) * 2 * np.pi
        self.dual = np.random.choice([-1, 1], (3, self.L, self.L, self.L))
    
    def get_plaqs(self):
        # The first index tells us, 0 -- side, 1 -- back, 2 -- top
        # Then the second, third, and fourth give us (x, y, z)
        # The final index gives us one bond of four in the plaquette! (i.e. how to index the dual lattice)
        self.plaq = np.empty((3, self.L, self.L, self.L, 4), dtype=(int,4))
        for x in xrange(self.L):
            for y in xrange(self.L):
                for z in xrange(self.L):
                    self.plaq[0, x, y, z, 0] = (0, x, y, z)
                    self.plaq[0, x, y, z, 1] = (2, x, y, z)
                    self.plaq[0, x, y, z, 2] = (0, x, y, (z + 1) % self.L)
                    self.plaq[0, x, y, z, 3] = (2, (x + 1) % self.L, y, z)

                    self.plaq[1, x, y, z, 0] = (1, x, y, z)
                    self.plaq[1, x, y, z, 1] = (2, x, y, z)
                    self.plaq[1, x, y, z, 2] = (1, x, y, (z + 1) % self.L)
                    self.plaq[1, x, y, z, 3] = (2, x, (y + 1) % self.L, z)

                    self.plaq[2, x, y, z, 0] = (0, x, y, z)
                    self.plaq[2, x, y, z, 1] = (1, x, y, z)
                    self.plaq[2, x, y, z, 2] = (0, x, (y + 1) % self.L, z)
                    self.plaq[2, x, y, z, 3] = (1, (x + 1) % self.L, y, z)

    def get_energy(self):
        bond_energy = 0
        for x in xrange(self.L):
            for y in xrange(self.L):
                for z in xrange(self.L):
                    for i in xrange(3):
                        prod = 1.0
                        for index in self.plaq[i, x, y, z]:
                            prod *= self.dual[index[0], index[1], index[2], index[3]]
                        bond_energy += prod
        spin_energy = -self.J * np.sum([self.dual[i] * np.cos((self.spins - np.roll(self.spins, 1, axis=i)) / 1.0) for i in xrange(3)])
        return -self.K * bond_energy + spin_energy

    def flip_energy_change(self, x, y, z, bond, oldangle, newangle):
        def p_energy(index_list):
            prod = 1.0
            for index in index_list:
                prod *= self.dual[index[0], index[1], index[2], index[3]]
            return prod
        def s_energy(x, y, z):
            central = self.spins[x, y, z]
            return -self.J * (math.cos((self.spins[(x - 1) % self.L, y, z] - central) / 1.0) * self.dual[0, (x - 1) % self.L, y, z] + \
                              math.cos((self.spins[(x + 1) % self.L, y, z] - central) / 1.0) * self.dual[0, x, y, z] + \
                              math.cos((self.spins[x, (y - 1) % self.L, z] - central) / 1.0) * self.dual[1, x, (y - 1) % self.L, z] + \
                              math.cos((self.spins[x, (y + 1) % self.L, z] - central) / 1.0) * self.dual[1, x, y, z] + \
                              math.cos((self.spins[x, y, (z - 1) % self.L] - central) / 1.0) * self.dual[2, x, y, (z - 1) % self.L] + \
                              math.cos((self.spins[x, y, (z + 1) % self.L] - central) / 1.0) * self.dual[2, x, y, z])
        
        if bond == 0:
            p_energy = -self.K * (p_energy(self.plaq[2, x, y, z, :]) +
                                  p_energy(self.plaq[2, x, (y - 1) % self.L, z, :]) +
                                  p_energy(self.plaq[0, x, y, z, :]) +
                                  p_energy(self.plaq[0, x, y, (z - 1) % self.L, :]))
        elif bond == 1:
            p_energy = -self.K * (p_energy(self.plaq[2, x, y, z, :]) +
                                  p_energy(self.plaq[2, (x - 1) % self.L, y, z, :]) +
                                  p_energy(self.plaq[1, x, y, z, :]) +
                                  p_energy(self.plaq[1, x, y, (z - 1) % self.L, :]))

        else:
            p_energy = -self.K * (p_energy(self.plaq[1, x, y, z, :]) +
                                  p_energy(self.plaq[1, x, (y - 1) % self.L, z, :]) +
                                  p_energy(self.plaq[0, x, y, z, :]) +
                                  p_energy(self.plaq[0, (x - 1) % self.L, y, z, :]))

        s_energy_init = s_energy(x, y, z)
        self.spins[x, y, z] = newangle
        self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
        s_energy_fin = s_energy(x, y, z)
        self.spins[x, y, z] = oldangle
        self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
        return (s_energy_fin - s_energy_init) - 2 * p_energy

    # I've updated this function to return the change in energy from the flip, since we
    # only need that for doing local updates anyway
    def constrain(self, alpha):
        x = alpha % (2 * math.pi)
        if x > 0:
            return x
        return constrain(x + 2 * math.pi)

    def flip(self):
        x = random.randint(0, self.L - 1)
        y = random.randint(0, self.L - 1)
        z = random.randint(0, self.L - 1)
        bond = random.randint(0, 2)
        # Also need to flip the spin about a random axis
        flip_axis = random.random() * math.pi * 2
        newangle = self.constrain(2.0 * flip_axis - self.spins[x, y, z])
        E = self.flip_energy_change(x, y, z, bond, self.spins[x, y, z], newangle)
        if random.random() < min(1, math.exp(-E)):
            self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
            self.spins[x, y, z] = newangle
            self.energy += E
            
    def poly_loop(self):
        return [np.average(np.prod(self.dual[0, :, :, :], axis=0)),
                np.average(np.prod(self.dual[1, :, :, :], axis=1)),
                np.average(np.prod(self.dual[2, :, :, :], axis=2))]

    def simulate(self):
        # This time, we want to compute expectations of polyakov loops with constant K and varying J
        # Note: if this test fails too badly, we should try setting the gauge field to 1 and reproducing
        # the 3D xy model result (remeber not to allow for gauge field flips!)
        nmc = 150
        ntherm = 200
        meas_step = 35
        loop_x = []
        loop_y = []
        loop_z = []

        Jstart = 0.1
        Jstop = 2.5
        deltaJ = 0.05

        data = []
        
        for J in np.arange(Jstart, Jstop, deltaJ):
            # When we update J, we should re-initializ the system to the uniform state
            self.J = J
            self.uniform_init()

            # Thermalize
            for i in xrange(self.L**3 * ntherm):
                self.flip()

            # Take measurements every 35 flips
            loop = []
            for i in xrange(nmc):
                print "MC Step number", i
                for j in xrange(self.L**3 * meas_step):
                    self.flip()
                # It's possible that I should be handling axes independently, but let's ignore that for now
                loop.append(np.average(self.poly_loop()))
            data.append(np.average(loop))

        plt.plot(np.arange(Jstart, Jstop, deltaJ), data, 'o', markersize=3)
        plt.xlabel("J")
        plt.ylabel("<P>")
        plt.show()
        # plt.plot(range(nmc), loop_y, 'o', markersize=3)
        # plt.xlabel("MC Time")
        # plt.ylabel("<Py>")
        # plt.show()
        # plt.plot(range(nmc), loop_z, 'o', markersize=3)
        # plt.xlabel("MC Time")
        # plt.ylabel("<Pz>")
        # plt.show()
    
test = IsingGauge3d(8, 0.1, 0.0, False)
test.simulate()
