#!/usr/bin/env python

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import mpi_fanout
import time


def get_lattice_info(L):
    # This pre-computes the plaquettes and boundaries along which to compute the vorticity.
    # (Useful when running on the cluster with many instantiations of similar objects to avoid re-computation)

    # Each point on the lattice is associated with three plaquettes and three bond variables. The
    # array plaq tells us how to index self.dual. self.dual is indexed first by one of three directions
    # 0 -> sigma_x, 1 -> sigma_y, 2 -> sigma_z, and the remaining three indices are its associated spin's
    # location (x, y, z). So to index all the bond variables in the x direction on the lattice, we would do
    # self.dual[0, :, :, :]

    # Then the plaquettes are indexed similarly: the first index specifies the plane of the plaquette
    # 0 -> xz, 1 -> yz, 2 -> xy, the next three specify the associated spins's location (x, y, z) and
    # the last index isn't important, it just tells us which bond we are indexing from the specified plaquette

    plaq = np.empty((3, L, L, L, 4), dtype=(int, 4))
    for x in xrange(L):
        for y in xrange(L):
            for z in xrange(L):
                plaq[0, x, y, z, 0] = (0, x, y, z)
                plaq[0, x, y, z, 1] = (2, x, y, z)
                plaq[0, x, y, z, 2] = (0, x, y, (z + 1) % L)
                plaq[0, x, y, z, 3] = (2, (x + 1) % L, y, z)
                plaq[1, x, y, z, 0] = (1, x, y, z)
                plaq[1, x, y, z, 1] = (2, x, y, z)
                plaq[1, x, y, z, 2] = (1, x, y, (z + 1) % L)
                plaq[1, x, y, z, 3] = (2, x, (y + 1) % L, z)
                plaq[2, x, y, z, 0] = (0, x, y, z)
                plaq[2, x, y, z, 1] = (1, x, y, z)
                plaq[2, x, y, z, 2] = (0, x, (y + 1) % L, z)
                plaq[2, x, y, z, 3] = (1, (x + 1) % L, y, z)
    return plaq


class FullModel():
    def __init__(self, L, J, K, rand, plaq=None):
        self.L = L
        self.K = K
        self.J = J

        # Specify how to initialize the lattice
        if rand:
            self.random_init()
        else:
            self.uniform_init()

        if plaq is None:
            self.plaq = get_lattice_info(L)
        else:
            self.plaq = plaq

        # Get the system's energy
        self.energy = self.get_energy()

    def uniform_init(self):
        self.spins = np.zeros((self.L, self.L, self.L))
        self.dual = np.ones((3, self.L, self.L, self.L))

    def random_init(self):
        self.spins = np.random.rand(self.L, self.L, self.L) * 4 * np.pi
        self.dual = np.ones((3, self.L, self.L, self.L))
        # self.dual = np.random.choice([-1, 1], (3, self.L, self.L, self.L))

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
        spin_energy = np.sum([self.dual[i] * np.cos((self.spins - np.roll(self.spins, 1, axis=i)) / 2.0) for i in xrange(3)])
        self.bond_energy = bond_energy
        return -self.J * spin_energy / self.L**3 -self.K * bond_energy / self.L**3

    def flip_energy_change(self, x, y, z, bond, oldangle, newangle):
        # Get the energy change associated with a spin flip at (x, y, z) and a bond flip at (bond, x, y, z)
        def p_energy(index_list):
            prod = 1.0
            for index in index_list:
                prod *= self.dual[index[0], index[1], index[2], index[3]]
            return prod

        def s_energy(x, y, z, central):
            return -self.J * (math.cos((self.spins[(x - 1) % self.L, y, z] - central) / 2.0) * self.dual[0, (x - 1) % self.L, y, z] +
                              math.cos((self.spins[(x + 1) % self.L, y, z] - central) / 2.0) * self.dual[0, x, y, z] +
                              math.cos((self.spins[x, (y - 1) % self.L, z] - central) / 2.0) * self.dual[1, x, (y - 1) % self.L, z] +
                              math.cos((self.spins[x, (y + 1) % self.L, z] - central) / 2.0) * self.dual[1, x, y, z] +
                              math.cos((self.spins[x, y, (z - 1) % self.L] - central) / 2.0) * self.dual[2, x, y, (z - 1) % self.L] +
                              math.cos((self.spins[x, y, (z + 1) % self.L] - central) / 2.0) * self.dual[2, x, y, z])

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
        elif bond == 2:
            p_energy = -self.K * (p_energy(self.plaq[1, x, y, z, :]) +
                                  p_energy(self.plaq[1, x, (y - 1) % self.L, z, :]) +
                                  p_energy(self.plaq[0, x, y, z, :]) +
                                  p_energy(self.plaq[0, (x - 1) % self.L, y, z, :]))
        else:
            p_energy = 0

        # Two cases
        # (a) flip spin, bond == -5, just update senergy
        # (b) flip bond, update both senergy and penergy
        if bond == -5:
            return s_energy(x, y, z, newangle) - s_energy(x, y, z, oldangle)
        else:
            assert oldangle == newangle
            s_energy_init = s_energy(x, y, z, oldangle)
            self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
            s_energy_fin = s_energy(x, y, z, oldangle)  # oldangle and newangle are the same here
            self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
            self.bond_energy_change = -2 * p_energy
            return (s_energy_fin - s_energy_init) - 2 * p_energy

    def flip(self, i):
        if (i % 2 == 0):
            x = random.randint(0, self.L - 1)
            y = random.randint(0, self.L - 1)
            z = random.randint(0, self.L - 1)
            newangle = random.gauss(self.spins[x, y, z], 1) % (math.pi * 4)
            E = self.flip_energy_change(x, y, z, -5, self.spins[x, y, z], newangle)

            p = 1.0 if E < 0 else math.exp(-E)
            if random.random() <= p:
                self.spins[x, y, z] = newangle
                self.energy += E / self.L**3
        else:
            x = random.randint(0, self.L - 1)
            y = random.randint(0, self.L - 1)
            z = random.randint(0, self.L - 1)
            bond = random.randint(0, 2)
            E = self.flip_energy_change(x, y, z, bond, self.spins[x, y, z], self.spins[x, y, z])

            p = 1.0 if E < 0 else math.exp(-E)
            if random.random() < p:
                self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
                self.energy += E / self.L**3
                self.bond_energy += self.bond_energy_change

    def poly_loop(self):
        # Return [<px>, <py>, <pz>]
        return [np.average(np.prod(self.dual[0, :, :, :], axis=0)),
                np.average(np.prod(self.dual[1, :, :, :], axis=1)),
                np.average(np.prod(self.dual[2, :, :, :], axis=2))]

    def magnetization(self):
        return math.sqrt(np.mean(np.cos(self.spins))**2 + np.mean(np.sin(self.spins))**2)

    def visualize(self, n):
        # Plot the first n slices of the state
        for i in xrange(n):
            spins = self.spins[:, :, i]
            gauge = self.dual[:, :, :, i]

            x = np.cos(spins)
            y = np.sin(spins)
            X, Y = np.mgrid[0:self.L, 0:self.L]
            plt.quiver(X, Y, x, y, spins, pivot='mid')
            plt.axis('equal')

            for irow in xrange(self.L):
                for icol in xrange(self.L):
                    cx = 'b_' if gauge[0, irow, icol] == 1 else 'r_'
                    cy = 'b|' if gauge[1, irow, icol] == 1 else 'r|'
                    plt.plot(irow + 0.5, icol, cx)
                    plt.plot(irow, icol + 0.5, cy)
            plt.plot(irow + 0.5, icol, cx)
            plt.plot(irow, icol + 0.5, cy)
            plt.show()


######################################################################


def f(L, J, K, rand, plaq, ntherm, nmc, nmeas):
    print "J", J, ", K", K
    test = FullModel(L, J, K, rand, plaq)
    ene = 0
    ene2 = 0
    mag = 0
    mag2 = 0
    flux = 0
    flux2 = 0
    
    # Thermalize
    for i in xrange(test.L**3 * ntherm):
        test.flip(i)

    # Take measurements every 35 flips
    for i in xrange(nmc):
        if i % 100 == 0:
            print "nmc", i
        for j in xrange(test.L**3 * nmeas):
            test.flip(i)
        # loop.append(np.average(test.poly_loop()))
        magnetization = test.magnetization()
        mag += magnetization
        mag2 += magnetization**2
        ene += test.energy / nmeas
        ene2 += test.energy**2 / nmeas**2
        # flux += test.bond_energy
        # flux2 += test.bond_energy**2

    # After each round, let's make sure the dual lattice is all 1s
    mag /= nmc
    mag2 /= nmc
    ene /= nmc
    ene2 /= nmc
    flux /= nmc
    flux2 /= nmc
    return J, K, mag, (mag2 - mag**2), ene, (ene2 - ene**2), (flux2 - flux**2)


def simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas):
    plaq = get_lattice_info(L)
    if varyJ:
        x = [f(L, i, const, True, plaq, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
        print x
    else:
        x = [f(L, const, i, True, plaq, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
        print x


def simulate_parallel(L, vary_J, start, stop, delta, const, ntherm, nmc, nmeas):
    # Uses mpi_fanout.py to execute tasks in mpi_fanout.task in parallel
    plaq = get_lattice_info(L)
    if vary_J:
        print L, "K =", const, "J =", np.arange(start, stop, delta)
        task_list = [mpi_fanout.task(f, L, i, const, True, plaq, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
    else:
        print L, "J =", const, "K =", np.arange(start, stop, delta)
        task_list = [mpi_fanout.task(f, L, const, i, True, plaq, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]

    print mpi_fanout.run_tasks(task_list)


######################################################################


# Note--each J takes around 50 seconds to run on my laptop

# K = 0.1, J in [1.2, 1.9] (28)
L = 10
varyJ = True
ntherm = 500
nmc = 800
nmeas = 2

start = 1.2
stop = 1.9
delta = 0.025
const = 0.1
simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas)

# K = 0.5, J in [0.4, 0.9] (20)
L = 10
varyJ = True
ntherm = 500
nmc = 800
nmeas = 2

start = 0.4
stop = 0.9
delta = 0.025
const = 0.5
simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas)

# K = 0.8, J in [0.1, 0.8] (28)
L = 10
varyJ = True
ntherm = 500
nmc = 800
nmeas = 2

start = 0.1
stop = 0.8
delta = 0.025
const = 0.8
simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas)

# K = 1.1, J in [0.1, 0.8] (28)
L = 10
varyJ = True
ntherm = 500
nmc = 800
nmeas = 2

start = 0.1
stop = 0.8
delta = 0.025
const = 1.1
simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas)

# J = 0.1, K in [0.5, 1.0] (20)
L = 10
varyJ = False
ntherm = 500
nmc = 800
nmeas = 2

start = 0.5
stop = 1.0
delta = 0.025
const = 0.1
simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas)

# J = 0.35, K in [0.5, 1.0] (20)
L = 10
varyJ = False
ntherm = 500
nmc = 800
nmeas = 2

start = 0.5
stop = 1.0
delta = 0.025
const = 0.35
simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas)
