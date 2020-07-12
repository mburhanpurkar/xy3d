#!/usr/bin/env python

# This is a direct modification of the code full_model_simplified.py
# Here, we merely remove the energy associated with the spins! 


import math
import random
import numpy as np
import time
import mpi_fanout
import sys


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
    def __init__(self, L, K, rand, plaq=None):
        self.L = L
        self.K = K

        if rand:
            self.random_init()
        else:
            self.uniform_init()

        if plaq is None:
            self.plaq = get_lattice_info(L)
        else:
            self.plaq = plaq

        self.energy = self.get_energy()

    def uniform_init(self):
        self.spins = np.zeros((self.L, self.L, self.L))
        self.dual = np.ones((3, self.L, self.L, self.L))

    def random_init(self):
        self.spins = np.random.rand(self.L, self.L, self.L) * 4 * np.pi
        self.dual = np.random.choice([-1, 1], (3, self.L, self.L, self.L))

    def get_energy(self):
        bond_energy = 0
        self.plaq_sgn = np.ones((3, self.L, self.L, self.L))
        for x in xrange(self.L): 
            for y in xrange(self.L):
                for z in xrange(self.L):
                    for i in xrange(3):
                        prod = 1.0
                        for index in self.plaq[i, x, y, z]:
                            prod *= self.dual[index[0], index[1], index[2], index[3]]
                            self.plaq_sgn[i, x, y, z] = -1
                        bond_energy += prod
        return -self.K * bond_energy / self.L**3

    def flip_energy_change(self, x, y, z, bond):
        if bond == 0:
            p_energy = -self.K * (self.plaq_sgn[2, x, y, z] + self.plaq_sgn[2, x, (y - 1) % self.L, z] +
                                  self.plaq_sgn[0, x, y, z] + self.plaq_sgn[0, x, y, (z - 1) % self.L])
        elif bond == 1:
            p_energy = -self.K * (self.plaq_sgn[2, x, y, z] + self.plaq_sgn[2, (x - 1) % self.L, y, z] +
                                  self.plaq_sgn[1, x, y, z] + self.plaq_sgn[1, x, y, (z - 1) % self.L])
        else:
            p_energy = -self.K * (self.plaq_sgn[1, x, y, z] + self.plaq_sgn[1, x, (y - 1) % self.L, z] +
                                  self.plaq_sgn[0, x, y, z] + self.plaq_sgn[0, (x - 1) % self.L, y, z])
        return -2 * p_energy

    def flip(self, i):
        x = random.randint(0, self.L - 1)
        y = random.randint(0, self.L - 1)
        z = random.randint(0, self.L - 1)
        bond = random.randint(0, 2)
        E = self.flip_energy_change(x, y, z, bond)
        
        p = 1.0 if E < 0 else math.exp(-E)
        if random.random() < p:
            self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
            if bond == 0:
                self.plaq_sgn[2, x, y, z] *= -1
                self.plaq_sgn[2, x, (y - 1) % self.L, z] *= -1
                self.plaq_sgn[0, x, y, z] *= -1
                self.plaq_sgn[0, x, y, (z - 1) % self.L] *= -1
            elif bond == 1:
                self.plaq_sgn[2, x, y, z] *= -1
                self.plaq_sgn[2, (x - 1) % self.L, y, z] *= -1
                self.plaq_sgn[1, x, y, z] *= -1
                self.plaq_sgn[1, x, y, (z - 1) % self.L] *= -1
            else:
                self.plaq_sgn[1, x, y, z] *= -1
                self.plaq_sgn[1, x, (y - 1) % self.L, z] *= -1
                self.plaq_sgn[0, x, y, z] *= -1
                self.plaq_sgn[0, (x - 1) % self.L, y, z] *= -1
            
            self.energy += E / self.L**3

    def poly_loop(self):
        # Return [<px>, <py>, <pz>]
        return [np.average(np.prod(self.dual[0, :, :, :], axis=0)),
                np.average(np.prod(self.dual[1, :, :, :], axis=1)),
                np.average(np.prod(self.dual[2, :, :, :], axis=2))]

    def magnetization(self):
        return 0


######################################################################


def energy_autocorrelation(L, J, K, rand, plaq, ntherm, nmc):
    # It might be intereting to see whether autocorrelation times vary between
    # ntherm = 0 and ntherm != 0--I suspect they will be much shorter in the
    # second case!!
    test = FullModel(L, J, K, rand, plaq)
    ene_arr = np.empty(nmc)
    ene = 0
    ene2 = 0

    for i in xrange(ntherm * test.L**3):
        test.flip(i)

    for i in xrange(nmc):
        for j in xrange(test.L**3):
            test.flip(j)
        ene_arr[i] = test.energy

    ene_arr2 = ene_arr**2

    # Compute the autocorrelations
    varE = np.mean(ene_arr2) - (np.mean(ene_arr))**2
    ac = np.empty(nmc)
    for i in xrange(nmc):
        shifted_ene = np.roll(ene_arr, i)
        ac[i] = np.mean(shifted_ene * ene_arr)
    return (ac - (np.mean(ene_arr))**2) / varE


def f(L, K, rand, plaq, ntherm, nmc, nmeas):
    print "K", K
    test = FullModel(L, K, rand, plaq)
    ene = 0
    ene2 = 0
    nvis = 0
    flux = 0
    flux2 = 0

    # Thermalize
    for i in xrange(test.L**3 * ntherm):
        test.flip(i)

    # Take measurements every nmeas flips
    for i in xrange(nmc):
        if i % 100 == 0:
            print "nmc", i
        for j in xrange(test.L**3 * nmeas):
            test.flip(i)
        # loop.append(np.average(test.poly_loop()))
        nvis += np.sum(test.plaq_sgn == -1)
        ene += test.energy
        ene2 += test.energy**2

    # After each round, let's make sure the dual lattice is all 1s
    nvis /= nmc
    ene /= nmc
    ene2 /= nmc
    return K, nvis, ene, (ene2 - ene**2)


def simulate_serial(L, start, stop, delta, ntherm, nmc, nmeas):
    plaq = get_lattice_info(L)
    print [f(L, i, False, plaq, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]


def simulate_parallel(L, start, stop, delta, ntherm, nmc, nmeas):
    plaq = get_lattice_info(L)
    x = [mpi_fanout.task(f, L, i, False, plaq, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
    a = mpi_fanout.run_tasks(x)
    print "\n\nK\t\tVisons\t\tEnergy\t\tSpecific Heat"
    for i in xrange(len(a)):
        print "%f\t\t%f\t\t%f\t\t%f" % a[i]


######################################################################


if __name__ == "__main__":
    mpi_fanout.init()
    L = 10
    ntherm = 1000
    nmc = 5000
    nmeas = 30
    
    # # disorder (0.5, 0.1)
    # # fractionalized (1.1, 0.1)
    # # ordered (1.1, 1.0)

    start = 0.1
    stop = 1.2
    delta = 0.05
    simulate_parallel(L, start, stop, delta, ntherm, nmc, nmeas)
    mpi_fanout.exit()

