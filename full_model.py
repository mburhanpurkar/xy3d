#!/usr/bin/env python

import math
import random
import numpy as np
# import mpi_fanout  # for cluster parallelization
import matplotlib.pyplot as plt


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

    # The 'boundary' is the same regardless of z (since we're only looking for vortices in the xy plane), so it
    # just contains 4 spin locations for each of the points in the xy plane
    plaq = np.empty((3, L, L, L, 4), dtype=(int, 4))
    boundary = np.empty((L, L, 5), dtype=(int, 2))

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

                boundary[x, y, 0] = (x, y)
                boundary[x, y, 1] = ((x + 1) % L, y)
                boundary[x, y, 2] = ((x + 1) % L, (y + 1) % L)
                boundary[x, y, 3] = (x, (y + 1) % L)
                boundary[x, y, 4] = (x, y)
    # For comparison to C++ code
    # for n in xrange(L * L * L):
    #     for i in xrange(3):
    #         for j in xrange(4):
    #             x = n / (L * L)
    #             y = (n - x * L * L) / L
    #             z = n - (x * L * L + y * L)
    #             print plaq[i, x, y, z, j][1] * L * L + plaq[i, x, y, z, j][2] * L + plaq[i, x, y, z, j][3], plaq[i, x, y, z, j][0]
    #     print "*****************"
    return plaq, boundary


class FullModel():
    def __init__(self, L, J, K, rand, plaq=None, boundary=None):
        self.L = L
        self.K = K
        self.J = J

        # Specify how to initialize the lattice
        if rand:
            self.random_init()
        else:
            self.uniform_init()

        # If plaq and boundary are not specified, compute them
        if plaq is None and boundary is None:
            self.plaq, self.boundary = get_lattice_info(L)
        else:
            self.plaq = plaq
            self.boundary = boundary

        # Get the system's energy
        self.energy = self.get_energy()
        print "INitial energy", self.energy
        self.m = np.zeros((2))

    # def energy_test(self):
    #     # Testing the energy of 2pi and 4pi vortices on 2x2x2 lattice
    #     assert self.L == 2
    #     self.dual[:, :, :, :] = np.ones((3, self.L, self.L, self.L))
    #     self.spins[0, 0, :] = 0.0
    #     self.spins[0, 1, :] = np.pi
    #     self.spins[1, 1, :] = 0.0
    #     self.spins[1, 0, :] = np.pi
    #     print "Energy of 4pi vortex", self.get_energy()

    #     self.spins[0, 0, :] = 0.0
    #     self.spins[0, 1, :] = np.pi / 2.0
    #     self.spins[1, 1, :] = np.pi
    #     self.spins[1, 0, :] = 3 * np.pi / 2.0
    #     print "Energy of 2pi vortex", self.get_energy()

    #     self.dual[0, 0, 0, :] = -1
    #     print "Energy of vison", self.get_energy()

    def uniform_init(self):
        self.spins = np.zeros((self.L, self.L, self.L))
        self.dual = np.ones((3, self.L, self.L, self.L))

    def random_init(self):
        self.spins = np.random.rand(self.L, self.L, self.L) * 2 * np.pi
        self.dual = np.random.choice([-1, 1], (3, self.L, self.L, self.L))

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
        spin_energy = -self.J * np.sum([self.dual[i] * np.cos((self.spins - np.roll(self.spins, 1, axis=i)) / 2.0) for i in xrange(3)])
        self.bond_energy = bond_energy
        return -self.K * bond_energy + spin_energy

    def flip_energy_change(self, x, y, z, bond, oldangle, newangle):
        # Get the energy change associated with a spin flip at (x, y, z) and a bond flip at (bond, x, y, z)
        def p_energy(index_list):
            prod = 1.0
            for index in index_list:
                prod *= self.dual[index[0], index[1], index[2], index[3]]
            return prod

        def s_energy(x, y, z):
            central = self.spins[x, y, z]
            return -self.J * (math.cos((self.spins[(x - 1) % self.L, y, z] - central) / 2.0) * self.dual[0, (x - 1) % self.L, y, z] + \
                              math.cos((self.spins[(x + 1) % self.L, y, z] - central) / 2.0) * self.dual[0, x, y, z] + \
                              math.cos((self.spins[x, (y - 1) % self.L, z] - central) / 2.0) * self.dual[1, x, (y - 1) % self.L, z] + \
                              math.cos((self.spins[x, (y + 1) % self.L, z] - central) / 2.0) * self.dual[1, x, y, z] + \
                              math.cos((self.spins[x, y, (z - 1) % self.L] - central) / 2.0) * self.dual[2, x, y, (z - 1) % self.L] + \
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
        else:
            p_energy = -self.K * (p_energy(self.plaq[1, x, y, z, :]) +
                                  p_energy(self.plaq[1, x, (y - 1) % self.L, z, :]) +
                                  p_energy(self.plaq[0, x, y, z, :]) +
                                  p_energy(self.plaq[0, (x - 1) % self.L, y, z, :]))

        # Flip and flip back the bond -- this seems to be the simplest way to do this
        s_energy_init = s_energy(x, y, z)
        self.spins[x, y, z] = newangle
        self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
        s_energy_fin = s_energy(x, y, z)
        self.spins[x, y, z] = oldangle
        self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]

        self.bond_energy_change = -2 * p_energy
        # Change in energy is the change in spin energy + twice the bond energy (since it just flops sign)
        return (s_energy_fin - s_energy_init) - 2 * p_energy

    def constrain(self, alpha):
        # Return alpha in [0, 2pi]
        return alpha % (2 * math.pi)

    def another_constrain(self, alpha):
        # Return alpha in [-pi, pi]
        if alpha < -np.pi:
            return self.another_constrain(alpha + 2 * np.pi)
        if alpha > np.pi:
            return self.another_constrain(alpha - 2 * np.pi)
        return alpha

    def flip(self, i):
        n = i % (self.L * self.L * self.L)
        x = n / (self.L * self.L)
        y = (n - x * self.L * self.L) / self.L
        z = n - (x * self.L * self.L + y * self.L)
        # x = random.randint(0, self.L - 1)
        # y = random.randint(0, self.L - 1)
        # z = random.randint(0, self.L - 1)
        bond = i % 3 # random.randint(0, 2)
        flip_axis = math.pi / 2.0 - 0.01# random.random() * math.pi
        newangle = self.constrain(2.0 * flip_axis - self.spins[x, y, z])
        E = self.flip_energy_change(x, y, z, bond, self.spins[x, y, z], newangle)

        print n, bond, self.spins[x, y, z], newangle, E

        p = 1.0 if E < 0 else math.exp(-E)
        #if random.random() < p:
        if (i % 2) < p:
            print "flip"
            self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
            self.spins[x, y, z] = newangle
            self.energy += E
            self.bond_energy += self.bond_energy_change

    def poly_loop(self):
        # Return [<px>, <py>, <pz>]
        return [np.average(np.prod(self.dual[0, :, :, :], axis=0)),
                np.average(np.prod(self.dual[1, :, :, :], axis=1)),
                np.average(np.prod(self.dual[2, :, :, :], axis=2))]

    def magnetization(self):
        return math.sqrt(np.mean(np.cos(self.spins))**2 + np.mean(np.sin(self.spins))**2)

    def vorticity(self, verbose=False):
        # Compute vorticity, returns list of vortex indices if verbose
        vort = np.zeros((4))
        if verbose:
            indices = []
        for z in xrange(self.L):
            coords = []
            for y in xrange(self.L):
                for x in xrange(self.L):
                    pi2 = True
                    delta = 0.0
                    i = 0
                    while i < len(self.boundary[x, y]) - 1:
                        x1, y1 = self.boundary[x, y, i + 1]
                        x2, y2 = self.boundary[x, y, i]
                        diff = self.another_constrain(self.spins[x1, y1, z] - self.spins[x2, y2, z])
                        delta += diff
                        test = abs(diff)

                        # Don't allow large angle changes for 2pi vortices (only relevant for 2pi since we're
                        # using elementary plaquettes to compute the vorticity)
                        if abs(test) > (np.pi / 1.5):
                            pi2 = False
                        i += 1

                    # Check if we have a vortex!
                    if abs(delta + 4 * np.pi) < 0.01:
                        vort[0] += 1
                        print "-4pi vortex at (" + str(x + 1.5) + ", " + str(y + 1.5) + ", " + str(z) + ")"
                    elif abs(delta - 4 * np.pi) < 0.01:
                        print "+4pi vortex at (" + str(x + 1.5) + ", " + str(y + 1.5) + ", " + str(z) + ")"
                        vort[3] += 1
                    if pi2:
                        if abs(delta + 2 * np.pi) < 0.01:
                            vort[1] += 1
                            coords.append(((x + 0.5) % self.L, (y + 0.5) % self.L))
                        elif abs(delta - 2 * np.pi) < 0.01:
                            vort[2] += 1
                            coords.append(((x + 0.5) % self.L, (y + 0.5) % self.L))

            if verbose:
                indices.append(coords)
        if verbose:
            return vort, indices
        return vort


######################################################################


def f(L, J, K, rand, plaq, boundary, ntherm, nmc, nmeas):
    test = FullModel(L, J, K, rand, plaq, boundary)
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
        for j in xrange(test.L**3 * nmeas):
            test.flip(j)
        # loop.append(np.average(test.poly_loop()))
        magnetization = test.magnetization()
        mag += magnetization
        mag2 += magnetization**2
        ene += test.energy
        ene2 += test.energy**2
        flux += test.bond_energy
        flux2 += test.bond_energy**2

    mag /= nmc
    mag2 /= nmc
    ene /= nmc
    ene2 /= nmc
    flux /= nmc
    flux2 /= nmc
    return J, K, mag, (mag2 - mag**2), ene, (ene2 - ene**2), (flux2 - flux**2)

def simulate_parallel(L, vary_J, const, start, stop, delta, ntherm, nmc, nmeas):
    # Uses mpi_fanout.py to execute tasks in mpi_fanout.task in parallel
    plaq, boundary = get_lattice_info(L)
    if vary_J:
        print L, "K =", const, "J =", np.arange(start, stop, delta)
        task_list = [mpi_fanout.task(f, L, i, const, True, plaq, boundary, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
    else:
        print L, "J =", const, "K =", np.arange(start, stop, delta)
        task_list = [mpi_fanout.task(f, L, const, i, True, plaq, boundary, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]

    print mpi_fanout.run_tasks(task_list)

# def vortex_serial(L, Kstart, Kstop, deltaK, J, ntherm, nmc, nmeas):
#     # Serial vorticity computatiopn
#     plaq, boundary = get_lattice_info(L)
#     for i in np.arange(Kstart, Kstop, deltaK):
#         print "K =", i
#         f(L, J, i, True, plaq, boundary, ntherm, nmc, nmeas)

def simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas):
    plaq, boundary = get_lattice_info(L)
    if varyJ:
        x = [f(L, i, const, False, plaq, boundary, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
        print x
        plt.plot(np.arange(start, stop, delta), [y[6] for y in x])
        plt.xlabel("J")
        plt.ylabel("Flux Susceptibility")
        plt.show()
    else:
        x = [f(L, const, i, False, plaq, boundary, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
        print x
        plt.plot(np.arange(start, stop, delta), [y[6] for y in x])
        plt.xlabel("K")
        plt.ylabel("Flux Susceptibility")
        plt.show()

def plot(x, y, xlabel, ylabel):
    plt.plot(np.arange(Kstart, Kstop, deltaK), sph, 'o', markersize=3)
    plt.xlabel("K")
    plt.ylabel("Flux Susceptibility")
    plt.show()

# def visual_test(L, J, K):
#     # Visual test! Plots xy-plane slices of LxLxL lattice with bond variables + spins + vortices marked!
#     test = FullModel(L, J, K, True)

#     # Thermalize
#     for i in xrange(test.L**3 * 500):
#         test.flip()

#     # Now we're ready to plot!
#     vort, coords = test.vorticity(verbose=True)

#     print "Vortex counts"
#     print "-4pi", vort[0]
#     print "-2pi", vort[1]
#     print "+2pi", vort[2]
#     print "+4pi", vort[3]

#     # Now plot everything: we will go row by row
#     for z in xrange(L):
#         x_spins = np.cos(test.spins[:, :, z])
#         y_spins = np.sin(test.spins[:, :, z])
#         X, Y = np.mgrid[0:L, 0:L]

#         # Plot the spins
#         plt.quiver(X, Y, x_spins, y_spins, test.spins[:, :, z], pivot='mid', cmap=plt.cm.hsv, clim=[0, 2 * 3.15])
#         plt.axis('equal')

#         # Plot the bond variables
#         for x in xrange(L):
#             for y in xrange(L):
#                 # First sigma_x
#                 color = "g" if (test.dual[0, x, y, z] == 1) else "r"
#                 plt.plot(x + 0.5, y, linestyle='None', marker='_', color=color)

#                 # Then sigma_y
#                 color = "g" if (test.dual[1, x, y, z] == 1) else "r"
#                 plt.plot(x, y + 0.5, linestyle='None', marker='|', color=color)

#         # Plot the vortices (and antivortices--both are in the same colour)
#         plt.plot(*zip(*coords[z]), linestyle='None', marker='o', color='b', markersize=5)
#         plt.show()

######################################################################


# L = 10
# Jstart = 1.0
# Jstop = 2.5
# deltaJ = 0.05
# ntherm = 200
# nmc = 500
# nmeas = 30
# K = 1e-1

# For parallel computation
# mpi_fanout.init()
# simulate_parallel(L, True, K, Jstart, Jstop, deltaJ, ntherm, nmc, nmeas)
# mpi_fanout.exit()

# Uncomment to re-produce plots from the 2001 paper
L = 4
varyJ = False
start = 1.2
stop = 1.25
delta = 1
const = 0.0
ntherm = 200
nmc = 500
nmeas = 30
simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas)

# This isn't too useful at the moment--uncomment for outputting (J, K, nvort)
# vortex_serial(L, Kstart, Kstop, deltaK, J, ntherm, nmc, nmeas)

# Runs the visual test J=0.1 and K=0.05 seemed somewhat promising (note: this will make
# 10 plots pop up, which is somewhat annoying)
# visual_test(10, 0.1, 0.05)

# This runs an energy test--this might be buggy; hasn't been properly tested!
# test = FullModel(2, 1.0, 1.0, False)
# test.energy_test()
