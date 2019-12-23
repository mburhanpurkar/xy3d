#!/usr/bin/env python

import math
import random
import numpy as np
import matplotlib.pyplot as plt
import time


def make_loop(size):
    # offset shifts the start of the loop (theta = 0) by 'offset' number of spins
    x, y = 0, 0
    ll = [(x, y)]

    for plus_x in xrange(0, size - 1):
        x += 1
        ll.append((x, y))
    for plus_y in xrange(0, size - 1):
        y += 1
        ll.append((x, y))
    for minus_x in xrange(0, size - 1):
        x -= 1
        ll.append((x, y))
    for minus_y in xrange(0, size - 1):
        y -= 1
        ll.append((x, y))

    return ll


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
    def __init__(self, L, J, K, rand, plaq=None, subir_t=False, subir_nt=False, b=0):
        self.L = L
        self.K = K
        self.J = J

        # Check for proper usage
        assert not (subir_t and subir_nt)
        if subir_t or subir_nt:
            assert b != 0
        self.subir_t = subir_t
        self.subir_nt = subir_nt
        self.b = b

        # This will be handy later
        self.bond_boundary = np.full((3, self.L, self.L, self.L), True, dtype=bool)
        if self.subir_t or self.subir_nt:
            self.subir = True
            self.bond_boundary[0, :self.b] = False
            self.bond_boundary[0, -(self.b + 1):] = False
            self.bond_boundary[0, :, 0] = False
            self.bond_boundary[0, :, self.L - 1] = False
            self.bond_boundary[1, :, :self.b] = False
            self.bond_boundary[1, :, -(self.b + 1):] = False
            self.bond_boundary[1, 0] = False
            self.bond_boundary[1, self.L - 1] = False
            self.bond_boundary[2, :self.b] = False
            self.bond_boundary[2, -(self.b):] = False
            self.bond_boundary[2, :, :self.b] = False
            self.bond_boundary[2, :, -(self.b):] = False
        else:
            self.subir = False

        # Initialize the lattice
        if rand:
            self.random_init()
        else:
            self.uniform_init()

        if subir_t:
            self.subir_init()
        elif subir_nt:
            self.unif_boundary()

        if plaq is None:
            self.plaq = get_lattice_info(L)
        else:
            self.plaq = plaq

        # Get the system's energy excluding the boundary!
        self.energy = self.get_energy()

    def uniform_init(self):
        self.spins = np.zeros((self.L, self.L, self.L))
        self.dual = np.ones((3, self.L, self.L, self.L))

    def random_init(self):
        self.spins = np.random.rand(self.L, self.L, self.L) * 4 * np.pi
        self.dual = np.random.choice([-1, 1], (3, self.L, self.L, self.L))

    def subir_init(self):
        self.dual[1, :, self.L - 1, :] = 0
        self.dual[0, self.L - 1, :, :] = 0

        loop_indices = make_loop(self.L)
        spacing = 2 * math.pi / (4 * (self.L - 1))
        for i, coords in enumerate(loop_indices[:-1]):
            x, y = coords
            self.spins[x, y, :] = (i * spacing) % (2 * math.pi)

    def unif_boundary(self):
        self.dual[1, :, self.L - 1, :] = 0
        self.dual[0, self.L - 1, :, :] = 0

        # Again, the 3 here is arbitrary, as is the 1.1
        loop_indices = make_loop(self.L, 3)
        for x, y in loop_indices:
            self.spins[x, y, :] = 1.1

    def get_energy(self):
        # Compute the energy excluding the boundary terms
        bond_energy = 0
        for x in xrange(self.b, self.L - (self.b + 1)):  # need -1 here bc each plaq is "upwards" from the spin
            for y in xrange(self.b, self.L - (self.b + 1)):
                for z in xrange(self.L):
                    for i in xrange(3):
                        prod = 1.0
                        for index in self.plaq[i, x, y, z]:
                            prod *= self.dual[index[0], index[1], index[2], index[3]]
                        bond_energy += prod

        # For the spin energies, temporarily set some of the dual lattice to 0
        tmp = self.dual.copy()
        self.dual[np.where(self.bond_boundary == False)] = 0
        # self.visualize(2, axis=0)
        self.visualize(2, axis=1)
        self.visualize(2, axis=2)
        spin_energy = np.sum([self.dual[i] * np.cos((self.spins - np.roll(self.spins, 1, axis=i)) / 2.0) for i in xrange(3)])
        self.bond_energy = bond_energy
        self.dual = tmp
        return (-self.J * spin_energy - self.K * bond_energy) / 3.0 / self.L**3

    def count_bond(self, i, x, y, z):
        # decides whether to include flipped bond in energy calculation
        return self.bond_boundary[i, x, y, z]

    def count_spin(self, x, y, z):
        # decides whether to include flipped spin in energy calculation
        if x >= self.L - self.b or y >= self.L - self.b or x < self.b or y < self.b:
            return False
        return True

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

        # Flip and flip back the bond -- this seems to be the simplest way to do this
        s_energy_init = s_energy(x, y, z, self.spins[x, y, z])
        self.spins[x, y, z] = newangle
        if bond != -5:
            self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
        s_energy_fin = s_energy(x, y, z, self.spins[x, y, z])
        self.spins[x, y, z] = oldangle
        if bond != -5:
            self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]

        self.bond_energy_change = -2 * p_energy
        # Change in energy is the change in spin energy + twice the bond energy (since it just flops sign)
        return (s_energy_fin - s_energy_init) - 2 * p_energy

    def flip(self, i):
        if (i % 2 == 0):
            if self.subir:
                x = random.randint(1, self.L - 2)
                y = random.randint(1, self.L - 2)
            else:
                x = random.randint(0, self.L - 1)
                y = random.randint(0, self.L - 1)
            z = random.randint(0, self.L - 1)
            newangle = random.gauss(self.spins[x, y, z], 1) % (math.pi * 4)
            E = self.flip_energy_change(x, y, z, -5, self.spins[x, y, z], newangle)

            p = 1.0 if E < 0 else math.exp(-E)
            if random.random() <= p:
                self.spins[x, y, z] = newangle
                if self.count_spin(x, y, z) and self.subir:
                    assert (self.b == 1)
                    self.energy += E / (self.L**3 - 4 * self.L * (self.L - 1)) / 3.0
                elif not self.subir:
                    self.energy += E / self.L**3 / 3.0
        else:
            # Select whatver it wants!
            x = random.randint(0, self.L - 1)
            y = random.randint(0, self.L - 1)
            z = random.randint(0, self.L - 1)
            bond = random.randint(0, 2)
            E = self.flip_energy_change(x, y, z, bond, self.spins[x, y, z], self.spins[x, y, z])

            p = 1.0 if E < 0 else math.exp(-E)
            if random.random() < p:
                self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
                if self.count_bond(bond, x, y, z) and self.subir:
                    self.energy += E / (self.L**3 - 4 * self.L * (self.L - 1))
                    self.bond_energy += self.bond_energy_change / (self.L**3 - 4 * self.L * (self.L - 1)) / 3.0
                elif not self.subir:
                    self.energy += E / self.L**3
                    self.bond_energy += self.bond_energy_change / self.L**3 / 3.0

    def poly_loop(self):
        # Return [<px>, <py>, <pz>]
        return [np.average(np.prod(self.dual[0, :, :, :], axis=0)),
                np.average(np.prod(self.dual[1, :, :, :], axis=1)),
                np.average(np.prod(self.dual[2, :, :, :], axis=2))]

    def magnetization(self):
        if not self.subir:
            return math.sqrt(np.mean(np.cos(self.spins))**2 + np.mean(np.sin(self.spins))**2)
        return math.sqrt(np.mean(np.cos(self.spins[self.b:-self.b, self.b:-self.b]))**2 +
                         np.mean(np.sin(self.spins[self.b:-self.b, self.b:-self.b]))**2)

    def visualize(self, n, axis=0):
        # Plot the first n slices of the state
        for i in xrange(n):
            if axis == 0:
                spins = self.spins[:, :, i]
                gauge = self.dual[:, :, :, i]
            elif axis == 1:
                spins = self.spins[:, i]
                gauge = self.dual[:, :, i]
            else:
                spins = self.spins[i]
                gauge = self.dual[:, i]
            x = np.cos(spins)
            y = np.sin(spins)
            X, Y = np.mgrid[0:self.L, 0:self.L]
            plt.quiver(X, Y, x, y, spins, cmap='jet', pivot='mid')
            plt.axis('equal')

            for irow in xrange(self.L):
                for icol in xrange(self.L):
                    if axis == 0:
                        i = 0
                        j = 1
                    elif axis == 1:
                        i = 0
                        j = 2
                    else:
                        i = 1
                        j = 2
                    cx = 'b_' if gauge[i, irow, icol] == 1 else 'r_'
                    cy = 'b|' if gauge[j, irow, icol] == 1 else 'r|'
                    if gauge[i, irow, icol] == 0:
                        cx = 'y_'
                    if gauge[j, irow, icol] == 0:
                        cy = 'y|'
                    plt.plot(irow + 0.5, icol, cx)
                    plt.plot(irow, icol + 0.5, cy)
            plt.plot(irow + 0.5, icol, cx)
            plt.plot(irow, icol + 0.5, cy)
            plt.show()


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


def f(L, J, K, rand, plaq, ntherm, nmc, nmeas, subir_t=False, subir_nt=False, b=0):
    print "J", J, ", K", K
    test = FullModel(L, J, K, rand, plaq, subir_t, subir_nt, b)
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
        ene += test.energy
        ene2 += test.energy**2
        flux += test.bond_energy
        flux2 += test.bond_energy**2

    test.visualize(2)
    
    # After each round, let's make sure the dual lattice is all 1s
    mag /= nmc
    mag2 /= nmc
    ene /= nmc
    ene2 /= nmc
    flux /= nmc
    flux2 /= nmc
    return J, K, mag, (mag2 - mag**2), ene, (ene2 - ene**2), flux, (flux2 - flux**2)


def simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas, subir=False, b=0):
    plaq = get_lattice_info(L)
    if subir:
        if varyJ:
            x = [f(L, i, const, False, plaq, ntherm, nmc, nmeas, True, False, b) for i in np.arange(start, stop, delta)]
            y = [f(L, i, const, False, plaq, ntherm, nmc, nmeas, False, True, b) for i in np.arange(start, stop, delta)]
        else:
            x = [f(L, const, i, False, plaq, ntherm, nmc, nmeas, True, False, b) for i in np.arange(start, stop, delta)]
            y = [f(L, const, i, False, plaq, ntherm, nmc, nmeas, False, True, b) for i in np.arange(start, stop, delta)]
        print x
        print y

        print "\n\nBoundary\tJ\tK\tMagnetization\tSusceptibility\tEnergy\t\tSpecific Heat\tFlux\tFlux Susceptibility"
        for i in xrange(len(x)):
            print "Twist\t\t%.2f\t%.2f\t%.5f\t\t%.8f\t\t%.5f\t%f\t%f\t%f" % x[i]
            print "NTwist\t\t%.2f\t%.2f\t%.5f\t\t%.8f\t\t%.5f\t%f\t%f\t%f" % y[i]

        print
        print
        print "J\tK\tE_t - E_nt\t\tError"
        for i in xrange(len(x)):
            # Now we print out the error. For a simple sum, the error is given by sqrt{sigma_1^2 + sigma_2^2}
            print "%f\t%f\t%f\t\t%f" % (x[i][0], x[i][1], x[i][4] - y[i][4], math.sqrt(y[i][5] + x[i][5]))

    else:
        if varyJ:
            print [f(L, i, const, True, plaq, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]
        else:
            print [f(L, const, i, True, plaq, ntherm, nmc, nmeas) for i in np.arange(start, stop, delta)]


def visual_unit_test():
    def myplot(d, xlabel, ylabels, title):
        print "is tis running at all?"
        plt.figure(figsize=(16, 8))
        for j in xrange(6):
            plt.subplot(23 * 10 + j + 1)
            if xlabel == "J":
                plt.plot([i[0] for i in d], [i[j + 2] for i in d], 'o')
            else:
                print "This is running"
                plt.plot([i[1] for i in d], [i[j + 2] for i in d], 'o')
            plt.xlabel(xlabel)
            plt.ylabel(ylabels[j])
            plt.title(title)
        print "Saving plot..."
        plt.savefig(title + ".png")

    rand = True
    nmc = 10
    nmeas = 20
    ntherm = 10
    L = 10
    plaq = get_lattice_info(L)
    d1 = [f(L, J, 0, rand, plaq, ntherm, nmc, nmeas) for J in np.arange(0.5, 2.5, 1.0)]
    d2 = [f(L, 0.1, K, rand, plaq, ntherm, nmc, nmeas) for K in np.arange(0.1, 1.1, 0.5)]
    d3 = [f(L, J, 2.0, rand, plaq, ntherm, nmc, nmeas) for J in np.arange(0.05, 1.05, 0.05)]
    myplot(d1, "J", ["Magnetization", "Magnetic Susceptibility", "Energy",
                     "Specific Heat", "Flux", "Flux Susceptibility"], "K=0, L=10")
    myplot(d2, "K", ["Magnetization", "Magnetic Susceptibility", "Energy",
                     "Specific Heat", "Flux", "Flux Susceptibility"], "J=0.1, L=10")
    plot(d3, "J", ["Magnetization", "Magnetic Susceptibility", "Energy",
                   "Specific Heat", "Flux", "Flux Susceptibility"], "K=2.0, L=10")


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


# An initial test is to compare the energy ratio between the trivially ordered
# and disordered phases
L = 5
varyJ = False
ntherm = 1000
nmc = 750
nmeas = 30

# disorder (0.5, 0.1)
# fractionalized (1.1, 0.1)
# ordered (1.1, 1.0)

start = 0.1
stop = 1.2
delta = 1.0
const = 0.1
subir = True
b = 1
plaq = get_lattice_info(L)
# First, just check that the initialization and inital energy calculations are working correctly
# simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas, subir, 1)
# f(L, 0.1, 0.1, True, plaq, ntherm, nmc, nmeas, False, True, b)
# f(L, 0.1, 0.1, True, plaq, ntherm, nmc, nmeas, True, False, b)
# f(L, 0.4, 1.0, False, plaq, ntherm, nmc, nmeas, False, True, b)
f(L, 0.4, 1.0, False, plaq, ntherm, nmc, nmeas, True, False, b)
# f(L, 1.0, 1.0, False, plaq, ntherm, nmc, nmeas, False, True, b)
# f(L, 1.0, 1.0, False, plaq, ntherm, nmc, nmeas, True, False, b)
# simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas, subir, 1)

# visual_unit_test()

# Check the magnetizations in each phase
# print "getting the magnetizations in each phase..."
# plaq = get_lattice_info(L)
# print "ordered", f(L, 1.0, 1.0, True, plaq, ntherm, 5000, nmeas)[2]
# print "disordered", f(L, 0.1, 0.1, True, plaq, ntherm, nmc, nmeas)[2]
# print "fractionalized", f(L, 0.1, 2.0, True, plaq, ntherm, nmc, nmeas)[2]

# # Try computing the autocorrelation times for a few hundred MC steps
# L = 10
# J = 0.2
# K = 0.1
# rand = True
# plaq = get_lattice_info(L)
# ntherm = 500
# nmc = 10000
# x = energy_autocorrelation(L, J, K, rand, plaq, ntherm, nmc)
# x = x[: 200]
# print x
# plt.semilogy(range(nmc)[: 200], x, 'o')
# plt.xlabel("Number of MC Steps")
# plt.ylabel("Energy Autocorrelation")
# plt.title("J=" + str(J) + ", K=" + str(K) + ", L=" + str(L))
# plt.show()
