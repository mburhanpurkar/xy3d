#!/usr/bin/env python

import math
import random
import numpy as np
import time
import mpi_fanout
import sys


def make_loop(size, offset):
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

    a = offset % len(ll)
    return ll[-a:] + ll[:-a]


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

    plaq = np.empty((L, L, 4), dtype=(int, 3))
    for x in xrange(L):
        for y in xrange(L):
            plaq[x, y, 0] = (0, x, y)
            plaq[x, y, 1] = (1, x, y)
            plaq[x, y, 2] = (0, x, (y + 1) % L)
            plaq[x, y, 3] = (1, (x + 1) % L, y)
    return plaq


class FullModel():
    def __init__(self, L, J, K, rand, plaq=None, subir_t=False, subir_nt=False, b=-1):
        self.L = L
        self.K = K
        self.J = J
        self.nvis = 0

        # Check for proper usage
        assert not (subir_t and subir_nt)
        if subir_t or subir_nt:
            assert b != 0
        self.subir_t = subir_t
        self.subir_nt = subir_nt
        self.b = b

        # This will be handy later
        self.bond_boundary = np.full((2, self.L, self.L), True, dtype=bool)
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
        self.spins = np.zeros((self.L, self.L))
        self.dual = np.ones((2, self.L, self.L))

    def random_init(self):
        self.spins = np.random.rand(self.L, self.L) * 4 * np.pi
        self.dual = np.random.choice([-1, 1], (2, self.L, self.L))

    def subir_init(self):
        self.dual[1, :, self.L - 1] = 0
        self.dual[0, self.L - 1, :] = 0

        # Make the loop--the 3 here is arbitrary
        loop_indices = make_loop(self.L, 3)
        spacing = 2 * math.pi / (4.0 * (self.L - 1))
        for i, coords in enumerate(loop_indices[:-1]):
            x, y = coords
            self.spins[x, y, :] = (i * spacing) % (2 * math.pi)

    def unif_boundary(self):
        self.dual[1, :, self.L - 1] = 0
        self.dual[0, self.L - 1, :] = 0

        # Again, the 3 here is arbitrary, as is the 1.1
        loop_indices = make_loop(self.L, 3)
        for x, y in loop_indices:
            self.spins[x, y, :] = 1.1

    def get_energy(self):
        self.plaq_sgn = np.zeros((self.L, self.L))
        bond_energy = 0
        for x in xrange(self.b, self.L - (self.b + 1)):  # need -1 here bc each plaq is "upwards" from the spin
            for y in xrange(self.b, self.L - (self.b + 1)):
                prod = 1.0
                for index in self.plaq[x, y]:
                    prod *= self.dual[index[0], index[1], index[2]]
                self.plaq_sgn[x, y] = prod
                bond_energy += prod

        # For the spin energies, temporarily set some of the dual lattice to 0
        tmp = self.dual.copy()
        self.dual[np.where(self.bond_boundary == False)] = 0
        spin_energy = np.sum([self.dual[i] * np.cos((self.spins - np.roll(self.spins, 1, axis=i)) / 2.0) for i in xrange(2)])
        self.bond_energy = bond_energy
        self.dual = tmp
        return (-self.J * spin_energy - self.K * bond_energy) / self.L**3

    def count_bond(self, i, x, y):
        # decides whether to include flipped bond in energy calculation
        return self.bond_boundary[i, x, y]

    def count_spin(self, x, y):
        # decides whether to include flipped spin in energy calculation
        if x >= self.L - self.b or y >= self.L - self.b or x < self.b or y < self.b:
            return False
        return True

    def flip_energy_change(self, x, y, bond, oldangle, newangle):
        # Get the energy change associated with a spin flip at (x, y, z) and a bond flip at (bond, x, y, z)
        def s_energy(x, y, central):
            return -self.J * (math.cos((self.spins[(x - 1) % self.L, y] - central) / 2.0) * self.dual[0, (x - 1) % self.L, y] +
                              math.cos((self.spins[(x + 1) % self.L, y] - central) / 2.0) * self.dual[0, x, y] +
                              math.cos((self.spins[x, (y - 1) % self.L] - central) / 2.0) * self.dual[1, x, (y - 1) % self.L] +
                              math.cos((self.spins[x, (y + 1) % self.L] - central) / 2.0) * self.dual[1, x, y])

        if bond == 0:
            p_energy = -self.K * (self.plaq_sgn[x, y] + self.plaq_sgn[x, (y - 1) % self.L])
        elif bond == 1:
            p_energy = -self.K * (self.plaq_sgn[x, y]) + self.plaq_sgn[(x - 1) % self.L, y])
        else:
            p_energy = 0

        # Flip and flip back the bond -- this seems to be the simplest way to do this
        s_energy_init = s_energy(x, y, self.spins[x, y])
        self.spins[x, y] = newangle
        if bond != -5:
            self.dual[bond, x, y] = -self.dual[bond, x, y]
        s_energy_fin = s_energy(x, y, self.spins[x, y])
        self.spins[x, y] = oldangle
        if bond != -5:
            self.dual[bond, x, y] = -self.dual[bond, x, y]

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
            newangle = random.gauss(self.spins[x, y], 1) % (math.pi * 4)
            E = self.flip_energy_change(x, y, -5, self.spins[x, y], newangle)

            p = 1.0 if E < 0 else math.exp(-E)
            if random.random() <= p:
                self.spins[x, y] = newangle
                if self.count_spin(x, y) and self.subir:
                    assert (self.b == 1)
                    self.energy += E / (self.L**2 - 4 * self.L + 4)
                elif not self.subir:
                    self.energy += E / self.L**3
        else:
            # Here we're doing a bond flip, so we need to update self.plaq_sgn
            x = random.randint(0, self.L - 1)
            y = random.randint(0, self.L - 1)
            bond = random.randint(0, 1)
            E = self.flip_energy_change(x, y, bond, self.spins[x, y], self.spins[x, y])

            p = 1.0 if E < 0 else math.exp(-E)
            if random.random() < p:
                self.dual[bond, x, y] = -self.dual[bond, x, y]
                self.nvis += self.tmp_nvis
                if self.count_bond(bond, x, y) and self.subir:
                    self.energy += E / (self.L**2 - 4 * self.L * (self.L - 1))
                    self.bond_energy += self.bond_energy_change / (self.L**2 - 4 * self.L + 4)
                elif not self.subir:
                    self.energy += E / self.L**2
                    self.bond_energy += self.bond_energy_change / self.L**2

    def magnetization(self):
        if not self.subir:
            return math.sqrt(np.mean(np.cos(self.spins))**2 + np.mean(np.sin(self.spins))**2)
        return math.sqrt(np.mean(np.cos(self.spins[self.b:-self.b, self.b:-self.b]))**2 +
                         np.mean(np.sin(self.spins[self.b:-self.b, self.b:-self.b]))**2)


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
    nvis = 0
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
        nvis += test.nvis
        ene += test.energy
        ene2 += test.energy**2
        flux += test.bond_energy
        flux2 += test.bond_energy**2

    # After each round, let's make sure the dual lattice is all 1s
    nvis /= nmc
    ene /= nmc
    ene2 /= nmc
    flux /= nmc
    flux2 /= nmc
    return J, K, nvis, ene, (ene2 - ene**2), flux, (flux2 - flux**2)


def simulate_serial(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas, subir=False, b=0):
    plaq = get_lattice_info(L)
    if subir:
        if varyJ:
            x = [f(L, i, const, True, plaq, ntherm, nmc, nmeas, True, False, b) for i in np.arange(start, stop, delta)]
            y = [f(L, i, const, True, plaq, ntherm, nmc, nmeas, False, True, b) for i in np.arange(start, stop, delta)]
        else:
            x = [f(L, const, i, True, plaq, ntherm, nmc, nmeas, True, False, b) for i in np.arange(start, stop, delta)]
            y = [f(L, const, i, True, plaq, ntherm, nmc, nmeas, False, True, b) for i in np.arange(start, stop, delta)]
        print x
        print y

        print "\n\nBoundary\tJ\tK\VisonsEnergy\t\tSpecific Heat\tFlux\tFlux Susceptibility"
        for i in xrange(len(x)):
            print "Twist\t\t%.2f\t%.2f\t%.5f\t\t%.8f\t\t%.5f\t%f\t%f" % x[i]
            print "NTwist\t\t%.2f\t%.2f\t%.5f\t\t%.8f\t\t%.5f\t%f\t%f" % y[i]

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


def visual_unit_test_parallel(i):
    nmc = 1000
    nmeas = 20
    ntherm = 3000
    L = 10
    plaq = get_lattice_info(L)
    if i == 1:
        d1 = [mpi_fanout.task(f, L, J, 0, True, plaq, ntherm, nmc, nmeas) for J in np.arange(0.5, 2.5, 0.1)]
        print mpi_fanout.run_tasks(d1)
    elif i == 2:
        d2 = [mpi_fanout.task(f, L, 0.1, K, False, plaq, ntherm, nmc, nmeas) for K in np.arange(0.1, 1.1, 0.05)]
        print mpi_fanout.run_tasks(d2)
    else:
        d3 = [mpi_fanout.task(f, L, J, 1.0, rand, False, ntherm, nmc, nmeas) for J in np.arange(0.15, 0.65, 0.025)]
        print mpi_fanout.run_tasks(d3)


def simulate_parallel(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas, subir=False, b=0):
    plaq = get_lattice_info(L)
    if subir:
        if varyJ:
            x = [mpi_fanout.task(f, L, i, const, False, plaq, ntherm, nmc, nmeas, True, False, b) for i in np.arange(start, stop, delta)]
            y = [mpi_fanout.task(f, L, i, const, False, plaq, ntherm, nmc, nmeas, False, True, b) for i in np.arange(start, stop, delta)]
        else:
            x = [mpi_fanout.task(f, L, const, i, False, plaq, ntherm, nmc, nmeas, True, False, b) for i in np.arange(start, stop, delta)]
            y = [mpi_fanout.task(f, L, const, i, False, plaq, ntherm, nmc, nmeas, False, True, b) for i in np.arange(start, stop, delta)]
        a = mpi_fanout.run_tasks(x)
        b = mpi_fanout.run_tasks(y)
        print a
        print b

        print "\n\nBoundary\tJ\tK\tVisons\tEnergy\t\tSpecific Heat\tFlux\tFlux Susceptibility"
        for i in xrange(len(a)):
            print "Twist\t\t%.2f\t%.5f\t\t%.8f\t\t%.5f\t%f\t%f\t%f" % a[i]
            print "NTwist\t\t%.2f\t%.5f\t\t%.8f\t\t%.5f\t%f\t%f\t%f" % b[i]

        print "\n\nJ\tK\tE_t - E_nt\t\tError"
        for i in xrange(len(a)):
            print "%f\t%f\t%f\t\t%f" % (a[i][0], a[i][1], a[i][4] - b[i][4], math.sqrt(a[i][5] + b[i][5]))

        print "\n\nJ\tK\tE_t / E_nt\t\tError"
        for i in xrange(len(a)):
            print "%f\t%f\t%f\t\t%f" % (a[i][0], a[i][1], a[i][4] / b[i][4], math.sqrt(a[i][5] / b[i][4]**2 + b[i][5] * a[i][4]**2 / b[i][4]**4))

    else:
        if varyJ:
            x = [mpi_fanout.task(f, L, i, const, False, plaq, ntherm, nmc, nmeas, False, False, b) for i in np.arange(start, stop, delta)]
        else:
            x = [mpi_fanout.task(f, L, const, i, False, plaq, ntherm, nmc, nmeas, False, False, b) for i in np.arange(start, stop, delta)]
        a = mpi_fanout.run_tasks(x)

        print "\n\nJ\tK\tVisons\t\tEnergy\t\tSpecific Heat\tFlux\tFlux Susceptibility"
        for i in xrange(len(a)):
            print "%.2f\t%.2f\t\t%.8f\t\t%.5f\t%f\t%f\t%f" % a[i]

######################################################################


if __name__ == "__main__":
    assert len(sys.argv) == 2
    mpi_fanout.init()
    # visual_unit_test_parallel(int(sys.argv[1]))
    L = 20
    varyJ = False
    ntherm = 1000
    nmc = 5000
    nmeas = 30
    
    # # disorder (0.5, 0.1)
    # # fractionalized (1.1, 0.1)
    # # ordered (1.1, 1.0)

    start = 0.1
    stop = 2.0
    delta = 0.05
    const = 0.3
    subir = False
    simulate_parallel(L, varyJ, start, stop, delta, const, ntherm, nmc, nmeas, subir=subir, b=-1)
    mpi_fanout.exit()

