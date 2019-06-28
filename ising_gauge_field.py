#!/usr/bin/env python

import math
import random
import numpy as np
import mpi_fanout
# import matplotlib.pyplot as plt


def get_lattice_info(L):
    # This function returns plquettes and vortex boundaries for a system of size L
    # The first index tells us, 0 -- side, 1 -- back, 2 -- top
    # Then the second, third, and fourth give us (x, y, z)
    # The final index gives us one bond of four in the plaquette! (i.e. how to index the dual lattice)
    plaq = np.empty((3, L, L, L, 4), dtype=(int, 4))
    boundary = np.empty((L, L, 13), dtype=(int, 2))
    # boundary = np.empty((L, L, 5), dtype=(int, 2))
    
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
                boundary[x, y, 2] = ((x + 2) % L, y)
                boundary[x, y, 3] = ((x + 3) % L, y)
                boundary[x, y, 4] = ((x + 3) % L, (y + 1) % L)
                boundary[x, y, 5] = ((x + 3) % L, (y + 2) % L)
                boundary[x, y, 6] = ((x + 3) % L, (y + 3) % L)
                boundary[x, y, 7] = ((x + 2) % L, (y + 3) % L)
                boundary[x, y, 8] = ((x + 1) % L, (y + 3) % L)
                boundary[x, y, 9] = (x, (y + 3) % L)
                boundary[x, y, 10] = (x, (y + 2) % L)
                boundary[x, y, 11] = (x, (y + 1) % L)
                boundary[x, y, 12] = (x, y)

                # boundary[x, y, 0] = (x, y)
                # boundary[x, y, 1] = ((x + 1) % L, y)
                # boundary[x, y, 2] = ((x + 1) % L, (y + 1) % L)
                # boundary[x, y, 3] = (x, (y + 1) % L)
                # boundary[x, y, 4] = (x, y)
    return plaq, boundary


class IsingGauge3d():
    def __init__(self, L, J, K, rand, plaq=None, boundary=None):
        self.L = L
        self.K = K
        self.J = J
        if rand:
            self.random_init()
        else:
            self.uniform_init()
        if plaq is None and boundary is None:
            self.plaq, self.boundary = get_lattice_info(L)
        else:
            self.plaq = plaq
            self.boundary = boundary
        self.energy = self.get_energy()

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
        return -self.K * bond_energy + spin_energy

    def flip_energy_change(self, x, y, z, bond, oldangle, newangle):
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

        s_energy_init = s_energy(x, y, z)
        self.spins[x, y, z] = newangle
        self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
        s_energy_fin = s_energy(x, y, z)
        self.spins[x, y, z] = oldangle
        self.dual[bond, x, y, z] = -self.dual[bond, x, y, z]
        return (s_energy_fin - s_energy_init) - 2 * p_energy

    def constrain(self, alpha):
        x = alpha % (2 * math.pi)
        if x > 0:
            return x
        return constrain(x + 2 * math.pi)

    def another_constrain(self, alpha):
        if alpha < -np.pi:
            return self.another_constrain(alpha + 2 * np.pi)
        if alpha > np.pi:
            return self.another_constrain(alpha - 2 * np.pi)
        return alpha

    def flip(self):
        x = random.randint(0, self.L - 1)
        y = random.randint(0, self.L - 1)
        z = random.randint(0, self.L - 1)
        bond = random.randint(0, 2)
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

    def vorticity(self):
        # We have already verified that the number of vortices and antivortices are equal, so no need to
        # measure all four quantities!
        vort = np.zeros((2))
        for x in xrange(self.L):
            for y in xrange(self.L):
                for z in xrange(self.L):
                    delta = 0.0
                    i = 0
                    while i < len(self.boundary[x, y]) - 1:
                        x1, y1 = self.boundary[x, y, i + 1]
                        x2, y2 = self.boundary[x, y, i]
                        diff = self.another_constrain(self.spins[x1, y1, z] - self.spins[x2, y2, z])
                        delta += diff
                        test = abs(diff)
                        
                        if abs(test) > (np.pi / 2):
                            delta = 0.0
                            i = len(self.boundary[x, y])
                        i += 1
                    if abs(delta + 4 * np.pi) < 0.1:
                        vort[0] += 1
                    elif abs(delta + 2 * np.pi) < 0.1:
                        vort[1] += 1
        return vort


######################################################################


def f(L, J, K, rand, plaq, boundary, ntherm, nmc):
    vort = np.zeros((2))
    test = IsingGauge3d(L, J, K, rand, plaq, boundary)

    for i in xrange(L**3 * ntherm):
        test.flip()

    # Do nmc measurements and take a measurement every  MC steps
    for i in xrange(nmc):
        for j in xrange(L**3):
            test.flip()
        # Do a measurement of the vorticity!
        vort += test.vorticity()
        
    # Print out averaged (and size-normalized) quantities
    vort /= (nmc * L**3)

    print L, J, K, vort[0], vort[1]


def simulate_parallel(L, J, Kstart, Kstop, deltaK, ntherm, nmc):
    plaq, boundary = get_lattice_info(L)
    task_list = [mpi_fanout.task(f, L, J, i, True, plaq, boundary, ntherm, nmc) for i in np.arange(Kstart, Kstop, deltaK)]
    mpi_fanout.run_tasks(task_list)

def vortex_serial(L, Kstart, Kstop, deltaK, J, ntherm, nmc):
    plaq, boundary = get_lattice_info(L)
    for i in np.arange(Kstart, Kstop, deltaK):
        print "K =", i
        f(L, J, i, True, plaq, boundary, ntherm, nmc)
    
def simulate_serial(L, J, Kstart, Kstop, deltaK, ntherm, nmc):
    plaq, boundary = get_lattice_info(L)
    loop_x = []
    loop_y = []
    loop_z = []
    data = []
    
    # The only arguments that matter here are L, plaq, and boundary since everything else gets reset anyway
    test = IsingGauge3d(8, J, Kstart, True, plaq, boundary)  
    
    for K in np.arange(Kstart, Kstop, deltaK):
        print "K =", K
        # When we update J, we should re-initializ the system to the uniform state
        test.K = K
        test.uniform_init()

        # Thermalize
        for i in xrange(test.L**3 * ntherm):
            test.flip()

        # Take measurements every 35 flips
        loop = []
        for i in xrange(nmc):
            for j in xrange(test.L**3 * nmeas):
                test.flip()
            loop.append(np.average(test.poly_loop()))
        data.append(np.average(loop))

    plt.plot(np.arange(Kstart, Kstop, deltaK), data, 'o', markersize=3)
    plt.xlabel("K")
    plt.ylabel("<P>")
    plt.show()


######################################################################


L = 20
Kstart = 1.0
Kstop = 5.0
deltaK = 0.5
ntherm = 500
nmc = 500
J = 0.3

mpi_fanout.init()
simulate_parallel(L, J, Kstart, Kstop, deltaK, ntherm, nmc)
mpi_fanout.exit()

# simulate_serial(L, J, Kstart, Kstop, deltaK, ntherm, nmc)
# vortex_serial(L, Kstart, Kstop, deltaK, J, ntherm, nmc)


