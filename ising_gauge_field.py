#!/usr/bin/env python

import math
import numpy as np


class IsingGauge2d():
    def __init__(self, L, K):
        self.L = L
        self.K = K
        self.spins = np.random.rand(self.L, self.L) * np.pi
        self.dual = np.random.choice([-1, 1], (self.L, self.L, 2)) # the 2 here corresponds to top and left
        self.get_plaqs()
        self.energy = self.get_energy()

    def get_plaqs(self):
        self.plaqs = np.empty((self.L, self.L, 4), dtype=(int,3))
        for x in xrange(self.L):
            for y in xrange(self.L):
                self.plaqs[x, y, 0] = (x, y, 0)
                self.plaqs[x, y, 1] = (x, y, 1)
                self.plaqs[x, y, 2] = ((x + 1) % self.L, y, 0)
                self.plaqs[x, y, 3] = (x, (y + 1) % self.L, 1)

    def get_energy(self):
        energy = 0
        for x in xrange(self.L):
            for y in xrange(self.L):
                prod = 1.0
                for index in self.plaqs[x, y]:
                    prod *= self.dual[index[0], index[1], index[2]]
                energy += prod
        return energy

    def dual_flip_energy(self, x, y, bond, central):
        def p_energy(x, y):
            prod = central
            for index in self.plaqs[x, y][1:]:
                prod *= self.dual[index[0], index[1], index[2]]
            return prod
        if bond == 0:
            return -self.K * (p_energy(x, y) + p_energy((x - 1) % self.L, y))
        return -self.K * (p_energy(x, y) + p_energy(x, (y - 1) % self.L))

    def flip(self):
        x = math.random.randint(0, self.L - 1)
        y = math.random.randint(0, self.L - 1)
        bond = math.random.randint(0, 1)

        E1 = dual_flip_energy(x, y, bond, self.dual[x, y, bond])
        E2 = dual_flip_energy(x, y, bond, -self.dual[x, y, bond])

        if math.rand() < min(1, math.exp(-(E2 - E1))):
            self.dual[x, y, bond] = -self.dual[x, y, bond]
            self.energy += E2 - E1
            
    def poly_loop(self):
        # Returns px, py
        return np.prod(dual[:, :, 1], axis=1), np.prod(dual[:, :, 0], axis=0)

    def simulate(self):
        


        
    
test = IsingGauge2d(2, 1)
test.simulate()
