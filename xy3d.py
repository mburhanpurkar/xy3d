#!/usr/bin/env python


import numpy as np
import math
import random
import matplotlib.pyplot as plt


def flip(state, J, L, phi):
    dE = 0
    x = random.randint(0, L - 1)
    y = random.randint(0, L - 1)
    z = random.randint(0, L - 1)
    newangle = random.gauss(state[x, y, z], 1) % (math.pi * 4)
    old_e = -J * (math.cos((state[x, y, z] - state[(x + 1) % L, y, z]) / phi) + math.cos((state[x, y, z] - state[x, (y + 1) % L, z]) / phi)
                  + math.cos((state[x, y, z] - state[x, y, (z + 1) % L]) / phi) + math.cos((state[x, y, z] - state[(x - 1) % L, y, z]) / phi)
                  + math.cos((state[x, y, z] - state[x, (y - 1) % L, z]) / phi) + math.cos((state[x, y, z] - state[x, y, (z - 1) % L]) / phi))
    new_e = -J * (math.cos((newangle - state[(x + 1) % L, y, z]) / phi) + math.cos((newangle - state[x, (y + 1) % L, z]) / phi)
                  + math.cos((newangle - state[x, y, (z + 1) % L]) / phi) + math.cos((newangle - state[(x - 1) % L, y, z]) / phi)
                  + math.cos((newangle - state[x, (y - 1) % L, z]) / phi) + math.cos((newangle - state[x, y, (z - 1) % L]) / phi))
    if new_e < old_e:
        p = 1.0
    else:
        p = math.exp(-(new_e - old_e))

    if random.random() < p:
        state[x, y, z] = newangle
        dE = new_e - old_e
    return x, y, z, dE

def mag(state):
    return math.sqrt(np.mean(np.cos(state))**2 + np.mean(np.sin(state))**2)

def init_ene(state, J, L, phi):
    return -J * np.sum([np.cos((state - np.roll(state, 1, axis=i)) / phi) for i in xrange(3)]) / L**3

def make_plots(phi):
    L = 10
    state = np.pi * 2 * np.random.rand(L, L, L)
    Jmin = 0.01
    Jmax = 1.5
    deltaJ = 0.05
    nmc = 500
    nmeas = 2
    ntherm = 500

    energy = init_ene(state, Jmin, L, phi)

    ene_out = np.zeros(len(np.arange(Jmin, Jmax, deltaJ)))
    sph_out = np.zeros(len(np.arange(Jmin, Jmax, deltaJ)))
    mag_out = np.zeros(len(np.arange(Jmin, Jmax, deltaJ)))
    sus_out = np.zeros(len(np.arange(Jmin, Jmax, deltaJ)))

    for idx, J in enumerate(np.arange(Jmin, Jmax, deltaJ)):
        print "J", J
        for i in xrange(ntherm * L**3):
            x, y, z, dE = flip(state, J, L, phi)
            energy += dE / L**3

        for i in xrange(nmc):
            if i % 100 == 0:
                print i
            for j in xrange(nmeas * L**3):
                x, y, z, dE = flip(state, J, L, phi)
                energy += dE / L**3

            ene_out[idx] += energy
            sph_out[idx] += energy**2
            mag_out[idx] += mag(state)
            sus_out[idx] += mag(state)**2
        print ene_out[idx] / nmc

    # Normalize, then output
    ene_out /= nmc
    sph_out /= nmc
    mag_out /= nmc
    sus_out /= nmc

    print np.arange(Jmin, Jmax, deltaJ)
    print ene_out
    print sph_out - ene_out**2
    print mag_out
    print sus_out - mag_out**2

    plt.plot(np.arange(Jmin, Jmax, deltaJ), ene_out, 'o')
    plt.xlabel("J")
    plt.ylabel("Energy")
    plt.show()
    plt.plot(np.arange(Jmin, Jmax, deltaJ), sph_out - ene_out**2, 'o')
    plt.xlabel("J")
    plt.ylabel("Specific Heat")
    plt.show()
    plt.plot(np.arange(Jmin, Jmax, deltaJ), mag_out, 'o')
    plt.xlabel("J")
    plt.ylabel("Magnetization")
    plt.show()
    plt.plot(np.arange(Jmin, Jmax, deltaJ), sus_out - mag_out**2, 'o')
    plt.xlabel("J")
    plt.ylabel("Magnetic Susceptibility")
    plt.show()

    
make_plots(2.0)
make_plots(1.0)

