#!/usr/bin/env python


# Time to fix this plotting function!
# First, make sizes a commandline argument
# Make the prefix and argument too

import numpy as np
import matplotlib.pyplot as plt
import sys

def make_plot(sizes, pref):
    dataset = [np.loadtxt(pref + str(k) + ".txt") for k in sizes]
    zoom = np.power(sizes,2.0)
    labels = [ r"$L = {0}$".format(k) for k in sizes]

    plt.figure(figsize=(12,8))

    plt.subplot(221)

    for i in xrange(len(dataset)):
        plt.plot(dataset[i][:,0], dataset[i][:,1], ls="--", marker="o", label = labels[i], markersize=5)

    plt.legend(loc="upper right")
    plt.xlabel(r"$T^*$")
    plt.ylabel(r"$m$")
    plt.title("Magnetization per spin")

    plt.subplot(222)
    for i in xrange(len(dataset)):
        plt.plot(dataset[i][:,0], dataset[i][:,3] / zoom[i]**2, ls="--", marker="o", label = labels[i], markersize=5) 
    plt.legend(loc="upper right")
    plt.xlabel(r"$T^*$")
    plt.ylabel(r"$\chi$")
    plt.title("Magnetic Susceptibility")

    plt.subplot(223)
    for i in xrange(len(dataset)):
        plt.plot(dataset[i][:,0], dataset[i][:,2], ls="--", marker="o", label = labels[i], markersize=5)
    plt.legend(loc="upper left")
    plt.xlabel(r"$T^*$")
    plt.ylabel(r"$E$")
    plt.title("Energy per spin")

    plt.subplot(224)
    for i in xrange(len(dataset)):
        plt.plot(dataset[i][:,0], dataset[i][:,4] / zoom[i]**2, ls="--", marker="o", label = labels[i], markersize=5)
    plt.legend(loc="upper right")
    plt.xlabel(r"$T^*$")
    plt.ylabel(r"$C_h$")
    plt.title("Specific Heat")

    plt.tight_layout()
    plt.show()

    # Plot the vorticity
    plt.plot(dataset[i][:,0], dataset[i][:,6], ls="--", marker="o", label = labels[i], markersize=5)
    plt.title("Vorticity")
    plt.xlabel("J")
    plt.show()

if __name__ == "__main__":
    if sys.argv[1] == "test2d":
        sizes = [int(i) for i in sys.argv[2:]]
        pref = "testxy2d_n"
        make_plot(sizes, pref)
    elif sys.argv[1] == "test3d":
        sizes = [int(i) for i in sys.argv[2:]]
        pref = "testxy3d_n"
        make_plot(sizes, pref)
    else:
        make_plot([int(i) for i in sys.argv[2:]], sys.argv[1])

