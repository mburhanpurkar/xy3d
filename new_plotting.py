#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
np.set_printoptions(threshold=np.nan)


sizes = [4, 6, 8]
zoom = np.power(sizes,2.0)
labels = [ r"$L = {0}$".format(k) for k in sizes]
dataset = [np.loadtxt("mod_n" + str(k) + ".txt") for k in sizes]
np.append(zoom, 16.0)
np.append(zoom, 25.0)
#dataset.append(np.loadtxt("mod_n4_J_1.000000_K_0.100000.txt"))
#dataset.append(np.loadtxt("mod_n4_J_1.000000_K_0.001000.txt"))
dataset.append(np.loadtxt("mod_n4_J_1.000000_K_0.000000.txt"))
dataset.append(np.loadtxt("mod_n5_J_1.000000_K_0.000000.txt"))
dataset.append(np.loadtxt("mod_n6_J_1.000000_K_0.000000.txt"))
labels.append('mod4')
labels.append('mod5')
labels.append('mod6')

sizes = [4, 6, 8, 4, 5, 6]
zoom = np.power(sizes,2.0)
plt.figure(figsize=(12,8))
crit = 0#2.269185314

plt.subplot(221)

k = 0
for i in xrange(len(dataset)):
    plt.plot(dataset[i][:,0], dataset[i][:,1], ls="--", marker="o", label = labels[k], markersize=5)
    k += 1
#plt.axvline(x=crit, linestyle='--')
plt.legend(loc="upper right")
plt.xlabel(r"$T^*$")
plt.ylabel(r"$m$")
plt.title("Magnetization per spin")

plt.subplot(222)

#a = 24
#b = 48
a = 0
b = 60
step = 5

k = 0
for i in xrange(len(dataset)):
    print zoom[k]
    plt.plot(dataset[i][:,0], dataset[i][:,3] * zoom[k]**2, ls="--", marker="o", label = labels[k], markersize=5) 
    k += 1
#plt.axvline(x=crit, linestyle='--')
plt.legend(loc="upper right")
plt.xlabel(r"$T^*$")
plt.ylabel(r"$\chi$")
plt.title("Magnetic Susceptibility")

plt.subplot(223)

k = 0
for i in xrange(len(dataset)):
    plt.plot(dataset[i][:,0], dataset[i][:,2], ls="--", marker="o", label = labels[k], markersize=5)
    k += 1
plt.legend(loc="upper left")
plt.xlabel(r"$T^*$")
plt.ylabel(r"$E$")
#plt.axvline(x=crit, linestyle='--')
plt.title("Energy per spin")


plt.subplot(224)

k = 0
for i in xrange(len(dataset)):
    plt.plot(dataset[i][:,0], dataset[i][:,4] * zoom[k]**2, ls="--", marker="o", label = labels[k], markersize=5)
    k += 1
plt.legend(loc="upper right")
plt.xlabel(r"$T^*$")
plt.ylabel(r"$C_h$")
#plt.axvline(x=crit, linestyle='--')
plt.title("Specific Heat")

plt.tight_layout()
plt.savefig("observ.eps")
plt.show()

