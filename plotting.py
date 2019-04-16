#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sco
np.set_printoptions(threshold=np.nan)


sizes = [8, 16]
zoom = np.power(sizes,2.0)
#labels = [ r"$L = {0}$".format(k) for k in sizes]
#dataset = [np.loadtxt("n" + str(k) + ".txt") for k in sizes]
labels = ['old', 'new']
dataset = [np.loadtxt("n8_nice.txt"), np.loadtxt("n8.txt")]

sizes = [8, 8]
zoom = np.power(sizes,2.0)
labels.append('merp')               
plt.figure(figsize=(12,8))
crit = 2.269185314

plt.subplot(221)

k = 0
for i in xrange(len(dataset)):
    if i == 1:
        plt.plot(dataset[i][:,0], dataset[i][:,1], ls="--", marker="o", label = labels[k], markersize=5) 
    else:
        plt.plot(dataset[i][:,0], dataset[i][:,2], ls="--", marker="o", label = labels[k], markersize=5)
    print dataset[i][:, 2]
    # plt.errorbar(dataset[i][:,0], dataset[i][:,2], yerr=dataset[i][:,3], fmt="none")
    k += 1
plt.axvline(x=crit, linestyle='--')
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
    if i == 1:
        plt.plot(dataset[i][:,0], dataset[i][:,3] * zoom[k]**2, ls="--", marker="o", label = labels[k], markersize=5) 
    else:
        plt.plot(dataset[i][a:b,0], dataset[i][a:b,4]* zoom[k]**2, ls="--", marker="o", label = labels[k], markersize=5)
    # plt.errorbar(dataset[i][a:b:step,0], dataset[i][a:b:step,4]* zoom[k], yerr=dataset[i][a:b:step,5]* zoom[k], fmt="none")
    k += 1
plt.axvline(x=crit, linestyle='--')
plt.legend(loc="upper right")
plt.xlabel(r"$T^*$")
plt.ylabel(r"$\chi$")
plt.title("Magnetic Susceptibility")

plt.subplot(223)

k = 0
for i in xrange(len(dataset)):
    if i == 1:
        plt.plot(dataset[i][:,0], dataset[i][:,2], ls="--", marker="o", label = labels[k], markersize=5) 
    else:
        plt.plot(dataset[i][:,0], dataset[i][:,6], ls="--", marker="o", label = labels[k], markersize=5)
    # plt.errorbar(dataset[i][:,0], dataset[i][:,6], yerr=dataset[i][:,7], fmt="none")
    k += 1
plt.legend(loc="upper left")
plt.xlabel(r"$T^*$")
plt.ylabel(r"$E$")
plt.axvline(x=crit, linestyle='--')
plt.title("Energy per spin")


plt.subplot(224)

k = 0
for i in xrange(len(dataset)):
    if i == 1:
        plt.plot(dataset[i][:,0], dataset[i][:,4] * zoom[k]**2, ls="--", marker="o", label = labels[k], markersize=5) 
    else:
        plt.plot(dataset[i][a:b,0], dataset[i][a:b,8] * zoom[k]**2, ls="--", marker="o", label = labels[k], markersize=5)
    # plt.errorbar(dataset[i][a:b:step,0], dataset[i][a:b:step,8] * zoom[k], yerr=dataset[i][a:b:step,9]*zoom[k], fmt="none")
    k += 1
plt.legend(loc="upper right")
plt.xlabel(r"$T^*$")
plt.ylabel(r"$C_h$")
plt.axvline(x=crit, linestyle='--')
plt.title("Specific Heat")

plt.tight_layout()
plt.savefig("observ.eps")
plt.show()

