import numpy as np
import matplotlib.pyplot as plt
from mpltools import color

# load data
N = np.load("N.npy")
R = np.load("R.npy")
offset = np.load("offset.npy")
defects = np.load("defects.npy")
area_simple = np.load("area_simple.npy")
area_weighted = np.load("area_weighted.npy")

# defect and curvatures
plt.figure()
plt.xlabel(r"$R$ in km")
plt.ylabel(r"Angular defect in rad")
ax = plt.gca()
for j, n in enumerate(N):
    y = np.average(defects[j,:,:], axis=1)
    std = np.std(defects[j,:,:], axis=1)
    plt.plot(R, y, label="N = {}".format(n))
    ax.fill_between(R, (y-std), (y+std), alpha=0.2)
plt.legend(loc="lower right")
plt.savefig("defect_vs_R.pdf")

curvatures = defects / area_simple
plt.figure()
plt.xlabel(r"$R$ in km")
plt.ylabel(r"Curvature in 1/min$^2$")
ax = plt.gca()
for j, n in enumerate(N):
    y = np.average(curvatures[j,:,:], axis=1)
    std = np.std(curvatures[j,:,:], axis=1)
    plt.plot(R, y, label="N = {}".format(n))
    ax.fill_between(R, (y-std), (y+std), alpha=0.2)
plt.legend(loc="upper right")
plt.savefig("curvature_simple_vs_R.pdf")

curvatures = defects / area_weighted
plt.figure()
plt.xlabel(r"$R$ in km")
plt.ylabel(r"Curvature in 1/min$^2$")
ax = plt.gca()
for j, n in enumerate(N):
    y = np.average(curvatures[j,:,:], axis=1)
    std = np.std(curvatures[j,:,:], axis=1)
    plt.plot(R, y, label="N = {}".format(n))
    ax.fill_between(R, (y-std), (y+std), alpha=0.2)
plt.legend(loc="upper right")
plt.savefig("curvature_weighted_vs_R.pdf")

# defects vs offsets
Nselect = 4
selection = np.arange(0, R.size, R.size//4)
Rselection = R[selection]
for i in selection:
    plt.figure()
    plt.xlabel(r"Offset in rad")
    plt.ylabel(r"Angular defect in rad")
    plt.title("R = {}".format(R[i]))
    for j, n in enumerate(N):
        plt.plot(offset, defects[j,i,:], label="N = {}".format(n))
    plt.legend()
    plt.savefig("offset_R={}.pdf".format(R[i]))

plt.close("all")
