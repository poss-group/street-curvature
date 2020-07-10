import numpy as np
import sys, os
sys.path.append('/home/david/code_repo/osrm-py-master/')
import osrm

from utils import construct_polygon, measure_polygon

if len(sys.argv) != 8:
    print("ERROR: Wrong number of arguments.")
    print("Usage: python polygon.py longitude latitude Nmax Rmin Rmax Rvalues offset_values")
else:
    client = osrm.Client(host='http://134.76.24.136/osrm')

    # command line arguments
    A = np.array([float(sys.argv[1]), float(sys.argv[2])])
    Nmax = int(sys.argv[3])
    Rmin = float(sys.argv[4])
    Rmax = float(sys.argv[5])
    Rvalues = int(sys.argv[6])
    offset_values = int(sys.argv[7])

    # parameters
    N = np.arange(3, Nmax+1)
    R = np.linspace(Rmin, Rmax, Rvalues)
    offset = np.linspace(0, 2*np.pi, offset_values)

    # data containers
    defects = np.zeros((N.size, R.size, offset.size))
    area_simple = np.zeros((N.size, R.size, offset.size))
    area_weighted = np.zeros((N.size, R.size, offset.size))

    # calculation
    Npolygons = N.size * R.size * offset.size
    counter = 0
    for k, n in enumerate(N):
        for i, r in enumerate(R):
            for j, alpha in enumerate(offset):
                B = construct_polygon(n, r, A, offset=alpha)
                angles, areas = measure_polygon(A, B, client)
                defects[k][i][j] = 2*np.pi - np.sum(angles)
                area_simple[k][i][j] = np.sum(areas) / 3
                area_weighted[k][i][j] = np.sum(areas*angles) / np.sum(angles)
                counter += 1
                print("{} of {} polygons evaluated.".format(counter, Npolygons))

    # save data
    np.save("defects.npy", defects)
    np.save("area_simple.npy", area_simple)
    np.save("area_weighted.npy", area_weighted)
    np.save("N.npy", N)
    np.save("R.npy", R)
    np.save("offset.npy", offset)

    # write metadata to file
    logfile = open("polygon.log", 'w')
    logfile.write("Location: "+str(sys.argv[1])+str(sys.argv[2])+"\n")
    logfile.write("Edges: "+str(N)+"\n")
    logfile.write("Polygon circumradii: {} values from {} to {}\n".format(Rvalues, Rmin, Rmax))
    logfile.write("{} offset values each.\n".format(offset_values))
