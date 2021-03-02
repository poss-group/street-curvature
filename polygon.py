import numpy as np
import sys, os
sys.path.append('/home/david/code_repo/osrm-py-master/')
import osrm

from utils import construct_polygon, measure_polygon

def polygon_scan(location, N, R, offset, store_path):
    """
    """
    client = osrm.Client(host='http://134.76.24.136/osrm')

    # data containers
    defects = np.zeros((N.size, R.size, offset.size, 2))
    area_simple = np.zeros((N.size, R.size, offset.size, 2))
    area_weighted = np.zeros((N.size, R.size, offset.size, 2))
    mean_R = np.zeros((N.size, R.size, offset.size, 2))

    # calculation
    Npolygons = N.size * R.size * offset.size
    counter = 0
    for k, n in enumerate(N):
        for i, r in enumerate(R):
            for j, alpha in enumerate(offset):
                B = construct_polygon(n, r, location, offset=alpha)
                angles, areas, meanR = measure_polygon(location, B, client,
                                                       meanR=True)
                mean_R[k][i][j] = meanR
                defects[k][i][j] = 2*np.pi - np.sum(angles, axis=0)
                area_simple[k][i][j] = np.sum(areas, axis=0) / 3
                area_weighted[k][i][j] = (np.sum(areas*angles, axis=0)
                                          / np.sum(angles, axis=0))
                counter += 1
                print("{} of {} polygons evaluated.".format(counter, Npolygons))

    # if necessary, create directory for storing
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # save data
    np.save(store_path+"/location.npy", location)
    np.save(store_path+"/defects.npy", defects)
    np.save(store_path+"/area_simple.npy", area_simple)
    np.save(store_path+"/area_weighted.npy", area_weighted)
    np.save(store_path+"/meanR.npy", mean_R)
    np.save(store_path+"/N.npy", N)
    np.save(store_path+"/R.npy", R)
    np.save(store_path+"/offset.npy", offset)
