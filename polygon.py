import numpy as np
import sys, os
sys.path.append('/home/david/code_repo/osrm-py-master/')
import osrm
import pandas as pd

from utils import construct_polygon, measure_polygon

def polygon_scan(location, N, R, offset, store_path):
    """
    """
    client = osrm.Client(host='http://134.76.24.136/osrm')

    # set up dataframe
    defects = np.zeros((N.size, R.size, offset.size, 2))
    area_simple = np.zeros((N.size, R.size, offset.size, 2))
    area_weighted = np.zeros((N.size, R.size, offset.size, 2))
    mean_R = np.zeros((N.size, R.size, offset.size, 2))
    R_full = np.zeros((N.size, R.size, offset.size))
    N_full = np.zeros((N.size, R.size, offset.size), dtype=int)
    offset_full = np.zeros((N.size, R.size, offset.size))

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
                R_full[k][i][j] = r
                offset_full[k][i][j] = alpha
                N_full[k][i][j] = n
                counter += 1
                print("{} of {} polygons evaluated.".format(counter, Npolygons))

    # if necessary, create directory for storing
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # set up dataframe
    df = pd.DataFrame()
    df['defect_duration'] = (defects[:,:,:,0]).flatten()
    df['area_duration'] = (area_simple[:,:,:,0]).flatten()
    df['meanR_duration'] = (mean_R[:,:,:,0]).flatten()
    df['defect_distance'] = (defects[:,:,:,1]).flatten()
    df['area_distance'] = (area_simple[:,:,:,1]).flatten()
    df['meanR_distance'] = (mean_R[:,:,:,1]).flatten()
    df['radius'] = R_full.flatten()
    df['number of edges'] = N_full.flatten()
    df['offset'] = offset_full.flatten()

    # save data
    np.save(store_path+"/location.npy", location)
    df.to_pickle(store_path+"/scan_data.pkl")
