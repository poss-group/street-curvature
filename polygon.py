import numpy as np
import networkx as nx
import sys, os
import osrm
import pandas as pd
from copy import deepcopy

from utils import construct_polygon, measure_polygon, interior_angle, heron
from utils_network import *
from models import get_polygon_coordinates

def model_measure_polygon(G, A, B):
    N = B.shape[0]
    d = np.zeros((N, 2))
    c = np.zeros((N, 2))
    for j in range(N):
        c[j] = np.array([nx.shortest_path_length(G, source=A,
                                          target=B[j],
                                          weight='time'),
                  nx.shortest_path_length(G, source=A,
                                          target=B[j],
                                          weight='dist')])
        d[j] = np.array([nx.shortest_path_length(G, source=B[j],
                                          target=B[(j+1)%N],
                                          weight='time'),
                  nx.shortest_path_length(G, source=B[j],
                                          target=B[(j+1)%N],
                                          weight='dist')])
    mask1 = d > (c + np.roll(c, 1, axis=0))
    mask2 = np.abs(c - np.roll(c, 1 ,axis=0)) > d
    angles = interior_angle(c, np.roll(c, 1, axis=0), d)
    areas = heron(c, np.roll(c, 1, axis=0), d)
    angles[mask1] = np.pi
    angles[mask2] = 0
    areas[mask1] = 0
    areas[mask2] = 0

    return angles, areas, np.average(c, axis=0)

def polygon_scan(location, N, R, offset, store_path):
    """
    """
    client = osrm.Client(host='http://134.76.24.136/osrm')

    # set up dataframe
    defects = np.zeros((N.size, R.size, offset.size, 2))
    area_simple = np.zeros((N.size, R.size, offset.size, 2))
    # area_weighted = np.zeros((N.size, R.size, offset.size, 2))
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
    df.to_csv(store_path+"/scan_data.csv")

def model_polygon_scan(G, center, N, R, offset, store_path):
    """
    """
    # get center coordinates
    A = G.nodes[center]['pos']

    # convert graph to GeoDataFrame
    gdf = graph_to_gdf(G)
    geom = gdf["geometry"]

    # calculate spatial index
    rtree = RTreeIndex()
    for pos, bounds in enumerate(geom.bounds.values):
            rtree.insert(pos, bounds)

    # set up data containers
    defects = np.zeros((N.size, R.size, offset.size, 2))
    area_simple = np.zeros((N.size, R.size, offset.size, 2))
    # area_weighted = np.zeros((N.size, R.size, offset.size, 2))
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
                Bpos = get_polygon_coordinates(A, n, r, alpha)
                ne, refdist = snap_to_edge_position(gdf, Bpos, rtree=rtree)
                H = deepcopy(G)
                for ell, e in enumerate(ne):
                    u = gdf.loc[e]["u"]
                    v = gdf.loc[e]["v"]
                    subdivide_edge(H, u, v, [refdist[ell]], 'dist')
                B = G.number_of_nodes() + np.arange(1, n+1)
                angles, areas, meanR = model_measure_polygon(H, center, B)
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
    df.to_pickle(store_path+"/scan_data.pkl")

if __name__ == "__main__":
    # test model scan
    from models import boost_random_fraction, gabriel
    np.random.seed(seed=250)
    N = 2000
    points = np.sqrt(N)*np.random.random((N, 2))
    G = gabriel(points)
    boost_random_fraction(G, 0.3, 0.9)
    A = get_barycentric_node(G)
    n = np.array([6])
    R = np.sqrt(N) / 3
    radii = np.linspace(0.9, 1, 2) * R
    offset = (np.pi/3) * np.arange(10) / 10
    model_polygon_scan(G, A, n, radii, offset, './data/')
