import numpy as np
import sys, csv, os
sys.path.append('/home/david/code_repo/osrm-py-master/')
import osrm

# random commment

from utils import interior_angle, heron, mercator, get_tripmeasures, asymmetry_parameter
from scipy.spatial import Delaunay, ConvexHull

if len(sys.argv) != 3:
    print("ERROR: Wrong number of arguments.")
    print("Usage: python AGS.py AGS_code store_path")
else:
    # parameters
    EPS = 1e-14

    # command line arguments
    AGS = str(sys.argv[1])
    store_path = str(sys.argv[2]) + AGS

    # load coordinates
    coordinates = []
    with open('/home/david/geonames/DE.txt', newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter='\t')
        for row in datareader:
            if row[8][:len(AGS)] == AGS:
                latitude = float(row[9])
                longitude = float(row[10])
                coordinates.append([longitude,latitude])
    coordinates = np.array(coordinates)

    # Mercator projection and Delaunay triangulation
    coordinates_mercator = mercator(coordinates)
    tri = Delaunay(coordinates_mercator)
    N_triangles = tri.simplices.shape[0]
    hull = ConvexHull(coordinates_mercator)
    interior = np.setdiff1d(np.arange(coordinates.shape[0]),
                            hull.vertices)

    # collect edges
    edges = []
    simplices_to_edges = np.zeros_like(tri.simplices)
    counter = 0
    for i, s in enumerate(tri.simplices):
        for j in range(3):
            v1 = s[j]
            v2 = s[(j+1)%3]
            if v1 < v2:
                edges.append((v1, v2))
                simplices_to_edges[i][j] = counter
                counter += 1
    for i, s in enumerate(tri.simplices):
        for j in range(3):
            v1 = s[j]
            v2 = s[(j+1)%3]
            if v1 > v2:
                try:
                    idx = edges.index((v2, v1))
                    simplices_to_edges[i][j] = idx
                except:
                    edges.append((v2, v1))
                    simplices_to_edges[i][j] = counter
                    counter += 1

    # measure edges
    client = osrm.Client(host='http://134.76.24.136/osrm')
    edge_lengths = np.zeros((len(edges),2))
    for i, e in enumerate(edges):
        A = coordinates[e[0]]
        B = coordinates[e[1]]
        edge_lengths[i] = get_tripmeasures(A, B, client)
        print("{} of {} edges measured.".format(i+1, len(edges)))

    # assign lengths to simplices
    D = np.zeros((N_triangles, 3, 2))
    for i in range(N_triangles):
        simplex = tri.simplices[i,:]
        for j in range(3):
            e = simplices_to_edges[i][j]
            D[i][j] = edge_lengths[e]

    # calculate angles and areas
    angles = interior_angle(D, np.roll(D,1,axis=1),
                            np.roll(D,2,axis=1))
    areas = heron(D[:,0,:], D[:,1,:], D[:,2,:])
    durations_sorted = np.sort(D[:,:,0])
    durations_check = (durations_sorted[:,2] - durations_sorted[:,1]
                       - durations_sorted[:,0])
    mask0 = np.where(np.logical_or(durations_check > 0,
                                  np.abs(durations_check) < EPS))
    if np.sum(mask0[0] > 0):
        for s in mask0[0]:
            half_angle = (np.argmin(D[s,:,0]) + 1) % 3
            angles[s,:,0] = 0
            angles[s,half_angle,0] = np.pi
            areas[s,0] = 0
    distances_sorted = np.sort(D[:,:,1])
    distances_check = (distances_sorted[:,2] - distances_sorted[:,1]
                       - distances_sorted[:,0])
    mask0 = np.where(np.logical_or(distances_check > 0,
                                  np.abs(distances_check) < EPS))
    if np.sum(mask0[0] > 0):
        for s in mask0[0]:
            half_angle = (np.argmin(D[s,:,1]) + 1) % 3
            angles[s,:,1] = 0
            angles[s,half_angle,1] = np.pi
            areas[s,1] = 0

    # calculate defects and curvatures
    defects = []
    curvatures = []
    for k in interior:
        mask = np.array(tri.simplices == k)
        theta = angles[mask]
        R = 2*np.pi - np.sum(theta, axis=0)
        defects.append(R)
        F = areas[np.sum(mask,axis=1) == 1]
        curvatures.append(R / (np.sum(F, axis=0)/3))

    # if necessary, create directory for storing
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    # save stuff
    np.save(store_path+"/coordinates.npy", coordinates)
    np.save(store_path+"/coordinates_mercator.npy", coordinates_mercator)
    np.save(store_path+"/simplices.npy", tri.simplices)
    np.save(store_path+"/interior.npy", interior)
    np.save(store_path+"/D.npy", D)
    np.save(store_path+"/defects.npy", defects)
    np.save(store_path+"/curvatures.npy", curvatures)
    np.save(store_path+"/angles.npy", angles)
    np.save(store_path+"/areas.npy", areas)
