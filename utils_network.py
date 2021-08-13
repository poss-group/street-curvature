import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import KDTree
from rtree.index import Index as RTreeIndex
from shapely.geometry import Point, LineString
import pandas as pd
import geopandas as gpd
import xml.etree.ElementTree as etree
from scipy.integrate import cumtrapz
from copy import deepcopy
import multiprocessing as mp
from scipy.interpolate import PchipInterpolator
from scipy.sparse.linalg import eigsh

def overlap(t, left, right):
    return np.minimum(np.maximum(0, t-left), right-left)

def boxcar(t, left, right):
    return (t >= left) * (t < right)

def params_to_bins(a, b, smax, kind='both'):
    if kind == 'both':
        return np.array([a, np.minimum(b, a+smax), np.maximum(b, a+smax), b+smax])
    if kind == 'single':
        return np.array([a, a+smax, b, b])

def get_ramp_bins(lu, lv, dAu, dAv, dBu, dBv, wAB, wuv):
    bins1 = params_to_bins(dAu, 0.5*(dAu+dAv+wuv), lu)
    bins2 = params_to_bins(dAv, 0.5*(dAu+dAv+wuv), lu)
    bins3 = params_to_bins(dBu+wAB-lu, 0.5*(wAB+wuv+dBu+dAv), lv-lu, kind='single')
    bins4 = params_to_bins(dAv-lu, 0.5*(wAB+wuv+dBu+dAv), lv-lu, kind='single')
    bins5 = params_to_bins(dBu, 0.5*(dBu+dBv+wuv), wAB-lv)
    bins6 = params_to_bins(dBu, 0.5*(dBu+dBv+wuv), wAB-lv)

    return np.concatenate((bins1, bins2, bins3, bins4, bins5, bins6), axis=-1)

def ramp_function(times, bins, slope):
    ll, lr, rl, rr = bins
    # ramp function expressed as sum of areas
    t = np.expand_dims(times, axis=-1)
    base1 = overlap(t, ll, lr)
    base2 = overlap(t, lr, rl)
    base3 = (rr-rl) - overlap(t, rl, rr)
    area1 = 0.5 * slope * base1**2
    area2 = slope * (lr-ll) * base2
    area3 = (0.5 * slope * (rr-rl)**2
             - 0.5 * slope * base3**2)

    return np.sum(area1 + area2 + area3, axis=-1)

def self_ramp(t, w):
    return (t < w) * (2*t*w-t**2) + (t >= w) * w**2

def calc_row_idx(k, n):
    return int(np.ceil((1/2.) * (- (-8*k + 4 *n**2 -4*n - 7)**0.5 + 2*n -1) - 1))

def elem_in_i_rows(i, n):
    return i * (n - 1 - i) + (i*(i + 1))//2

def calc_col_idx(k, i, n):
    return int(n - elem_in_i_rows(i + 1, n) + k)

def condensed_to_square(k, n):
    """
    Get the square matrix indices from a condensed distance matrix index.

    Parameters
    ----------
    k : int
        Index of condensed matrix
    n : int
        size of square matrix

    Returns
    -------
    i, j : ints
        Indices in square matrix
    """
    i = calc_row_idx(k, n)
    j = calc_col_idx(k, i, n)
    return i, j

def equally_spaced_edge_position_sample(G, Nsamples, weight):
    """
    Sample equally spaced positions on the (undirected)
    edges of G.

    Parameters
    ----------
    G : nx.networkx.MultiDiGraph
       The network, must allow directed and multi-edges.
    Nsamples : int
        Number of samples.
    weight : string
        Edge weight

    Returns
    -------
    edge_positions : dict
        Dictionary of edge positions keyed by edge, with 1-D ndarrays
        as values.

    Notes
    -----
    The sample is equally spaced on the directed edges of G, so that opposite
    edges are not oversampled.
    """
    oneway = np.array(list(nx.get_edge_attributes(G, 'oneway').values()))
    u, v, k, w = zip(*G.edges(data=weight, keys=True))
    u = np.array(u)
    v = np.array(v)
    mask = np.logical_or(oneway, u < v) # avoid double counting of two-way edges
    k = np.array(k)[mask]
    w = np.array(w)[mask]
    u = u[mask]
    v = v[mask]
    bins = np.cumsum(w)
    L = bins[-1]
    print(L)

    # take sample and find edges and positions
    x = (np.arange(Nsamples) + 0.5) * (L/(Nsamples+1))
    x_indices = np.digitize(x, bins)
    x_positions = (x - bins[x_indices-1]) % L

    # collect positions for each edge
    edge_positions = {}
    for idx in np.unique(x_indices):
        mask = x_indices == idx
        positions = x_positions[mask] / w[idx]
        key = (u[idx], v[idx], k[idx])
        if not G.has_edge(*key):
            # wrong edge direction
            key = (v[idx], u[idx], k[idx])
        edge_positions[key] = positions

    return edge_positions

def subdivide_edge(G, u, v, positions_on_edge, weight):
    """
    Subdivide edge between u and v at specified positions.

    Parameters
    ----------
    G : networkx.Graph
        The graph the edge belongs to.
    u, v : nodes
        The edge between nodes u and v is subdivided.
    positions_on_edge : list of floats
        List of linear values referencing the positions on the edge.
    weight : string
        Edge weight used in the linear referencing. Either 'dist' for
        Euclidean distance or 'time' for travel time along edge.
    """
    N = G.number_of_nodes()
    M = len(positions_on_edge)
    pos = nx.get_node_attributes(G, "pos")
    edge_attributes = G[u][v]
    direction = (pos[v] - pos[u]) / edge_attributes['dist']
    if weight == 'time':
        direction *= edge_attributes['speed']
    new_nodes = [(N+i, {"pos": pos[u]+l*direction})
                 for i, l in enumerate(positions_on_edge)]
    G.add_nodes_from(new_nodes)
    path = [u] + list(range(N, N+M)) + [v]
    nx.add_path(G, path)
    for i in range(M+1):
        v1 = path[i]
        v2 = path[i+1]
        d = np.linalg.norm(G.nodes[v1]['pos']-G.nodes[v2]['pos'])
        G[v1][v2]['dist'] = d
        if weight == 'time':
            G[v1][v2]['speed'] = edge_attributes['speed']
            G[v1][v2]['time'] = d / edge_attributes['speed']

    G.remove_edge(u, v)

def get_circles(G, center, radii, weight):
    """
    Get points on a circle in a graph G.

    Parameters
    ----------
    G : networkx.Graph
        The network
    center : node identifier
        The center around which to construct the circles.
    radii : iterable of floats
        The radii of the circles
    weight : string
        Edge attribute with respect to which the circles are
        constructed.

    Returns
    -------
    circles : list of tuples
        Circles as a lists of points on edges, which are represented
        tuples (u, v, ell) of the edge vertices u and v and the linear
        referencing distance ell.
    """
    # calculate shortest path lengths to center
    dist = pd.Series(nx.shortest_path_length(G, source=center, weight=weight))

    # get edge data
    u, v, w = zip(*G.edges(data=True))
    u = np.array(u)
    v = np.array(v)
    w = np.array([item[weight] for item in w])
    du = np.array(dist[u])
    dv = np.array(dist[v])
    dmin = np.minimum(du, dv)
    dmax = np.maximum(du, dv)
    dmaxB = 0.5 * (du + dv + w)
    start = u * (du < dv) + v * (du >= dv)
    end = v * (du < dv) + u * (du >= dv)

    # loop through radii
    circles = []
    for R in radii:
        l1 = R - dmin
        l2 = dmax - dmin + w - l1
        mask1 = dmin < R
        mask2 = dmax >= R
        mask3 = dmaxB >= R
        typeA = mask1 * mask2
        typeB = mask1 * ~mask2 * mask3
        circles.append(list(zip(start[typeA], end[typeA],
                                np.array([l1[typeA]]).T)) +
                       list(zip(start[typeB], end[typeB], np.array([l1[typeB], l2[typeB]]).T)))

    return circles

def volume_growth_edge(G, A, B, distances, times, weight):
    # select distances
    dist2A = pd.Series(distances[A])
    dist2B = pd.Series(distances[B])
    wAB = G[A][B][weight]

    # loop through edges != (A,B) to get bins of ramp function
    n1, n2, w = zip(*G.edges(data=weight))
    n1 = np.array(n1)
    n2 = np.array(n2)
    mask = ~np.logical_or(np.logical_and(n1 == A, n2 == B),
                          np.logical_and(n1 == B, n2 == A))
    w = np.array(w)
    n1 = n1[mask]
    n2 = n2[mask]
    w = w[mask]
    dA1 = np.array(dist2A[n1])
    dA2 = np.array(dist2A[n2])
    dB1 = np.array(dist2B[n1])
    dB2 = np.array(dist2B[n2])
    ell1 = 0.5 * (wAB + dB1 - dA1)
    ell2 = 0.5 * (wAB + dB2 - dA2)
    u = n1 * (ell1 < ell2) + n2 * (ell1 >= ell2)
    v = n2 * (ell1 < ell2) + n1 * (ell1 >= ell2)
    lu = np.minimum(ell1, ell2)
    lv = np.maximum(ell1, ell2)
    dAu = dA1 * (ell1 < ell2) + dA2 * (ell1 >= ell2)
    dAv = dA2 * (ell1 < ell2) + dA1 * (ell1 >= ell2)
    dBu = dB1 * (ell1 < ell2) + dB2 * (ell1 >= ell2)
    dBv = dB2 * (ell1 < ell2) + dB1 * (ell1 >= ell2)
    bins = get_ramp_bins(lu, lv, dAu, dAv, dBu, dBv, wAB, w)
    print(bins)
    volume = ramp_function(times, bins, 1)

    # add the edge (A, B)
    volume += self_ramp(times, wAB)

    return volume / wAB

def get_edge_location_curves(G, edge, weight, times):
    edge_data = G.get_edge_data(*edge)
    if 'locations' not in edge_data:
        return [], []
    else:
        locations = edge_data['locations']
    if locations.size == 0:
        return [], []

    A = edge[0]
    B = edge[1]

    # calculate shortest path lengths
    dist2A = dict(nx.shortest_path_length(G, source=A, weight=weight))
    dist2B = dict(nx.shortest_path_length(G, source=B, weight=weight))
    wAB = G.get_edge_data(*edge)[weight]

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    for u, v, k, w in G.edges(data=weight, keys=True):
        if edge != (u, v, k):
            du = np.minimum(dist2A[u] + wAB*locations,
                            dist2B[u] + wAB*(1-locations))
            dv = np.minimum(dist2A[v] + wAB*locations,
                            dist2B[v] + wAB*(1-locations))
            dmax = 0.5 * (du + dv + w)
            left.append(du)
            left.append(dv)
            right.append(dmax)
            right.append(dmax)
        else:
            left.append(np.zeros_like(locations))
            left.append(np.zeros_like(locations))
            right.append(wAB*locations)
            right.append(wAB*(1-locations))
    left = np.array(left)
    right = np.array(right)
    t = np.expand_dims(times, axis=(1, 2))
    ramps = overlap(t, left, right)
    steps = boxcar(t, left, right)

    volumes = np.sum(ramps, axis=1)
    rates = np.sum(steps, axis=1)

    return [v for v in volumes.T], [r for r in rates.T]

def edge_location_analysis(G, edge, weight, Ninter):
    edge_data = G.get_edge_data(*edge)
    if 'locations' not in edge_data:
        return None, 0
    else:
        locations = edge_data['locations']
    if locations.size == 0:
        return [], 0

    A = edge[0]
    B = edge[1]

    # calculate shortest path lengths
    dist2A = dict(nx.shortest_path_length(G, source=A, weight=weight))
    dist2B = dict(nx.shortest_path_length(G, source=B, weight=weight))
    wAB = G.get_edge_data(*edge)[weight]

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    for u, v, k, w in G.edges(data=weight, keys=True):
        if edge != (u, v, k):
            du = np.minimum(dist2A[u] + wAB*locations,
                            dist2B[u] + wAB*(1-locations))
            dv = np.minimum(dist2A[v] + wAB*locations,
                            dist2B[v] + wAB*(1-locations))
            dmax = 0.5 * (du + dv + w)
            left.append(du)
            left.append(dv)
            right.append(dmax)
            right.append(dmax)
        else:
            left.append(np.zeros_like(locations))
            left.append(np.zeros_like(locations))
            right.append(wAB*locations)
            right.append(wAB*(1-locations))
    times = np.linspace(0, 2*np.amax(right), Ninter)
    left = np.array(left)
    right = np.array(right)
    t = np.expand_dims(times, axis=(1, 2))
    ramps = overlap(t, left, right)
    volumes = np.sum(ramps, axis=1)
    splines = []
    for v in volumes.T:
        splines.append(PchipInterpolator(times, v))

    return splines, np.amax(right)

def node_analysis(G, A, weight, Ninter):

    # calculate shortest path lengths
    dist2A = dict(nx.shortest_path_length(G, source=A, weight=weight))

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    for u, v, w in G.edges(data=weight):
        du = dist2A[u]
        dv = dist2A[v]
        dmax = 0.5 * (du + dv + w)
        left.append(du)
        left.append(dv)
        right.append(dmax)
        right.append(dmax)
    times = np.linspace(0, 2*np.amax(right), Ninter)
    left = np.array(left)
    right = np.array(right)
    t = np.expand_dims(times, axis=-1)
    ramps = overlap(t, left, right)
    volume = np.sum(ramps, axis=-1)
    spl = PchipInterpolator(times, volume)

    return spl, np.amax(right)

def get_node_curves(G, A, weight, times):

    # calculate shortest path lengths
    dist2A = dict(nx.shortest_path_length(G, source=A, weight=weight))

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    for u, v, w in G.edges(data=weight):
        du = dist2A[u]
        dv = dist2A[v]
        dmax = 0.5 * (du + dv + w)
        left.append(du)
        left.append(dv)
        right.append(dmax)
        right.append(dmax)
    left = np.array(left)
    right = np.array(right)
    t = np.expand_dims(times, axis=-1)
    ramps = overlap(t, left, right)
    steps = boxcar(t, left, right)
    volume = np.sum(ramps, axis=-1)
    rate = np.sum(steps, axis=-1)

    return volume, rate

def volume_growth_analysis(G, weight, Ninter):
    # set up and perform mutliprocessing
    cpus = mp.cpu_count() - 1
    pool = mp.Pool(cpus)

    # volume growth at nodes
    nodes = list(G.nodes())
    params = [(G, A, weight, Ninter) for A in nodes]
    sma = pool.starmap_async(node_analysis, params)
    results = sma.get()
    pool.close()
    pool.join()
    node_splines = []
    distances = []
    for spl, d in results:
        node_splines.append(spl)
        distances.append(d)
    spline_dict = dict(zip(nodes, node_splines))
    nx.set_node_attributes(G, spline_dict, name='volume_growth')

    # volume growth at edge locations
    pool = mp.Pool(cpus)
    edges = list(G.edges(keys=True))
    params = [(G, e, weight, Ninter) for e in edges]
    sma = pool.starmap_async(edge_location_analysis, params)
    results = sma.get()
    pool.close()
    pool.join()
    edge_splines = []
    for spl, d in results:
        edge_splines.append(spl)
        distances.append(d)
    spline_dict = dict(zip(edges, edge_splines))
    nx.set_edge_attributes(G, spline_dict, name='volume_growth')

    return np.amax(distances)

def get_volume_growth_curves(G, weight, times):
    # set up and perform mutliprocessing
    cpus = mp.cpu_count() - 1
    pool = mp.Pool(cpus)

    # volume growth at nodes
    nodes = list(G.nodes())
    params = [(G, A, weight, times) for A in nodes]
    sma = pool.starmap_async(get_node_curves, params)
    results = sma.get()
    pool.close()
    pool.join()
    node_volumes = dict(zip(nodes, list(results)))

    # volume growth at edge locations
    pool = mp.Pool(cpus)
    edges = list(G.edges(keys=True))
    params = [(G, e, weight, times) for e in edges]
    sma = pool.starmap_async(get_edge_location_curves, params)
    results = sma.get()
    pool.close()
    pool.join()
    edge_volumes = dict(zip(edges, list(results)))

    return node_volumes, edge_volumes


def volume_growth_grid_measure_performance(G, Npos, times, weight):
    import time

    # construct distance matrix
    start = time.time()
    S = []
    N = G.number_of_nodes()
    distances = np.zeros((N, N))
    for source, ddict in nx.shortest_path_length(G, weight=weight):
        distances[source,list(ddict.keys())] = np.array(list(ddict.values()))
    stop = time.time()
    print("Shortest path calculation: {} s".format(stop-start))

    start = time.time()
    u, v, w = zip(*G.edges(data=weight))
    u = np.array(u)
    v = np.array(v)
    w = np.array(w)
    bins = np.cumsum(w)
    L = bins[-1]
    stop = time.time()
    print("Edge manipulations: {} s".format(stop-start))

    start = time.time()
    x = (np.arange(Npos) + 0.5) * (L/(Npos+1))
    X, Y = np.meshgrid(x, x)
    X = X.flatten()
    Y = Y.flatten()
    # X, Y = L * np.random.random((2, Npos**2))

    x_indices = np.digitize(X, bins)
    x_positions = (X - bins[x_indices-1]) % L
    y_indices = np.digitize(Y, bins)
    y_positions = (Y - bins[y_indices-1]) % L
    stop = time.time()
    print("Grid setup: {} s".format(stop-start))

    start = time.time()
    d1 = x_positions + distances[u[x_indices], u[y_indices]] + y_positions
    d2 = x_positions + distances[u[x_indices], v[y_indices]] + w[y_indices] - y_positions
    d3 = w[x_indices] - x_positions + distances[v[x_indices], u[y_indices]] + y_positions
    d4 = w[x_indices] - x_positions + distances[v[x_indices], v[y_indices]] + w[y_indices] - y_positions
    on_diagonal = x_indices == y_indices
    d = on_diagonal * np.abs(X-Y) + ~on_diagonal * np.amin([d1, d2, d3, d4], axis=0)
    stop = time.time()
    print("Distance calculation: {} s".format(stop-start))


    start = time.time()
    mask = np.expand_dims(d, axis=-1) <= times
    volume = np.average(mask, axis=0)
    stop = time.time()
    print("Volume calculation: {} s".format(stop-start))

    return volume, L

def volume_growth_grid(G, Npos, times, weight):
    # construct distance matrix
    N = G.number_of_nodes()
    distances = np.zeros((N, N))
    for source, ddict in nx.shortest_path_length(G, weight=weight):
        distances[source,list(ddict.keys())] = np.array(list(ddict.values()))

    u, v, w = zip(*G.edges(data=weight))
    u = np.array(u)
    v = np.array(v)
    w = np.array(w)
    bins = np.cumsum(w)
    L = bins[-1]

    x = (np.arange(Npos) + 0.5) * (L/(Npos+1))
    X, Y = np.meshgrid(x, x)
    X = X.flatten()
    Y = Y.flatten()
    # X, Y = L * np.random.random((2, Npos**2))

    x_indices = np.digitize(X, bins)
    x_positions = (X - bins[x_indices-1]) % L
    y_indices = np.digitize(Y, bins)
    y_positions = (Y - bins[y_indices-1]) % L

    d1 = x_positions + distances[u[x_indices], u[y_indices]] + y_positions
    d2 = x_positions + distances[u[x_indices], v[y_indices]] + w[y_indices] - y_positions
    d3 = w[x_indices] - x_positions + distances[v[x_indices], u[y_indices]] + y_positions
    d4 = w[x_indices] - x_positions + distances[v[x_indices], v[y_indices]] + w[y_indices] - y_positions
    on_diagonal = x_indices == y_indices
    d = on_diagonal * np.abs(X-Y) + ~on_diagonal * np.amin([d1, d2, d3, d4], axis=0)

    mask = np.expand_dims(d, axis=-1) <= times
    volume = np.average(mask, axis=0)

    return volume, L, np.amax(distances)

def volume_growth_edge_sample(G, edge_positions, Nsamples, Ninter, weight):
    N = G.number_of_nodes()
    distances = np.zeros((N, N))
    for source, ddict in nx.shortest_path_length(G, weight=weight):
        distances[source,list(ddict.keys())] = np.array(list(ddict.values()))
    diam = np.amax(distances)

    u, v, w = zip(*G.edges(data=weight))
    u = np.array(u)
    v = np.array(v)
    w = np.array(w)
    bins = np.cumsum(w)
    L = bins[-1]

    if type(edge_positions) == int:
        x = (np.arange(edge_positions) + 0.5) * (L/(edge_positions+1))
    else:
        x = edge_positions
    y = (np.arange(Nsamples) + 0.5) * (L/(Nsamples+1))
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    # X, Y = L * np.random.random((2, Npos**2))

    x_indices = np.digitize(X, bins)
    x_positions = (X - bins[x_indices-1]) % L
    y_indices = np.digitize(Y, bins)
    y_positions = (Y - bins[y_indices-1]) % L

    d1 = x_positions + distances[u[x_indices], u[y_indices]] + y_positions
    d2 = x_positions + distances[u[x_indices], v[y_indices]] + w[y_indices] - y_positions
    d3 = w[x_indices] - x_positions + distances[v[x_indices], u[y_indices]] + y_positions
    d4 = w[x_indices] - x_positions + distances[v[x_indices], v[y_indices]] + w[y_indices] - y_positions
    on_diagonal = x_indices == y_indices
    d = on_diagonal * np.abs(X-Y) + ~on_diagonal * np.amin([d1, d2, d3, d4], axis=0)

    times = np.linspace(0, 1.5*diam, Ninter)
    volumes = np.expand_dims(d, axis=-1) <= times
    splines = []
    for v in volumes:
        splines.append(PchipInterpolator(times, v))

    return splines, L, np.amax(distances)

def get_ellipse_detours(distances, x_positions, x_indices, u, v, w):
    # calculate detours
    detours = np.zeros_like(x_positions)
    for dist in distances:
        d1 = x_positions + dist[u[x_indices]]
        d2 = w[x_indices] - x_positions + dist[v[x_indices]]
        detours += np.minimum(d1, d2)

    return detours

def pairwise_expected_detours(G, weight, Npos):
    N = G.number_of_nodes()
    distances = np.zeros((N, N))
    for source, ddict in nx.shortest_path_length(G, weight=weight):
        distances[source,list(ddict.keys())] = np.array(list(ddict.values()))

    u, v, w = zip(*G.edges(data=weight))
    u = np.array(u)
    v = np.array(v)
    w = np.array(w)
    bins = np.cumsum(w)
    L = bins[-1]

    x = (np.arange(Npos) + 0.5) * (L/(Npos+1))
    x_indices = np.digitize(x, bins)
    x_positions = (x - bins[x_indices-1]) % L

    # loop through pairs
    expected_detour = []
    delta_max = []
    counter = 0
    for A in np.arange(N):
        for B in np.arange(A+1, N):
            detours = get_ellipse_detours(distances[[A, B], :], x_positions,
                                          x_indices, u, v, w) / distances[A,B]
            detours, counts = np.unique(detours, return_counts=True)
            detours = np.sort(detours)
            counts = counts[np.argsort(detours)]
            delta_max.append(detours)
            expected_detour.append(np.cumsum(detours*counts) / np.cumsum(counts))
            counter += 1
            print("{} of {} pairs calculated".format(counter, int(N*(N-1) // 2)))

    return delta_max, expected_detour

def volume_growth(G, times, weight):
    # first compute shortest path lengths and diameter
    distances = dict(nx.shortest_path_length(G, weight=weight))
    diam = []
    for n, ddict in distances.items():
        diam += list(ddict.values())
    diam = np.amax(diam)

    # set up mutliprocessing
    cpus = mp.cpu_count() - 1
    pool = mp.Pool(cpus)

    u, v, w = zip(*G.edges(data=weight))
    params = ((G, u, v, distances, times, weight) for u, v in zip(u, v))
    sma = pool.starmap_async(volume_growth_edge, params)
    growth_at_edge = sma.get()
    pool.close()
    pool.join()

    # take (weighted) average over edges
    w = np.array(w)
    volume = np.tensordot(w, np.array(growth_at_edge), axes=1) / np.sum(w)

    return volume, diam, np.sum(w)


def volume_growth_at_point(G, A, times, weight):

    # calculate shortest path lengths
    dist2A = dict(nx.shortest_path_length(G, source=A, weight=weight))

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    for u, v, w in G.edges(data=weight):
        du = dist2A[u]
        dv = dist2A[v]
        dmax = 0.5 * (du + dv + w)
        left.append(du)
        left.append(dv)
        right.append(dmax)
        right.append(dmax)
    left = np.array(left)
    right = np.array(right)

    t = np.expand_dims(times, axis=-1)
    ramps = overlap(t, left, right)
    volume = np.sum(ramps, axis=-1)

    return volume

def volume_growth_random_vertices(G, Npos, times, weight, normalize=True):
    lbd = np.zeros_like(times)
    for A in np.random.choice(list(G.nodes()), size=Npos, replace=False):
        lbd += volume_growth_at_point(G, A, times, weight)
    lbd /= Npos

    if normalize:
        lbd /= np.sum([w for u, v, w, in G.edges(data=weight)])

    return lbd

def volume_growth_all_vertices(G, times, weight, normalize=True):
    # set up mutliprocessing
    cpus = mp.cpu_count() - 1
    pool = mp.Pool(cpus)

    params = ((G, n, times, weight) for n in G.nodes())
    sma = pool.starmap_async(volume_growth_at_point, params)
    growth_at_point = sma.get()
    pool.close()
    pool.join()

    lbd = np.sum(growth_at_point, axis=0)
    N = G.number_of_nodes()
    lbd /= N

    if normalize:
        lbd /= np.sum([w for u, v, w, in G.edges(data=weight)])

    return lbd

def volume_growth_at_edge_position(G, u, v, pos, times, weight):
    N = G.number_of_nodes()
    H = deepcopy(G)
    subdivide_edge(H, u, v, [pos], weight)

    return volume_growth_at_point(H, N, times, weight)

def volume_growth_uniform_edge_sample(G, Npos, times, weight, normalize=True):
    # sample edge positions
    u, v, w = zip(*G.edges(data=weight))
    w = np.array(w)
    bins = np.cumsum(w)
    offset = np.random.random(1) * bins[0]
    samples = ((np.arange(Npos) / Npos ) * bins[-1] + offset) % bins[-1]
    edge_indices = np.digitize(samples, bins)
    edge_positions = bins[edge_indices] - samples

    # set up mutliprocessing
    cpus = mp.cpu_count() - 1
    pool = mp.Pool(cpus)
    params = ((G, u[e], v[e], pos, times, weight) for e, pos in zip(edge_indices, edge_positions))
    sma = pool.starmap_async(volume_growth_at_edge_position, params)
    growth_at_point = sma.get()
    pool.close()
    pool.join()

    # expand graph and calculate volume growth
    lbd = np.sum(growth_at_point, axis=0)
    lbd /= Npos

    if normalize:
        lbd /= bins[-1]

    return lbd

def volume_growth_random_edge_positions(G, Npos, times, weight,
                                        normalize=True):
    N = G.number_of_nodes()

    # sample edge positions
    u, v, w = zip(*G.edges(data=weight))
    w = np.array(w)
    bins = np.cumsum(w)
    samples = np.random.random(Npos) * bins[-1]
    edge_indices = np.digitize(samples, bins)
    edge_positions = bins[edge_indices] - samples

    # expand graph and calculate volume growth
    lbd = np.zeros_like(times)
    for e, pos in zip(edge_indices, edge_positions):
        lbd += volume_growth_at_edge_position(G, u[e], v[e], pos,
                                              times, weight)
    lbd /= Npos

    if normalize:
        lbd /= bins[-1]

    return lbd


def sample_volume_growth(G, times, weight):
    N = G.number_of_nodes()
    growth_rate = np.zeros_like(times)
    for A in G.nodes():
        circles = get_circles(G, A, times, weight)
        growth_rate += np.array([len(c) for c in circles])
    growth_rate /= N

    # integrate
    dt = times[1:]-times[:-1]
    left_sum = np.cumsum(growth_rate[:-1]*dt)
    right_sum = np.cumsum(growth_rate[1:]*dt)

    return left_sum, right_sum, (left_sum+right_sum)/2

def get_effective_dimension_bound(times, volumes):
    mask = times > 0
    y = np.log(volumes[mask])
    x = np.log(times[mask])


    return np.amax(np.gradient(y, x))

def graph_spectrum(G, weight, k):
    """
    Find k smallest eigenvalues of graph laplacian.
    """
    L = nx.laplacian_matrix(G, weight=weight).asfptype()
    return eigsh(L, k=k, which='SM')

def graph_to_gdf(G):
    pos = nx.get_node_attributes(G, 'pos')

    geom = []
    for u, v in G.edges():
        pts = [Point(pos[u]), Point(pos[v])]
        geom.append(LineString(pts))

    u, v = zip(*G.edges())
    gdf = gpd.GeoDataFrame(geometry=geom)
    gdf['u'] = u
    gdf['v'] = v

    return gdf


def snap_to_network_nodes(G, points):
    """
    Snap given points in the plane to nodes in a spatial network.

    Parameters
    ----------
    G : networkx.Graph
        The spatial network. Needs to have a 'pos' node attribute.
    points : array of floats, shape (M, 2)
        The cartesian coordinates of the points to be snapped.

    Returns
    -------
    neighbors : integer or array of integers, shape (M,)

    """
    nodes = np.array(G.nodes())
    pos = nx.get_node_attributes(G, 'pos')
    network_points = np.array(list(pos.values()))
    tree = KDTree(network_points)
    d, neighbors = tree.query(points)

    return nodes[neighbors]

def snap_to_edge_position(gdf, points, k=3, rtree=None):
    """
    Snap given points in the plane to edges in GeoDataFrame of edges.

    Parameters
    ----------
    gdf : GeoDataframe
        The edges of spatial network as a Geodataframe.
    points : array of floats, shape (M, 2)
        The cartesian coordinates of the points to be snapped.
    k : integer, optional
        Number of nearest edges to consider.

    Returns
    -------
    nearest_edges : list of integers, length M
        Indices of nearest edges in the GeoDataframe.
    refdistances : list of floats, length M
        Linear referencing distances of points along nearest edge.
    """
    X, Y = points.T
    geom = gdf["geometry"]

    # If not passed, build the r-tree spatial index by position for subsequent iloc
    if rtree == None:
        rtree = RTreeIndex()
        for pos, bounds in enumerate(geom.bounds.values):
            rtree.insert(pos, bounds)

    # use r-tree to find possible nearest neighbors, one point at a time,
    # then minimize euclidean distance from point to the possible matches
    nearest_edges = list()
    refdistances = list()
    for xy in zip(X, Y):
        p = Point(xy)
        dists = geom.iloc[list(rtree.nearest(xy, num_results=k))].distance(p)
        ne = geom[dists.idxmin()]
        nearest_edges.append(dists.idxmin())
        refdistances.append(ne.project(p))

    return nearest_edges, refdistances

def get_diameter(G, weight):
    diam = []
    for n, ddict in nx.shortest_path_length(G, weight=weight):
        diam += list(ddict.values())

    return np.amax(diam)


def get_barycentric_node(G):
    "Get node closest to the barycenter of node positions of G."
    pos = nx.get_node_attributes(G, 'pos')
    barycenter = np.mean(np.array(list(pos.values())), axis=0)

    return snap_to_network_nodes(G, barycenter)

def graph_to_matsim_xml(G, fname, length_scale=None, speed=None):
    """
    """
    # open file and write header
    outfile = open(fname, 'w')
    outfile.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>")
    outfile.write("<!DOCTYPE network SYSTEM \"http://www.matsim.org/files/dtd/network_v2.dtd\">")

    # set up element tree
    root = etree.Element('network')
    nodes = etree.SubElement(root, 'nodes')
    links = etree.SubElement(root, 'links')

    # add nodes
    for n, p in G.nodes(data='pos'):
        X = np.array(p)
        if length_scale is not None:
            X = length_scale * X
        attribute_dict = {"id": str(n), "x": str(X[0]), "y": str(X[1])}
        etree.SubElement(nodes, "node", attrib=attribute_dict)

    # add links
    id_counter = 1
    for u, v, ddict in G.to_directed().edges(data=True):
        d = ddict["dist"]
        if speed is None:
            freespeed = ddict["speed"]
        else:
            freespeed = speed
        if length_scale is not None:
            d *= length_scale
        attribute_dict = {"id": str(id_counter),
                          "from": str(u),
                          "to": str(v),
                          "length": str(d),
                          "freespeed": str(freespeed),
                          "modes": "car",
                          "permlanes": str(100),
                          "capacity": str(9999999.0)}
        etree.SubElement(links, "link", attrib=attribute_dict)
        id_counter += 1

    # create a new XML file with the results, the ckose
    etree.ElementTree(root).write(outfile, encoding="unicode")
    outfile.close()



if __name__ == "__main__":
    from models import gabriel, assign_uniform_speeds, taxicab
    np.random.seed(seed=42)
    N = 100
    points = np.random.random((N, 2))
    G = gabriel(points)

    # # test xml writing
    # assign_uniform_speeds(G, 1, 1.5)
    # graph_to_matsim_xml(G, "test.xml")

    # # test volume calculation
    # diam = []
    # for u, ddict in nx.shortest_path_length(G, weight='dist'):
    #         diam += list(ddict.values())
    # diam = np.amax(diam)
    # times =np.linspace(0, 1.1*diam, 20)
    # lbd = np.zeros_like(times)
    # for A in G.nodes():
    #     lbd += volume_growth_at_point(G, A, times, 'dist')
    # total_length = np.sum([d for u, v, d in G.edges(data='dist')])
    # plt.plot(times/diam, lbd/N/total_length, 'o-')
    # plt.show()
    # left, right, ave = sample_volume_growth(G, times, 'dist')
    # plt.plot(times[1:]/diam, left/total_length, 'o-', label="using left sum")
    # plt.plot(times[1:]/diam, right/total_length, 'o-', label="using right sum")
    # plt.plot(times[1:]/diam, ave/total_length, 'o-', label="using average")
    # plt.legend()
    # plt.show()

    # # test ramp function
    # a = np.array([2, 8])
    # b = np.array([4, 9])
    # smax = np.ones_like(a)
    # times = np.linspace(0, 12, 400)
    # y = ramp_function(times, [a,a+smax,b,b+smax], 1)
    # plt.plot(times, y)
    # plt.show()
    # w = 5
    # times = np.linspace(0, 7, 200)
    # f = self_ramp(times, w)
    # plt.plot(times, f)
    # plt.show()

    # # test parallel path lengths
    # lengths = path_lengths_parallel(G, 'dist')
    # dist = dict(nx.shortest_path_length(G, weight='dist'))
    # for n1, ddict in dist.items():
    #     for n2, d in ddict.items():
    #         if d != (lengths[n1])[n2]:
    #             print("Wrong path length.")

    # # test volume growth
    # t = np.linspace(0, 2, 100)
    # lbd, diam, A = volume_growth(G, t, 'dist')
    # plt.plot(t/diam, lbd/A)
    # plt.axhline(y=0, ls='--', c='gray')
    # plt.axhline(y=1, ls='--', c='gray')
    # plt.show()

    # # simple test
    # M = 5
    # P = nx.path_graph(M)
    # a = 0.5
    # dist = {e: a for e in P.edges()}
    # nx.set_edge_attributes(P, dist, 'dist')
    # t = np.linspace(0, M*a, 250)

    # distances = dict(nx.shortest_path_length(P, weight='dist'))
    # plt.figure()
    # ax = plt.gca()
    # for u, v in P.edges():
    #     volume = volume_growth_edge(P, u, v, distances, t, 'dist')
    #     plt.plot(t, volume)
    #     c = ax.lines[-1].get_color()
    #     volume = volume_growth_edge(P, v, u, distances, t, 'dist')
    #     plt.plot(t, volume, c=c, ls='dashed')
    # plt.show()

    # # test parallelized version of vertex based volume growth
    # times = np.linspace(0, 2, 300)
    # lbd_nodes = volume_growth_all_vertices(G, times, 'dist')
    # lbd_edges = volume_growth_uniform_edge_sample(G, 500, times, 'dist')
    # plt.plot(times, lbd_nodes, label='nodes')
    # plt.plot(times, lbd_edges, label='edges')
    # plt.legend()
    # plt.show()

    # # test grid calculation again, measuring performance
    # times = np.linspace(0, 2, 300)
    # v, L = volume_growth_grid_measure_performance(G, 1000, times, 'dist')
    # plt.plot(times, v)
    # plt.show()

    # # test for taxicab
    # m = 10
    # T = taxicab(m)
    # diam = np.sqrt(2)*m
    # times = np.linspace(0, diam, 300)
    # v, L, diam = volume_growth_grid(T/diam, 500, times, 'dist')
    # plt.plot(times, v)
    # plt.show()

    # Nbins = 100
    # max_detour = 5
    # plt.figure()
    # x = np.linspace(1, max_detour, Nbins)
    # y = 2 * x / 3 + 1 / (3*x)
    # plt.plot(x, y, 'k--', label="Euclidean plane")
    # for xp, fp in zip(*pairwise_expected_detours(G, 'dist', 1000)):
    #     y = np.interp(x, xp, fp, left=np.nan, right=np.nan)
    #     plt.plot(x, y, alpha=0.5, lw=0.6)
    # plt.legend()
    # plt.show()

    # # test new wrapper
    # splines, tmax = volume_growth_analysis(G, 'dist', 1000)
    # print(tmax)
    # times = np.linspace(0, tmax, 200)
    # for spl in splines:
    #     lbd = spl(times)
    #     plt.plot(times, lbd/len(splines))
    # plt.show()

    # # test volume
    # splines, L, diam = volume_growth_edge_sample(G, 10, 1000, 500, 'dist')
    # times = np.linspace(0, diam, 200)
    # for spl in splines:
    #     plt.plot(times, spl(times))
    # plt.show()

    # # test volume growth rate calculation
    # Ninter = 50
    # diam = get_diameter(G, 'dist')
    # dmax = diam + 2*np.amax(list(nx.get_edge_attributes(G, 'dist').values()))
    # times = np.linspace(0, dmax, Ninter)
    # A = np.sum([w for u, v, w in G.edges(data='dist')])

    # node_data, edge_data = get_volume_growth_curves(nx.MultiGraph(G), 'dist', times)
    # plt.figure()
    # vol = np.mean([v for v, r in node_data], axis=0)
    # rates = np.mean([r for v, r in node_data], axis=0)
    # plt.plot(times/dmax, vol/A, label="volume")
    # plt.plot(times/dmax, rates/A, label="volume growth rate")
    # plt.plot(times/dmax, np.gradient(vol, times)/A, label="nuermical volume growth rate")
    # plt.legend()
    # plt.show()

    # test equally spaced edge sample
    # P = nx.MultiDiGraph(taxicab((11, 1)))
    # plt.figure()
    # for j, pos in enumerate(equally_spaced_edge_position_sample(P, 50, 'dist').values()):
    #     plt.plot(j+pos, np.zeros_like(pos), 'o')
    # plt.show()
    Nsamples = 400
    edge_pos = equally_spaced_edge_position_sample(nx.MultiDiGraph(G), Nsamples, 'dist')
    node_pos = nx.get_node_attributes(G, 'pos')
    plt.figure()
    plt.axis("equal")
    nx.draw(G, pos=node_pos, node_size=30)
    for edge, pos in edge_pos.items():
        u, v, k = edge
        direction = (node_pos[v] - node_pos[u])
        coords = node_pos[u] + (np.expand_dims(direction, axis=1) * pos).T
        plt.plot(coords[:,0], coords[:,1], 'o')
    plt.show()
