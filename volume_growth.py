import numpy as np
from scipy.interpolate import PPoly
import networkx as nx

def boxcar(t, left, right):
    return (t >= left) * (t < right)

def growth_rate_at_node(G, A, weight):
    # calculate shortest path lengths
    dist2A = dict(nx.shortest_path_length(G, source=A, weight=weight))

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    for u, v, k, ddict in G.edges(data=True, keys=True):
        du = dist2A[u]
        w = ddict[weight]
        if ddict['oneway']:
            left.append(du)
            right.append(du + w)
        else:
            dv = dist2A[v]
            dmax = 0.5 * (du + dv + w)
            left.append(du)
            right.append(dmax)
    times = np.sort(np.unique(left+right))
    left = np.array(left)
    right = np.array(right)
    t = np.expand_dims(times, axis=-1)
    rates = np.sum(boxcar(t, left, right), axis=-1)
    print(rates[-1])
    coeffs = np.array([np.zeros_like(rates), rates])
    breakpoints = np.concatenate((times, [times[-1]+1]))

    return PPoly(coeffs, breakpoints)

def growth_rate_at_edge_positions(G, edge, positions, weight):
    edge_data = G.get_edge_data(*edge)
    A = edge[0]
    B = edge[1]

    # calculate shortest path lengths
    dist2B = dict(nx.shortest_path_length(G, source=B, weight=weight))
    wAB = G.get_edge_data(*edge)[weight]

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    if edge_data['oneway']:
        for u, v, k, ddict in G.edges(data=True, keys=True):
            du = dist2B[u] + wAB*(1-positions)
            if edge != (u, v, k):
                w = ddict[weight]
                if ddict['oneway']:
                    left.append(du)
                    right.append(du + w)
                else:
                    dv = dist2B[v] + wAB*(1-positions)
                    dmax = 0.5 * (du + dv + w)
                    left.append(du)
                    right.append(dmax)
            else:
                left.append(np.zeros_like(positions))
                right.append(wAB*(1-positions))
                left.append(du)
                right.append(dist2B[A]+np.ones_like(positions)*wAB)
    else:
        dist2A = dict(nx.shortest_path_length(G, source=A, weight=weight))
        for u, v, k, ddict in G.edges(data=True, keys=True):

            if edge != (u, v, k) and edge != (v, u, k):
                du = np.minimum(dist2A[u] + wAB*positions,
                            dist2B[u] + wAB*(1-positions))
                w = ddict[weight]
                if ddict['oneway']:
                    left.append(du)
                    right.append(du + w)
                else:
                    dv = np.minimum(dist2A[v] + wAB*positions,
                            dist2B[v] + wAB*(1-positions))
                    dmax = 0.5 * (du + dv + w)
                    left.append(du)
                    right.append(dmax)
        left.append(np.zeros_like(positions))
        left.append(np.zeros_like(positions))
        right.append(wAB*positions)
        right.append(wAB*(1-positions))
    interpolators = []
    left = np.array(left)
    right = np.array(right)
    for j in range(positions.size):
        l = left[:,j]
        r = right[:,j]
        times = np.sort(np.unique(np.concatenate((l,r))))
        t = np.expand_dims(times, axis=-1)
        rates = np.sum(boxcar(t, l, r), axis=-1)
        coeffs = np.array([np.zeros_like(rates), rates])
        breakpoints = np.concatenate((times, [times[-1]+1]))
        interpolators.append(PPoly(coeffs, breakpoints))

    return interpolators
