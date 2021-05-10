import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import KDTree
from rtree.index import Index as RTreeIndex
from shapely.geometry import Point, LineString
import pandas as pd
import geopandas as gpd

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
    new_nodes = [(N+i+1, {"pos": pos[u]+l*direction})
                 for i, l in enumerate(positions_on_edge)]
    G.add_nodes_from(new_nodes)
    path = [u] + list(range(N+1, N+M+1)) + [v]
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
    circles : list of tu
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
    nodes = list(G.nodes())
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


def get_barycentric_node(G):
    "Get node closest to the barycenter of node positions of G."
    pos = nx.get_node_attributes(G, 'pos')
    barycenter = np.mean(np.array(list(pos.values())), axis=0)

    return snap_to_network_nodes(G, barycenter)
