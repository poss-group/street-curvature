import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import KDTree
from rtree.index import Index as RTreeIndex
from shapely.geometry import Point, LineString
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

def snap_to_edge_position(gdf, points, k=3):
    """
    Snap given points in the plane to edges in GeoDataFrame of edges.

    Parameters
    ----------
    G : GeoDataframe
        The edges of spatial network as a Geodataframe.
    points : array of floats, shape (M, 2)
        The cartesian coordinates of the points to be snapped.

    Returns
    -------
    neighbors : integer or array of integers, shape (M,)

    """
    X, Y = points.T
    geom = gdf["geometry"]

    # build the r-tree spatial index by position for subsequent iloc
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
    pos = nx.get_node_attributes(G, 'pos')
    barycenter = np.mean(np.array(list(pos.values())), axis=0)

    return snap_to_network_nodes(G, barycenter)
