import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import binned_statistic
from utils import interior_angle
from utils_network import *
from copy import deepcopy
from nglpy import EmptyRegionGraph as ERG

def polygon_defects(G, Nmax, x, distances, delta=1, sizes=30,
                    offset=16, print_updates=False):
    N = G.number_of_nodes()
    R = np.linspace(2, np.sqrt(N)/2, sizes)
    defects = []
    meanR = []
    for n in range(3, Nmax+1):
        A, B = generate_polygons(N, n, sizes, offset)
        idxA = snap_to_network_nodes(G, A)
        idxB = snap_to_network_nodes(G, B)
        c = distances[idxA, idxB]
        d = distances[idxB, np.roll(idxB, 1, axis=2)]
        c = (np.minimum(c, delta)
             + np.maximum(c-delta, 0))[:,:,:,np.newaxis] * x
        d = (np.minimum(d, delta)
             + np.maximum(d-delta, 0))[:,:,:,np.newaxis] * x
        meanR.append(np.average(c, axis=2))
        alpha = interior_angle(c, np.roll(c, 1, axis=2), d)
        mask1 = d >= (c + np.roll(c, 1, axis=2))
        mask2 = np.abs(c - np.roll(c, 1, axis=2)) > d
        alpha[mask1] = np.pi
        alpha[mask2] = 0
        defects.append(np.average(2*np.pi - np.sum(alpha, axis=2),
                                      axis=1))
        if print_updates == True:
            print("{} of {} edge numbers done.".format(n-2, Nmax-2))

    return np.array(defects), R, np.array(meanR)

def get_polygon_coordinates(A, n, R, offset):
    angles = ((2*np.pi*np.arange(n)) / n) + offset
    x = A[0] + R*np.cos(angles)
    y = A[1] + R*np.sin(angles)
    B = np.array([x,y]).T

    return B



def route_length_statistic(G, binwidth=0.2):
    N = G.number_of_nodes()
    pos = nx.get_node_attributes(G, 'pos')
    points = np.array(list(pos.values()))
    routes = dict(nx.all_pairs_dijkstra_path_length(G, weight='length'))
    dist = pdist(points)
    r = []
    nodes = list(routes.keys())
    for k, d in enumerate(dist):
        i, j = condensed_to_square(k, N)
        r.append(routes[nodes[i]][nodes[j]] / d - 1)
    r = np.array(r)
    bins = np.arange(0.2, np.amax(dist)+binwidth, binwidth)
    rho, bin_edges, binnumber = binned_statistic(dist, r,
                                                 statistic='mean',
                                                 bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2

    return rho, bin_centers

def delaunay(points):
    """
    Construct Delaunay triangulation of set of points.

    Parameters
    ----------
    points : array of floats, shape (N, 2)
        Planar coordinates of points.

    Returns
    -------
    G : networkx.Graph
        The Delaunay triangulation represented as a graph.
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'length', which is the
        Euclidean distance between nodes.
    """
    G = nx.Graph()
    N = points.shape[0]
    pos = {i: points[i] for i in range(N)}
    G.add_nodes_from(pos.keys())
    nx.set_node_attributes(G, pos, 'pos')
    tri = Delaunay(points)
    for triangle in tri.simplices:
        nx.add_cycle(G, triangle)

    distances = {(e1, e2): np.linalg.norm(G.nodes[e1]['pos']-G.nodes[e2]['pos'])
                 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'length')

    return G

def gabriel(points):
    """
    Construct Gabriel graph of a set of points.

    Parameters
    ----------
    points : array of floats, shape (N, 2)
        Planar coordinates of points.

    Returns
    -------
    G : networkx.Graph
        The Gabriel graph.
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'length', which is the
        Euclidean distance between nodes.
    """
    # start with Delaunay triangulation
    G = delaunay(points)
    pos = nx.get_node_attributes(G, 'pos')
    points = np.array([pos[i] for i in pos.keys()])
    tree = KDTree(points)
    # check which edges to remove
    for e in G.edges():
        midpoint = (points[e[0]] + points[e[1]]) / 2
        d, neighbors = tree.query(midpoint, 2)
        if e[0] not in neighbors or e[1] not in neighbors:
            G.remove_edge(e[0], e[1])

    # recalculate distances
    distances = {(e1, e2): np.linalg.norm(G.nodes[e1]['pos']-G.nodes[e2]['pos'])
                 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'length')

    return G

def euclidean_minimum_spanning_tree(points):
    """
    Construct the Euclidean minimum spanning tree (EMST)
    of a set of points.

    Parameters
    ----------
    points : array of floats, shape (N, 2)
        Planar coordinates of points.

    Returns
    -------
    G : networkx.Graph
        The EMST graph.
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'length', which is the
        Euclidean distance between nodes.
    """
    # start with Delaunay triangulation
    G = delaunay(points)
    EMST = nx.minimum_spanning_tree(G, weight='length')

    return EMST


def taxicab(size, periodic=False):
    """
    Construct a taxicab (square) lattice.

    Parameters
    ----------
    size : int, or tuple (m, n) of ints
        size of the lattice (mxn)
    periodic : bool
        Whether boundaries are periodic.

    Returns
    -------
    G : networkx.Graph
        The lattice represented as a networkx.Graph
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'length', which is the
        Euclidean distance between nodes.
    """
    if np.ndim(size) == 0:
        m = size
        n = size
    elif np.ndim(size) == 1:
        m, n = size
    G = nx.grid_2d_graph(m, n, periodic=periodic)
    pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')
    distances = {(e1, e2): 1 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'length')

    return nx.convert_node_labels_to_integers(G)

def beta_skeleton(points, beta, p=2):

    # use nglpy to construct graph
    graph_builder = ERG(beta=beta, p=p)
    graph_builder.build(points)

    # copy to nx.Graph
    G = nx.Graph()
    for u, nb in graph_builder.neighbors().items():
        for v in nb:
            G.add_edge(u, v)

    # add position and distance attributes
    N = points.shape[0]
    pos = {i: points[i] for i in range(N)}
    nx.set_node_attributes(G, pos, 'pos')
    distances = {(e1, e2): np.linalg.norm(G.nodes[e1]['pos']-G.nodes[e2]['pos'])
                 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'length')

    return G


def triangular(size, periodic=False):
    """
    Construct a triangular lattice, a lattice whose nodes and edges
    are the triangular tiling of the plane.

    Parameters
    ----------
    size : int, or tuple (m, n) of ints
        size of the lattice (mxn)
    periodic : bool
        Whether boundaries are periodic.

    Returns
    -------
    G : networkx.Graph
        The lattice represented as a networkx.Graph
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'length', which is the
        Euclidean distance between nodes.
    """
    if np.ndim(size) == 0:
        m = size
        n = size
    elif np.ndim(size) == 1:
        m, n = size
    G = nx.triangular_lattice_graph(m, n, periodic=periodic)
    # pos = nx.get_node_attributes(G, 'pos')
    # factor = np.sqrt(2/np.sqrt(3))
    # pos_scaled = {k: factor*np.array(v) for k, v in pos.items()}
    # nx.set_node_attributes(G, pos_scaled, 'pos')
    distances = {(e1, e2): 1 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'length')

    return nx.convert_node_labels_to_integers(G)

def hexagonal(size, periodic=False):
    """
    Construct a hexagonal lattice, a lattice whose nodes and edges
    are the hexagonal tiling of the plane.

    Parameters
    ----------
    size : int, or tuple (m, n) of ints
        size of the lattice (mxn)
    periodic : bool
        Whether boundaries are periodic.

    Returns
    -------
    G : networkx.Graph
        The lattice represented as a networkx.Graph
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'length', which is the
        Euclidean distance between nodes.
    """
    if np.ndim(size) == 0:
        m = size
        n = size
    elif np.ndim(size) == 1:
        m, n = size
    G = nx.hexagonal_lattice_graph(m, n, periodic=periodic)
    # pos = nx.get_node_attributes(G, 'pos')
    # factor = 2 * 3**(-0.75)
    # pos_scaled = {k: factor*np.array(v) for k, v in pos.items()}
    # nx.set_node_attributes(G, pos_scaled, 'pos')
    distances = {(e1, e2): 1 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'length')

    return nx.convert_node_labels_to_integers(G)

def taxicab_with_subgrid(size, spacing, x, periodic=False):
    """
    Construct a taxicab (square) lattice with a faster subgrid.

    Parameters
    ----------
    size : int, or tuple (m, n) of ints
        size of the lattice (mxn)
    spacing : int, or tuple (Lx, Ly) of ints
        spacing of the faster subgrid
    x : float
        Boost parameter, speed is 1/x.
    periodic : bool
        Whether boundaries are periodic.

    Returns
    -------
    G : networkx.Graph
        The lattice represented as a networkx.Graph
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'length', which is the
        Euclidean distance between nodes.
    """
    if np.ndim(size) == 0:
        m = size
        n = size
    elif np.ndim(size) == 1:
        m, n = size
    if np.ndim(spacing) == 0:
        Lx = spacing
        Ly = spacing
    elif np.ndim(spacing) == 1:
        Lx, Ly = spacing
    G = nx.grid_2d_graph(m, n, periodic=periodic)
    pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')
    distances = {(e1, e2): 1 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'length')

    is_on_subgrid = lambda n : True if (n[0]%Lx == 0 or n[1]%Ly == 0) else False
    edge_speed = lambda u, v : 1/x if (is_on_subgrid(u) and is_on_subgrid(v)) else 1
    speeds = {(u, v): edge_speed(u, v) for u, v in G.edges()}
    nx.set_edge_attributes(G, speeds, 'speed')
    times = {e: (distances[e] / speeds[e]) for e in G.edges()}
    nx.set_edge_attributes(G, times, 'time')

    return nx.convert_node_labels_to_integers(G)

def taxicab_village_grid(m, n, size, spacing, boost):
    G = nx.Graph()
    inter_village_roads = []
    ddict = {'length': spacing-2*size, 'speed': 1/boost, 'time': boost*(spacing-2*size)}
    for i in range(m):
        for j in range(n):
            village = nx.grid_2d_graph(2*size+1, 2*size+1)
            rename = {v: (v[0]+i*spacing, v[1]+j*spacing) for v in village.nodes()}
            nx.relabel_nodes(village, rename, copy=False)
            G = nx.compose(G, village)
            inter_village_roads += [((i*spacing+size, k*spacing+2*size),
                                     (i*spacing+size, (k+1)*spacing),
                                     ddict) for k in range(n-1)]
            inter_village_roads += [((k*spacing+2*size, j*spacing+size),
                                     ((k+1)*spacing, j*spacing+size),
                                      ddict) for k in range(m-1)]

    pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')

    nx.set_edge_attributes(G, 1, 'length')
    nx.set_edge_attributes(G, 1, 'speed')
    nx.set_edge_attributes(G, 1, 'time')

    G.add_edges_from(inter_village_roads)

    return nx.convert_node_labels_to_integers(G)

def assign_uniform_speeds(G, umin, umax):
    """
    Assign uniformly distributed speeds to the edges of a street network
    model. The speeds are stored as a new edge attribute 'speed', and are
    used to calculate a new edge attribute 'time', calculated as
    time = dist / speed for each edge.

    Parameters
    ----------
    G : networkx.Graph
        Graph representing the street network. Needs to have an edge attribute
        'length', which are the (Euclidean) distances between connected nodes.
    umin : float
        Minimum speed (left bound of uniform distribution)
    umax : float
        Maximum speed (right bound of uniform distribution)
    """
    M = G.number_of_edges()
    speeds = (umax-umin)*np.random.random(M) + umin
    speeds_dict = dict(zip(list(G.edges()), speeds))
    nx.set_edge_attributes(G, speeds_dict, 'speed')
    dist = nx.get_edge_attributes(G, 'length')
    times = {e: dist[e] / speeds_dict[e] for e in G.edges()}
    nx.set_edge_attributes(G, times, 'time')

def boost_edges(G, edgelist, x):
    """
    Boost a list of edges, by the factor x. The speeds are stored
    as a new edge attribute 'speed', and are used to calculate a new
    edge attribute 'time', calculated as time = dist / speed for each edge.

    Parameters
    ----------
    G : networkx.Graph
        Graph representing the street network. Needs to have an edge attribute
        'length', which are the (Euclidean) distances between connected nodes.
    edgelist : list
        List of edges to be boosted.
    x : float
        Factor in [0,1] to boost edges by.
    """
    speeds_dict = {e: 1/x if e in edgelist else 1 for e in G.edges()}
    nx.set_edge_attributes(G, speeds_dict, 'speed')
    dist = nx.get_edge_attributes(G, 'length')
    times = {e: dist[e] / speeds_dict[e] for e in G.edges()}
    nx.set_edge_attributes(G, times, 'time')

def boost_random_fraction(G, phi, x):
    """
    Boost a random fraction of edges, by the factor x. The speeds are stored
    as a new edge attribute 'speed', and are used to calculate a new
    edge attribute 'time', calculated as time = dist / speed for each edge.

    Parameters
    ----------
    G : networkx.Graph
        Graph representing the street network. Needs to have an edge attribute
        'length', which are the (Euclidean) distances between connected nodes.
    phi : float
        Fraction of edges to boost
    x : float
        Factor in [0,1] to boost edges by.

    Returns
    -------
    actual_fraction : float
        The actual fraction of boosted edges. In a graph with a total
        number E of edges, this is floor(phi*M) / M.
    """
    M = G.number_of_edges()
    m = int(np.floor(phi*M))
    actual_fraction = m / M
    E = list(G.edges())
    choice = [E[i] for i in np.random.choice(M, m)]
    boost_edges(G, choice, x)

    return actual_fraction

def upgrade_to_ox_graph(G):
    """
    Converts undirected simple graph to MultiDiGraph and adds attributes.

    Parameters
    ----------
    G : nx.Graph
        The graph to be converted.
    """
    # No streets are oneway in undirected graph
    nx.set_edge_attributes(G, False, name='oneway')
    nx.set_edge_attributes(G, 1, name='speed')

    return nx.MultiDiGraph(G)

def boost_path(G, path, x):
    u = path[:-1]
    v = path[1:]
    edge_list = list(zip(u, v))
    boost_edges(G, edge_list, x)

if __name__ == "__main__":
    np.random.seed(seed=42)
    # test polygon snapping
    # N = 500
    # n = 5
    # G = gabriel(N)
    # pos = nx.get_node_attributes(G, 'pos')
    # A, B = generate_polygons(N, n, 3, 4)
    # idxA = snap_to_network_nodes(G, A)
    # idxB = snap_to_network_nodes(G, B)
    # fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    # for i, row in enumerate(axes):
    #     for j, ax in enumerate(row):
    #         ax.axis("equal")
    #         nx.draw_networkx_edges(G, pos, ax=ax)
    #         ax.scatter(B[i,j,:,0], B[i,j,:,1], color='b')
    #         ax.scatter(A[0], A[1], color='b')
    #         nx.draw_networkx_nodes(G, pos, ax=ax,
    #                                nodelist=idxB[i,j,:],
    #                                node_color='r',
    #                                node_size=50*np.ones(n))
    # plt.show()

    # # test route length statistics
    # G = taxicab(10)
    # rho, d = route_length_statistic(G, binwidth=1)
    # plt.plot(d, rho)
    # plt.show()

    # # test uniform speeds
    # N = 200
    # points = np.random.random((N, 2))
    # G = gabriel(points)
    # assign_uniform_speeds(G, 1, 1.5)
    # pos = nx.get_node_attributes(G, 'pos')
    # speed = nx.get_edge_attributes(G, 'speed')
    # el = list(speed.keys())
    # speed = list(speed.values())
    # plt.figure(figsize=(6.6,6))
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False, edgelist=el,
    #                  node_size=40, edge_color=speed)
    # plt.tight_layout()
    # plt.show()

    # # test boosting
    # boost_random_fraction(G, 0.5, 0.6)
    # speed = nx.get_edge_attributes(G, 'speed')
    # el = list(speed.keys())
    # speed = list(speed.values())
    # plt.figure(figsize=(6.6,6))
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False, edgelist=el,
    #                  node_size=40, edge_color=speed)
    # plt.tight_layout()
    # plt.show()

    # # Gabriel graph of grid points
    # x = np.arange(6)
    # X, Y = np.meshgrid(x, x)
    # points = np.array([X.flatten(), Y.flatten()]).T
    # G = gabriel(points)
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.figure(figsize=(6.6,6))
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                  node_size=40)
    # plt.tight_layout()
    # plt.show()

    # # barycentric node
    # G = taxicab(6)
    # A = get_barycentric_node(G)
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.figure(figsize=(6.6,6))
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                  node_size=40)
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=[A], node_size=60,
    #                        node_color='red')
    # plt.tight_layout()
    # plt.show()

    # # snap to edge
    # N = 400
    # points = np.random.random((N, 2))
    # G = gabriel(points)
    # locs = np.random.random((10, 2))
    # gdf = graph_to_gdf(G)
    # ne, refdist = snap_to_edge_position(gdf, locs)
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.figure(figsize=(6.6,6))
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                  node_size=40)
    # for j, p in enumerate(locs):
    #     line = gdf["geometry"][ne[j]]
    #     psnapped = line.interpolate(refdist[j])
    #     plt.scatter([p[0], psnapped.x], [p[1], psnapped.y])

    # plt.tight_layout()
    # plt.show()

    # # snapping a polygon
    # np.random.seed(seed=250)
    # N = 500
    # points = np.sqrt(N)*np.random.random((N, 2))
    # G = gabriel(points)
    # A = get_barycentric_node(G)
    # n = 6
    # R = np.sqrt(N) / 3
    # angles = ((2*np.pi*np.arange(n)) / n)
    # x = points[A,0] + np.cos(angles)*R
    # y = points[A,1] + np.sin(angles)*R
    # B = np.array([x,y]).T
    # gdf = graph_to_gdf(G)
    # # first with one nearest edge
    # ne, refdist = snap_to_edge_position(gdf, B, k=1)
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.subplot(121)
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                  node_size=40, node_color='black')
    # B_snapped = []
    # for j, e in enumerate(ne):
    #     line = gdf["geometry"][e]
    #     psnapped = line.interpolate(refdist[j])
    #     B_snapped.append([psnapped.x, psnapped.y])
    # B_snapped = np.array(B_snapped)
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=[A], node_size=60,
    #                         node_color='red')
    # plt.scatter(B[:,0], B[:,1], color='orange')
    # plt.scatter(B_snapped[:,0], B_snapped[:,1], color='red', zorder=3)
    # # then with more
    # ne, refdist = snap_to_edge_position(gdf, B, k=3)
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.subplot(122)
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                  node_size=40, node_color='black')
    # B_snapped = []
    # for j, e in enumerate(ne):
    #     line = gdf["geometry"][e]
    #     psnapped = line.interpolate(refdist[j])
    #     B_snapped.append([psnapped.x, psnapped.y])
    # B_snapped = np.array(B_snapped)
    # nx.draw_networkx_nodes(G, pos=pos, nodelist=[A], node_size=60,
    #                         node_color='red')
    # plt.scatter(B[:,0], B[:,1], color='orange')
    # plt.scatter(B_snapped[:,0], B_snapped[:,1], color='red', zorder=3)
    # plt.tight_layout()
    # plt.show()

    # test circles
    import matplotlib.cm as cm
    np.random.seed(seed=250)
    N = 500
    Ncircles = 20
    colors = cm.viridis(np.linspace(0, 1, Ncircles))
    points = np.sqrt(N)*np.random.random((N, 2))
    G = gabriel(points)
    A = get_barycentric_node(G)
    R = np.linspace(0.2, 1, Ncircles) * np.sqrt(N) / 3
    circles = get_circles(G, A, R, 'length')
    pos = nx.get_node_attributes(G, 'pos')
    plt.subplot(121)
    plt.axis("equal")
    nx.draw_networkx(G, pos=pos, with_labels=False,
                      node_size=20, node_color='black')
    for j, C in enumerate(circles):
        Cpos = []
        for u, v, lvec in C:
            direction = (pos[v] - pos[u]) / G[u][v]['length']
            for l in lvec:
                Cpos.append(pos[u]+l*direction)
        Cpos = np.array(Cpos)
        plt.scatter(Cpos[:,0], Cpos[:,1], color=colors[j])
    plt.title("Circles with distance metric")
    # circles with speeds
    boost_random_fraction(G, 0.3, 0.6)
    R = np.linspace(0.2, 1, Ncircles) * np.sqrt(N) / 3
    circles = get_circles(G, A, R, 'time')
    pos = nx.get_node_attributes(G, 'pos')
    plt.subplot(122)
    plt.axis("equal")
    nx.draw_networkx(G, pos=pos, with_labels=False,
                      node_size=20, node_color='black')
    for j, C in enumerate(circles):
        Cpos = []
        for u, v, lvec in C:
            direction = (pos[v] - pos[u]) / G[u][v]['time']
            for l in lvec:
                Cpos.append(pos[u]+l*direction)
        Cpos = np.array(Cpos)
        plt.scatter(Cpos[:,0], Cpos[:,1], color=colors[j])
    plt.title("Circles with duration metric")
    plt.show()

    # # test subdivide edges
    # plt.figure(figsize=(10, 9))
    # np.random.seed(seed=220)
    # N = 50
    # points = np.sqrt(N)*np.random.random((N, 2))
    # G = gabriel(points)
    # A = get_barycentric_node(G)
    # R = np.array([np.sqrt(N) / 4])
    # circles = get_circles(G, A, R, 'length')
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.subplot(221)
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                   node_size=20, node_color='black')
    # H = deepcopy(G)
    # for C in circles:
    #     Cpos = []
    #     for u, v, lvec in C:
    #         direction = (pos[v] - pos[u]) / G[u][v]['length']
    #         subdivide_edge(H, u, v, lvec, 'length')
    #         for l in lvec:
    #             Cpos.append(pos[u]+l*direction)
    #     Cpos = np.array(Cpos)
    #     plt.scatter(Cpos[:,0], Cpos[:,1])
    # plt.title("dist, scatter")
    # plt.subplot(222)
    # plt.axis("equal")
    # posH = nx.get_node_attributes(H, "pos")
    # circle_nodes = list(range(N+1, H.number_of_nodes()+1))
    # nx.draw_networkx_edges(H, pos=posH, width=0.6)
    # nx.draw_networkx_nodes(H, pos=posH, nodelist=circle_nodes,
    #                    node_color='blue', node_size=20)
    # plt.title('dist, subdivide')
    # Cpos_subdivide = np.array(list(posH.values()))[N:]
    # print(Cpos-Cpos_subdivide)
    # # circles with speeds
    # boost_random_fraction(G, 0.3, 0.6)
    # circles = get_circles(G, A, R, 'time')
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.subplot(223)
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                   node_size=20, node_color='black')
    # H = deepcopy(G)
    # for C in circles:
    #     Cpos = []
    #     for u, v, lvec in C:
    #         direction = (pos[v] - pos[u]) / G[u][v]['time']
    #         subdivide_edge(H, u, v, lvec, 'time')
    #         for l in lvec:
    #             Cpos.append(pos[u]+l*direction)
    #     Cpos = np.array(Cpos)
    #     plt.scatter(Cpos[:,0], Cpos[:,1])
    # plt.title("time, scatter")
    # plt.subplot(224)
    # plt.axis("equal")
    # posH = nx.get_node_attributes(H, "pos")
    # circle_nodes = list(range(N+1, H.number_of_nodes()+1))
    # nx.draw_networkx_edges(H, pos=posH, width=0.6)
    # nx.draw_networkx_nodes(H, pos=posH, nodelist=circle_nodes,
    #                    node_color='blue', node_size=20)
    # plt.title('time, subdivide')
    # plt.show()
    # Cpos_subdivide = np.array(list(posH.values()))[N:]
    # print(Cpos-Cpos_subdivide)

    # # test EMST
    # np.random.seed(seed=42)
    # N = 100
    # points = np.random.random((N, 2))
    # G = euclidean_minimum_spanning_tree(points)
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.figure(figsize=(10, 9))
    # plt.axis("equal")
    # nx.draw(G, pos=pos, node_size=40)
    # plt.show()

    # # compare two methods to construct Gabriel graph
    # np.random.seed(seed=42)
    # N = 100
    # points = np.random.random((N, 2))
    # G1 = gabriel(points)
    # pos = nx.get_node_attributes(G1, 'pos')
    # G2 = beta_skeleton(points, 1)
    # plt.subplot(121)
    # plt.axis("equal")
    # plt.title("Own Gabriel graph implementation")
    # nx.draw(G1, pos=nx.get_node_attributes(G1, 'pos'), node_size=40)
    # plt.subplot(122)
    # plt.axis("equal")
    # plt.title("Nglpy's Gabriel graph implementation")
    # nx.draw(G2, pos=nx.get_node_attributes(G2, 'pos'), node_size=40)
    # plt.tight_layout()
    # plt.show()

    # # test taxicab with subgrid
    # G = taxicab_with_subgrid(13, 3, 0.5)
    # plt.figure(figsize=(10, 9))
    # plt.axis("equal")
    # u, v, speeds = zip(*G.edges(data='speed'))
    # el = list(zip(u, v))
    # c = list(speeds)
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw_networkx_nodes(G, pos=pos, node_size=30)
    # nx.draw_networkx_edges(G, pos=pos, edgelist=el, edge_color=c)
    # plt.show()

    # # test village grid
    # G = taxicab_village_grid(3, 2, 2, 10, 0.5)
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.figure()
    # plt.axis("equal")
    # u, v, speeds = zip(*G.edges(data='speed'))
    # el = list(zip(u, v))
    # c = list(speeds)
    # pos = nx.get_node_attributes(G, 'pos')
    # nx.draw_networkx_nodes(G, pos=pos, node_size=30)
    # nx.draw_networkx_edges(G, pos=pos, edgelist=el, edge_color=c)
    # plt.show()
