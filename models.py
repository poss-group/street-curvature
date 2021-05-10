import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import nglpy
from scipy.spatial import Delaunay, KDTree
from scipy.spatial.distance import pdist, squareform
from scipy.stats import binned_statistic
from utils import interior_angle
from utils_network import *
from copy import deepcopy


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

def generate_polygons(N, n, sizes, offset):
    A = np.array([np.sqrt(N), np.sqrt(N)]) / 2
    offset = ((2*np.pi*np.arange(offset)) / offset)
    angles = ((2*np.pi*np.arange(n)) / n)
    angles = angles[:,np.newaxis] + offset
    R = np.linspace(2, np.sqrt(N)/2, sizes)
    x = A[0] + np.cos(angles)[:,:,np.newaxis]*R
    y = A[1] + np.sin(angles)[:,:,np.newaxis]*R
    B = np.array([x,y]).T

    return A, B

def route_length_statistic(G, binwidth=0.2):
    N = G.number_of_nodes()
    pos = nx.get_node_attributes(G, 'pos')
    points = np.array(list(pos.values()))
    routes = dict(nx.all_pairs_dijkstra_path_length(G, weight='dist'))
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
        coordinates, and the edge attribute 'dist', which is the
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
    nx.set_edge_attributes(G, distances, 'dist')

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
        coordinates, and the edge attribute 'dist', which is the
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
    nx.set_edge_attributes(G, distances, 'dist')

    return G

def taxicab(m, periodic=False):
    """
    Construct a taxicab (square) lattice.

    Parameters
    ----------
    m : int
        size of the lattice (mxm)
    periodic : bool
        Whether boundaries are periodic.

    Returns
    -------
    G : networkx.Graph
        The lattice represented as a networkx.Graph
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'dist', which is the
        Euclidean distance between nodes.
    """
    G = nx.grid_2d_graph(m, m, periodic=periodic)
    pos = {(i, j): np.array([i, j]) for i, j in G.nodes()}
    nx.set_node_attributes(G, pos, 'pos')
    distances = {(e1, e2): np.linalg.norm(G.nodes[e1]['pos']-G.nodes[e2]['pos'])
                 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'dist')

    return G

def triangular(N, periodic=False):
    """
    Construct a triangular lattice, a lattice whose nodes and edges
    are the triangular tiling of the plane.

    Parameters
    ----------
    m : int
        size of the lattice (mxm)
    periodic : bool
        Whether boundaries are periodic.

    Returns
    -------
    G : networkx.Graph
        The lattice represented as a networkx.Graph
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'dist', which is the
        Euclidean distance between nodes.
    """
    G = nx.triangular_lattice_graph(m, m, periodic=periodic)
    pos = nx.get_node_attributes(G, 'pos')
    factor = np.sqrt(2/np.sqrt(3))
    pos_scaled = {k: factor*np.array(v) for k, v in pos.items()}
    nx.set_node_attributes(G, pos_scaled, 'pos')
    distances = {(e1, e2): np.linalg.norm(G.nodes[e1]['pos']-G.nodes[e2]['pos'])
                 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'dist')

    return G

def hexagonal(N, periodic=False):
    """
    Construct a hexagonal lattice, a lattice whose nodes and edges
    are the hexagonal tiling of the plane.

    Parameters
    ----------
    m : int
        size of the lattice (mxm)
    periodic : bool
        Whether boundaries are periodic.

    Returns
    -------
    G : networkx.Graph
        The lattice represented as a networkx.Graph
        The graph has the node attribute 'pos' with the given point
        coordinates, and the edge attribute 'dist', which is the
        Euclidean distance between nodes.
    """
    G = nx.hexagonal_lattice_graph(m, m, periodic=periodic)
    pos = nx.get_node_attributes(G, 'pos')
    factor = 2 * 3**(-0.75)
    pos_scaled = {k: factor*np.array(v) for k, v in pos.items()}
    nx.set_node_attributes(G, pos_scaled, 'pos')
    distances = {(e1, e2): np.linalg.norm(G.nodes[e1]['pos']-G.nodes[e2]['pos'])
                 for e1, e2 in G.edges()}
    nx.set_edge_attributes(G, distances, 'dist')

    return G

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
        'dist', which are the (Euclidean) distances between connected nodes.
    umin : float
        Minimum speed (left bound of uniform distribution)
    umax : float
        Maximum speed (right bound of uniform distribution)
    """
    M = G.number_of_edges()
    speeds = (umax-umin)*np.random.random(M) + umin
    speeds_dict = dict(zip(list(G.edges()), speeds))
    nx.set_edge_attributes(G, speeds_dict, 'speed')
    dist = nx.get_edge_attributes(G, 'dist')
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
        'dist', which are the (Euclidean) distances between connected nodes.
    edgelist : list
        List of edges to be boosted.
    x : float
        Factor in [0,1] to boost edges by.
    """
    speeds_dict = {e: 1/x if e in edgelist else 1 for e in G.edges()}
    nx.set_edge_attributes(G, speeds_dict, 'speed')
    dist = nx.get_edge_attributes(G, 'dist')
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
        'dist', which are the (Euclidean) distances between connected nodes.
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

if __name__ == "__main__":
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

    # # test circles
    # np.random.seed(seed=250)
    # N = 500
    # Ncircles = 20
    # points = np.sqrt(N)*np.random.random((N, 2))
    # G = gabriel(points)
    # A = get_barycentric_node(G)
    # R = np.linspace(0.2, 1, Ncircles) * np.sqrt(N) / 3
    # circles = get_circles(G, A, R, 'dist')
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.subplot(121)
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                   node_size=20, node_color='black')
    # for C in circles:
    #     Cpos = []
    #     for u, v, lvec in C:
    #         direction = (pos[v] - pos[u]) / G[u][v]['dist']
    #         for l in lvec:
    #             Cpos.append(pos[u]+l*direction)
    #     Cpos = np.array(Cpos)
    #     plt.scatter(Cpos[:,0], Cpos[:,1])
    # plt.title("Circles with distance metric")
    # # circles with speeds
    # boost_random_fraction(G, 0.3, 0.6)
    # R = np.linspace(0.2, 1, Ncircles) * np.sqrt(N) / 3
    # circles = get_circles(G, A, R, 'time')
    # pos = nx.get_node_attributes(G, 'pos')
    # plt.subplot(122)
    # plt.axis("equal")
    # nx.draw_networkx(G, pos=pos, with_labels=False,
    #                   node_size=20, node_color='black')
    # for C in circles:
    #     Cpos = []
    #     for u, v, lvec in C:
    #         direction = (pos[v] - pos[u]) / G[u][v]['time']
    #         for l in lvec:
    #             Cpos.append(pos[u]+l*direction)
    #     Cpos = np.array(Cpos)
    #     plt.scatter(Cpos[:,0], Cpos[:,1])
    # plt.title("Circles with duration metric")
    # plt.show()

    # test subdivide edges
    plt.figure(figsize=(10, 9))
    np.random.seed(seed=220)
    N = 50
    points = np.sqrt(N)*np.random.random((N, 2))
    G = gabriel(points)
    A = get_barycentric_node(G)
    R = np.array([np.sqrt(N) / 4])
    circles = get_circles(G, A, R, 'dist')
    pos = nx.get_node_attributes(G, 'pos')
    plt.subplot(221)
    plt.axis("equal")
    nx.draw_networkx(G, pos=pos, with_labels=False,
                      node_size=20, node_color='black')
    H = deepcopy(G)
    for C in circles:
        Cpos = []
        for u, v, lvec in C:
            direction = (pos[v] - pos[u]) / G[u][v]['dist']
            subdivide_edge(H, u, v, lvec, 'dist')
            for l in lvec:
                Cpos.append(pos[u]+l*direction)
        Cpos = np.array(Cpos)
        plt.scatter(Cpos[:,0], Cpos[:,1])
    plt.title("dist, scatter")
    plt.subplot(222)
    plt.axis("equal")
    posH = nx.get_node_attributes(H, "pos")
    circle_nodes = list(range(N+1, H.number_of_nodes()+1))
    nx.draw_networkx_edges(H, pos=posH, width=0.6)
    nx.draw_networkx_nodes(H, pos=posH, nodelist=circle_nodes,
                       node_color='blue', node_size=20)
    plt.title('dist, subdivide')
    Cpos_subdivide = np.array(list(posH.values()))[N:]
    print(Cpos-Cpos_subdivide)
    # circles with speeds
    boost_random_fraction(G, 0.3, 0.6)
    circles = get_circles(G, A, R, 'time')
    pos = nx.get_node_attributes(G, 'pos')
    plt.subplot(223)
    plt.axis("equal")
    nx.draw_networkx(G, pos=pos, with_labels=False,
                      node_size=20, node_color='black')
    H = deepcopy(G)
    for C in circles:
        Cpos = []
        for u, v, lvec in C:
            direction = (pos[v] - pos[u]) / G[u][v]['time']
            subdivide_edge(H, u, v, lvec, 'time')
            for l in lvec:
                Cpos.append(pos[u]+l*direction)
        Cpos = np.array(Cpos)
        plt.scatter(Cpos[:,0], Cpos[:,1])
    plt.title("time, scatter")
    plt.subplot(224)
    plt.axis("equal")
    posH = nx.get_node_attributes(H, "pos")
    circle_nodes = list(range(N+1, H.number_of_nodes()+1))
    nx.draw_networkx_edges(H, pos=posH, width=0.6)
    nx.draw_networkx_nodes(H, pos=posH, nodelist=circle_nodes,
                       node_color='blue', node_size=20)
    plt.title('time, subdivide')
    plt.show()
    Cpos_subdivide = np.array(list(posH.values()))[N:]
    print(Cpos-Cpos_subdivide)
