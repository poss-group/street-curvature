from shapely.geometry import Polygon, LineString
import geopandas as gdp
import osmnx as ox
import numpy as np
import networkx as nx
from utils_network import *
from waiting_times import RV_minimal_from_graph, RV_WLC_from_graph

# constant dictionary for mapping highway types with speeds
SPEEDS_POSS = {
    "motorway": 120,
    "trunk": 80,
    "primary": 50,
    "secondary": 50,
    "tertiary": 50,
    "unclassified": 30,
    "residential": 30,
    "service": 20,
    "motorway_link": 120,
    "trunk_link": 80,
    "primary_link": 50,
    "secondary_link": 50,
    "tertiary_link": 50,
    "living_street": 20,
    "pedestrian": 30,
    "road": 30
}

def add_edge_locations(G):
    # project graph
    G_proj = ox.project_graph(G)
    locations = {}
    for u, v, key, ddict in G_proj.edges(data=True, keys=True):
        if "geometry" not in ddict:
            locations[(u, v, key)] = None
        else:
            ls = ddict["geometry"]
            p = np.array(ls.coords)
            dists = ox.distance.euclidean_dist_vec(p[:-1,1],
                                                   p[:-1,0],
                                                   p[1:,1],
                                                   p[1:,0])
            locs = np.cumsum(dists) / np.sum(dists)
            locations[(u, v, key)] = locs[:-1]
    nx.set_edge_attributes(G, locations, name='locations')


# def volume_growth_analysis(G, weight, Ninter):

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    G = ox.graph_from_place("Broitzem", network_type="drive")
    G = nx.convert_node_labels_to_integers(G)
    G = G.to_undirected()
    ox.add_edge_speeds(G, hwy_speeds=SPEEDS_POSS)
    ox.add_edge_travel_times(G)
    A = np.sum([w for u, v, w in G.edges(data='travel_time')])
    add_edge_locations(G)


    # # test volume growth with edge locations
    # d = volume_growth_analysis(G, 'travel_time', 500)
    # times = np.linspace(0, 1.5*d, 400)
    # for spl in nx.get_node_attributes(G, 'volume_growth').values():
    #     plt.plot(times, spl(times)/A, lw=0.7, alpha=0.5)
    # plt.show()
    # for j, spl_list in enumerate(nx.get_edge_attributes(G, 'volume_growth').values()):
    #     c = "C{}".format(j)
    #     for spl in spl_list:
    #         plt.plot(times, spl(times)/A, c=c, lw=0.7, alpha=0.5)
    # plt.show()

    # # compare computation time for simplified vs unsimplified graph
    # import time
    # Gfull = ox.graph_from_place("Broitzem", network_type="drive", simplify=False)
    # Gfull = nx.convert_node_labels_to_integers(Gfull)
    # Gfull = Gfull.to_undirected()
    # ox.add_edge_speeds(Gfull, hwy_speeds=SPEEDS_POSS)
    # ox.add_edge_travel_times(Gfull)
    # Afull = np.sum([w for u, v, w in Gfull.edges(data='travel_time')])
    # add_edge_locations(Gfull)
    # start = time.time()
    # d = volume_growth_analysis(Gfull, 'travel_time', 500)
    # stop = time.time()
    # print("Unsimplified graph: {} s".format(stop-start))
    # start = time.time()
    # d = volume_growth_analysis(G, 'travel_time', 500)
    # stop = time.time()
    # print("Simplified graph: {} s".format(stop-start))

    # # test RV creation
    # RV, d = RV_minimal_from_graph(G, 'travel_time', 500, 10)
    # t = np.linspace(0, d, 300)
    # for N in range(1, 5):
    #     plt.plot(t, RV.pdf(t, N), label="N = {}".format(N))
    # plt.legend()
    # plt.show()

    # test WLC model
    RV, d, v = RV_WLC_from_graph(G, 'travel_time', 500, 16)
    lpvec = np.logspace(-2, 3, 10)
    for N in [2, 4, 8, 16]:
        mean_wt = []
        for lp in lpvec:
            mean_wt.append(RV.mean(N, lp))
        plt.loglog(lpvec, mean_wt, 'o-', label="N = {}".format(N))
    plt.legend()
    plt.show()
