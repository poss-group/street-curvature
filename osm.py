from shapely.geometry import Polygon, LineString
import geopandas as gdp
import osmnx as ox
import numpy as np
import networkx as nx
from utils_network import *
from waiting_times import *

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

def get_graph_area(query, which_result=None, buffer_dist=None):
    # create a GeoDataFrame with the spatial boundaries of the place(s)
    if isinstance(query, (str, dict)):
        # if it is a string (place name) or dict (structured place query),
        # then it is a single place
        gdf_place = ox.geocoder.geocode_to_gdf(
            query, which_result=which_result, buffer_dist=buffer_dist
        )
    elif isinstance(query, list):
        # if it is a list, it contains multiple places to get
        gdf_place = ox.geocoder.geocode_to_gdf(query, buffer_dist=buffer_dist)
    else:  # pragma: no cover
        raise TypeError("query must be dict, string, or list of strings")

    # project gdf
    gdf = ox.project_gdf(gdf_place)

    # find polygon
    polygon = gdf["geometry"].unary_union

    return polygon.area

def get_average_speed(G, weight):
    H = G.to_undirected()
    V = np.array(list(nx.get_edge_attributes(H, 'speed_kph').values()))
    w = np.array(list(nx.get_edge_attributes(H, weight).values()))
    V = np.average(V, weights=w) / 3.6

    return V


# def volume_growth_analysis(G, weight, Ninter):

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from volume_growth import *
    G = ox.graph_from_place("Broitzem", network_type="drive")
    G = nx.convert_node_labels_to_integers(G)
    ox.add_edge_speeds(G, hwy_speeds=SPEEDS_POSS)
    ox.add_edge_travel_times(G)
    A = np.sum([w for u, v, w in G.to_undirected().edges(data='travel_time')])

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

    # # test WLC model
    # rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    # RV = RV_WLC(rates, A, 5)
    # Nvec = [2, 4, 8]
    # lpvec = np.logspace(-2, 2, 10)
    # for N in Nvec:
    #     mean_wt = []
    #     for lp in lpvec:
    #         mean_wt.append(RV.mean(N, lp))
    #     plt.loglog(lpvec, mean_wt, 'o-', label="N = {}".format(N))
    # plt.legend()
    # plt.show()

    # # test minimal model
    # rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    # RV = RV_minimal(rates, A, 10)
    # mean_wt = []
    # Nvec = np.arange(1, 15)
    # for N in Nvec:
    #     mean_wt.append(RV.mean(N))
    # plt.plot(Nvec, mean_wt, 'o-')
    # plt.show()

    # # test mean waiting times
    # rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    # Nmax = 10
    # plt.figure()
    # mean_wt = get_mean_waiting_times_minimal(rates, A, Nmax)
    # plt.plot(np.arange(1, Nmax+1), mean_wt,'o-')
    # plt.show()

    # # plot pdfs in WLC model
    # rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    # RV = RV_WLC(rates, A, 5)
    # Nvec = np.arange(2, 10)
    # tp = 50
    # tmax = np.amax([r.x[-1] for r in rates])
    # b = get_upper_limit(tmax, tp)
    # times = np.linspace(0, b, 400)
    # for N in Nvec:
    #     plt.plot(times, RV.pdf(times, N, tp), label="N = {}".format(N))
    # plt.legend()
    # plt.show()

    # test custom waiting time sampling
    rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    RV = RV_minimal(rates, A)
    Nsamples = 10000
    N = 4
    wt = RV.custom_rvs(N, size=Nsamples)
    plt.figure()
    plt.hist(wt, bins=40, density=True)
    tmax = np.amax([r.x[-1] for r in rates])
    t = np.linspace(0, tmax, 200)
    plt.plot(t, RV.pdf(t, N))
    #plt.show()

    # # compare means: sampling vs integration
    # rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    # RV = RV_minimal(rates, A)
    # N = 4
    # mu = RV.mean(N)
    # Nsamples = 2 ** np.arange(1, 11)
    # plt.axhline(y=mu, label="integration")
    # sampled_means = []
    # for size in Nsamples:
    #     wt = RV.custom_rvs(N, size=size)
    #     sampled_means.append(np.mean(wt))
    # plt.loglog(Nsamples, sampled_means, 'o-', label="sampled")
    # plt.show()

    # # redo mean waiting time vs tp plots
    # rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    # RV = RV_minimal(rates, A)
    # Nmax = 16
    # size = 10000
    # distances = RV.sample_distances(Nmax, size)
    # Nvec = [2, 4, 8, 16]
    # lpvec = np.logspace(-2, 2, 100)
    # for N in Nvec:
    #     wt = np.amin(distances[:,:N], axis=1)
    #     wt = inverse_wlc(wt, lpvec)
    #     plt.loglog(lpvec, np.mean(wt, axis=0), 'o-', label="N = {}".format(N))
    # plt.legend()
    # plt.show()

    # # test area calculation
    # print(get_graph_area("Broitzem"))

    # # test speed calculation
    # print(get_average_speed(G, 'travel_time'))

    # contour plot of waiting times
    from scipy.special import loggamma, gammainc
    rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    RV = RV_minimal(rates, A)
    V = get_average_speed(G, 'travel_time')
    Nmax = 16
    size = 5000
    distances = RV.sample_distances(Nmax, size)
    lpvec = RV.t0 * np.logspace(-1, 1, 50)
    wt = np.minimum.accumulate(distances, axis=1)
    wt = inverse_wlc(wt, lpvec)
    mean_wt = np.mean(wt, axis=0)
    plt.figure()
    plt.xlabel(r"$ 2 B {t_p}^2$")
    plt.ylabel(r"Mean waiting time / persistence time")
    b = np.arange(1, Nmax+1) * V**2 / get_graph_area("Broitzem")
    B, Lp = np.meshgrid(b, lpvec, indexing='ij')
    S = 2 * B * Lp**2 * np.pi
    for j in range(Nmax):
        plt.loglog(S[j,:], mean_wt[j,:]/lpvec, 'o-')
    s = np.logspace(np.log(np.amin(S)), np.log(np.amax(S)), 100, base=np.exp(1))
    plane_wt = np.exp(s*(1-np.log(s)) + loggamma(s)) * gammainc(s, s)
    plt.loglog(s, plane_wt, 'k--', label="plane prediction")
    plt.legend()

    plt.figure()
    plt.imshow(np.log(mean_wt/lpvec), origin='lower')

    plt.show()
