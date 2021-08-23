from shapely.geometry import Polygon, LineString
import geopandas as gdp
import osmnx as ox
import numpy as np
import networkx as nx
from utils_network import *
from waiting_times import *
from matplotlib.colors import ListedColormap

# constant dictionary for mapping road classes with speeds
SPEEDS_POSS = {
    1: 120,
    2: 80,
    3: 50,
    4: 50,
    5: 50,
    6: 30,
    7: 20,
    8: 20
}

ROAD_CLASSES = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "unclassified": 6,
    "residential": 6,
    "service": 7,
    "motorway_link": 1,
    "trunk_link": 2,
    "primary_link": 3,
    "secondary_link": 4,
    "tertiary_link": 5,
    "living_street": 8,
    "pedestrian": 8,
    "road": 6
}

roadclass_cmap = ListedColormap(["gray", "yellow", "purple", "orange", "red", "black", "blue", "green"], name="roadclass")

def get_hwy_filter(road_classes):
    hwy_filter = '[\"highway\"~\"'
    hwy_types = []
    for tag, cls in ROAD_CLASSES.items():
        if cls in road_classes:
            hwy_types.append(tag)
    tags = '|'.join(hwy_types)
    hwy_filter += tags
    hwy_filter += '\"]'

    return hwy_filter


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

def prepare_graph(G):
    """
    Trims to strongly connected graph, computes travel times and
    converts node labels.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The street network obtained with OSMnx

    Returns
    -------
    G : nx.MultiDiGraph
        Copy of the graph
    """
    # remove all but largest strongly connected component
    largest = max(nx.strongly_connected_components(G), key=len)
    G.remove_nodes_from(set(G.nodes())-largest)

    # collapse highway types into road classes
    road_classes = {}
    for u, v, k, tag in G.edges(data='highway', keys=True):
        if isinstance(tag, list):
            cls = min([ROAD_CLASSES[t] for t in tag])
        else:
            cls = ROAD_CLASSES[tag]
        road_classes[(u, v, k)] = cls
    nx.set_edge_attributes(G, road_classes, name='roadclass')

    # calculate speeds and travel times
    speeds = {edge: SPEEDS_POSS[cls] for edge, cls in road_classes.items()}
    nx.set_edge_attributes(G, speeds, name='speed_kph')
    ox.add_edge_travel_times(G)

    return nx.convert_node_labels_to_integers(G, label_attribute='OSM_ID')

# def volume_growth_analysis(G, weight, Ninter):

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from volume_growth import *
    G = ox.graph_from_place("Broitzem", network_type="drive")
    G = prepare_graph(G)

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

    # # test custom waiting time sampling
    # rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    # RV = RV_minimal(rates)
    # Nsamples = 10000
    # N = 4
    # wt = RV.custom_rvs(N, size=Nsamples)
    # plt.figure()
    # plt.hist(wt, bins=40, density=True)
    # tmax = np.amax([r.x[-1] for r in rates])
    # t = np.linspace(0, tmax, 200)
    # plt.plot(t, RV.pdf(t, N))
    # plt.show()

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

    # # contour plot of waiting times
    # from scipy.special import loggamma, gammainc
    # rates = [growth_rate_at_node(G, n, 'travel_time') for n in G.nodes()]
    # RV = RV_minimal(rates, A)
    # V = get_average_speed(G, 'travel_time')
    # Nmax = 16
    # size = 5000
    # distances = RV.sample_distances(Nmax, size)
    # lpvec = RV.t0 * np.logspace(-1, 1, 50)
    # wt = np.minimum.accumulate(distances, axis=1)
    # wt = inverse_wlc(wt, lpvec)
    # mean_wt = np.mean(wt, axis=0)
    # plt.figure()
    # plt.xlabel(r"$ 2 B {t_p}^2$")
    # plt.ylabel(r"Mean waiting time / persistence time")
    # b = np.arange(1, Nmax+1) * V**2 / get_graph_area("Broitzem")
    # B, Lp = np.meshgrid(b, lpvec, indexing='ij')
    # S = 2 * B * Lp**2 * np.pi
    # for j in range(Nmax):
    #     plt.loglog(S[j,:], mean_wt[j,:]/lpvec, 'o-')
    # s = np.logspace(np.log(np.amin(S)), np.log(np.amax(S)), 100, base=np.exp(1))
    # plane_wt = np.exp(s*(1-np.log(s)) + loggamma(s)) * gammainc(s, s)
    # plt.loglog(s, plane_wt, 'k--', label="plane prediction")
    # plt.legend()

    # plt.figure()
    # plt.imshow(np.log(mean_wt/lpvec), origin='lower')

    # # test fitting
    # tpvec = RV.t0 * np.logspace(-0.5, 0.5, 3)
    # RV2 = RV_WLC(rates, A)
    # for tp in tpvec:
    #     rvs = inverse_wlc(wt, tp)
    #     t = np.linspace(0, rvs.max(), 200)
    #     plt.hist(rvs, bins=50, density=True)
    #     plt.plot(t, RV2.pdf(t, N, tp))
    #     plt.show()
    #     # tp_fit = RV2.fit(rvs[:500], RV.t0, floc=0, fscale=1, fN=4)
    #     # print("tp = {}, fitted: {}".format(tp, tp_fit))

    # # wrong A in the PPoly??
    # print("A = {}".format(A))
    # print("The PPolys give me:")
    # for r in rates:
    #     Arate = r.antiderivative()(r.x[-1])
    #     print("\t {}".format(Arate))

    # # another value of the total volume
    # Anew = get_total_volume(G, 'travel_time')
    # print("Anew = {}".format(Anew))

    # # test new version of equally spaced edge sample
    # print(get_total_volume(G, 'length'))
    # edge_pos = equally_spaced_edge_position_sample(G, 200, 'length')

    # test new average VOLUME calculation
    rates = volume_growth(G, 'travel_time', 500, 'length')
    A = get_total_volume(G, 'travel_time')
    tmax = np.amax([r.x[-1] for r in rates])
    t = np.linspace(0, tmax, 400)
    plt.figure()
    for r in rates:
        v = r.antiderivative()(t)
        plt.plot(t, v/A)
    plt.show()
