import numpy as np
from utils_network import *
from scipy.stats import rv_continuous

def wlc(L, lp):
    return 2*lp*(L-lp*(1-np.exp(-L/lp)))

def RV_minimal_from_graph(G, weight, Ninter, Nmax, weighting='uniform'):
    # calculate diameter to find interpolation times
    diam = get_diameter(G, weight)
    dmax = diam + 2*np.amax(list(nx.get_edge_attributes(G, weight).values()))
    times = np.linspace(0, dmax, Ninter)

    # calculate volume growths
    v_nodes, v_edges = get_volume_growth_curves(G, weight, times)
    volumes = v_nodes
    for e, v in v_edges.items():
        if type(v) == list:
            if len(v) > 0:
                volumes += v

    # calculate total volume
    A = np.sum([w for u, v, w in G.edges(data=weight)])

    # calcualte weights
    if weighting == 'uniform':
        weights = None

    # interpolate CDFs for different N
    CDFinterpolators = {}
    u = 1 - volumes/A
    current_mom = u
    for N in np.arange(1, Nmax+1):
        cdf = 1 - np.average(current_mom, axis=0, weights=weights)
        CDFinterpolators[N] = PchipInterpolator(times, cdf)
        current_mom *= u

    class minimal_model(rv_continuous):
        def _cdf(self, t, N):
            N = N[0]
            spl = CDFinterpolators[N]
            return spl(t)

        def _pdf(self, t, N):
            N = N[0]
            spl = CDFinterpolators[N].derivative()
            return spl(t)

    RV = minimal_model(momtype=0, a=0)
    return RV, dmax

def RV_WLC_from_graph(G, weight, Ninter, Nmax, weighting='uniform', speed='speed_kph'):
    # calculate diameter to find interpolation times
    diam = get_diameter(G, weight)
    dmax = diam + 2*np.amax(list(nx.get_edge_attributes(G, weight).values()))
    times = np.linspace(0, dmax, Ninter)

    # calculate volume growths
    v_nodes, v_edges = get_volume_growth_curves(G, weight, times)
    volumes = v_nodes
    for e, v in v_edges.items():
        if type(v) == list:
            if len(v) > 0:
                volumes += v


    # calculate total volume
    A = np.sum([w for u, v, w in G.edges(data=weight)])

    # calcualte weights
    if weighting == 'uniform':
        weights = None


    # interpolate CDFs for different N
    CDFinterpolators = {}
    u = 1 - volumes/A
    current_mom = u
    for N in np.arange(1, Nmax+1):
        cdf = 1 - np.average(current_mom, axis=0, weights=weights)
        CDFinterpolators[N] = PchipInterpolator(times, cdf)
        current_mom *= u

    # find average speed
    if speed == "speed_kph":
        V = np.array(list(nx.get_edge_attributes(G, speed).values()))
        w = np.array(list(nx.get_edge_attributes(G, weight).values()))
        V = np.average(V, weights=w) / 3.6
    if type(speed) is float:
        V = speed

    class WLC_model(rv_continuous):

        def _cdf(self, t, N, lp):
            N = N[0]
            r = np.sqrt(wlc(t*V, lp))
            spl = CDFinterpolators[N]
            return spl(r)

        def _pdf(self, t, N, lp):
            N = N[0]
            r = np.sqrt(wlc(t*V, lp))
            spl = CDFinterpolators[N].derivative()
            return spl(r) * (lp*V/r) * (1 - np.exp(-(t*V/lp)))

    RV = WLC_model(momtype=0, a=0)
    return RV, dmax, v
