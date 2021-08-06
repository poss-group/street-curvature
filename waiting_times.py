import numpy as np
from utils_network import *
from scipy.stats import rv_continuous
from scipy.interpolate import CubicHermiteSpline
from volume_growth import *
from scipy.optimize import newton

def poly_convolve(A, B):
    kA = A.shape[0]
    kB = B.shape[0]
    m = A.shape[1]
    A_ = np.zeros((m, kA+kB-1), dtype=A.dtype)
    A_[:, :kA] = A.T
    B_ = np.zeros((m, kA+kB-1), dtype=B.dtype)
    B_[:, :kB] = B.T
    C_fft = np.fft.fft(A_, axis=1) * np.fft.fft(B_, axis=1)

    return np.real(np.fft.ifft(C_fft)).T

def average_weight(rate):
    return np.sum(rate.c[1,:] *
                  (rate.x[1:]**2 - rate.x[:-1]**2))

def wlc(L, lp):
    """
    Mean-square end-to-end distance of a worm-like chain.

    Parameters
    ----------
    L : array-like
        Length of the chain.
    lp : array-like
        Persistence length. Must be broadcastable to the same shape as L.

    Returns
    -------
    meanRsquared : array-like
        The mean-square end-to-end distance.
    """
    return 2*lp*(L-lp*(1-np.exp(-L/lp)))

def inverse_wlc(r, lp):
    """
    Given a RMS end-to-end distance of a worm-like chain, return its chain length.

    Parameters
    ----------
    r : array-like
        RMS end-to-end distance.
    lp : array-like
        Persistence length. Must be broadcastable to the same shape as L.

    Returns
    -------
    L : array-like
        Length of the chain
    """
    if np.ndim(lp) > 0:
        r = np.expand_dims(r, axis=-1) * np.ones_like(lp)
    f = lambda t : wlc(t, lp) - r**2
    f1 = lambda t : 2*lp*(1-np.exp(-t/lp))
    f2 = lambda t: 2*np.exp(-t/lp)
    return newton(f, r**2, fprime=f1, fprime2=f2)

def get_upper_limit(tmax, tp):
    f = lambda t : wlc(t, tp) - tmax**2
    fprime = lambda t : 2*tp*(1-np.exp(-t/tp))
    return newton(f, tmax**2, fprime=fprime)

def draw_distances(rate, size):
    """
    Draw distance according to a piecewise constant pdf.

    Parameters
    ----------
    rate : scipy.interpolate.PPoly object
        The pdf as a piecewise polynomial. Must have order 1.
    size : int or tuple of ints
        Shape of the sample.

    Returns
    -------
    distances : array, shape size
        The sampled distances.
    """
    # determine weights of individual segments
    Nsteps = rate.x.size - 1
    step_widths = rate.x[1:] - rate.x[:-1]
    w = rate.c[1,:] * step_widths
    w /= np.sum(w)

    # draw a step and then a location on that steo
    steps = np.random.choice(Nsteps, size=size, p=w)
    location_on_step = np.random.random(size)
    distances = rate.x[steps] + step_widths[steps] * location_on_step

    return distances

def get_single_mean_waiting_times_minimal(r, A, Nmax):
    tmax = r.x[-1]
    identity = r.x[:-1]
    identity = np.array([np.ones_like(identity), identity])

    Uc = - r.antiderivative().c[1:,:] / A
    Uc[1,:] += 1

    # iteratively determine PDF PPolys and integrate
    mean_wt = []
    PDF = poly_convolve(r.c/A, identity)
    N = 1
    while N <= Nmax:
        mean_wt.append(N * PPoly(PDF, r.x).integrate(0, tmax))
        N += 1
        PDF = poly_convolve(PDF, Uc)

    return mean_wt

def get_mean_waiting_times_minimal(rates, A, Nmax, weights=None):
    return np.average([get_single_mean_waiting_times_minimal(r, A, Nmax) for r in rates], weights=weights, axis=0)


def RV_minimal(rates, A, weights=None):
    """
    Build a minimal model waiting time random variable from a set of volume growth rates.

    Parameters
    ----------
    rates : list of scipy.interpolate.PPoly instances
        The volume growth rates as piecewise polynomials.
    A : float
        Total volume of the graph.
    weights : array-like, optional
        Weights used for weighted average. Passed directly to numpy.average.
        If `weights=None`, uniform weights are assumed.

    Returns
    -------
    RV : Subclass of scipy.stats.rv_continuous
        The waiting time random variable. The shape parameter of the distribution
        is `N`, the number of buses.
    """
    # get volumes from growth rates
    volumes = []
    for r in rates:
        volumes.append(r.antiderivative())

    # find maximum waiting time
    tmax = np.amax([r.x[-1] for r in rates])

    # # calculate mean volume (growth rate)
    # x = np.linspace(0, tmax, Nknots)
    # y = np.average([v(x) for v in volumes], weights=weights, axis=0)
    # dydx = np.average([r(x) for r in rates], weights=weights, axis=0)
    # spl = CubicHermiteSpline(x, y, dydx)

    class minimal_model(rv_continuous):
        def __init__(self, **kwargs):
            self.tmax = np.amax([r.x[-1] for r in rates])
            self.t0 = np.average([average_weight(r)/A for r in rates], weights=weights)
            super().__init__(**kwargs)

        def custom_rvs(self, N, size=1):
            # get total number of samples
            Nsamples = np.prod(size)

            # choose pick-up locations randomly
            locations = np.random.choice(len(rates), size=Nsamples, p=weights)
            u, counts = np.unique(locations, return_counts=True)

            # draw distances
            distances = np.concatenate([draw_distances(rates[i], (s, N)) for i, s in zip(u, counts)], axis=0)

            # waiting times are minima over N distances
            wt = np.amin(distances, axis=1)
            return wt

        def sample_distances(self, Nmax, size=1):
            # get total number of samples
            Nsamples = np.prod(size)

            # choose pick-up locations randomly
            locations = np.random.choice(len(rates), size=Nsamples, p=weights)
            u, counts = np.unique(locations, return_counts=True)

            # draw distances
            distances = np.concatenate([draw_distances(rates[i], (s, Nmax)) for i, s in zip(u, counts)], axis=0)

            return distances

        def _cdf(self, t, N):
            return np.average([1 - (1-v(t)/A)**N for v in volumes],
                              axis=0, weights=weights)

        def _pdf(self, t, N):
            return np.average([N * (1-v(t)/A)**(N-1) * r(t) / A for r, v in zip(rates, volumes)],
                              axis=0, weights=weights)

    RV = minimal_model(momtype=0, a=0, b=tmax)
    return RV

def RV_WLC(rates, A, Nknots, weights=None):
    """
    Build a WLC model waiting time random variable from a set of volume growth rates.

    Parameters
    ----------
    rates : list of scipy.interpolate.PPoly instances
        The volume growth rates as piecewise polynomials.
    A : float
        Total volume of the graph.
    weights : array-like, optional
        Weights used for weighted average. Passed directly to numpy.average.
        If `weights=None`, uniform weights are assumed.

    Returns
    -------
    RV : Subclass of scipy.stats.rv_continuous
        The waiting time random variable. The shape parameters of the distribution
        are `N`, the number of buses, and `tp`, the persistence time.
    """
    # get volumes from growth rates
    volumes = []
    for r in rates:
        volumes.append(r.antiderivative())

    # # calculate mean volume (growth rate)
    # x = np.linspace(0, tmax, Nknots)
    # y = np.average([v(x) for v in volumes], weights=weights, axis=0)
    # dydx = np.average([r(x) for r in rates], weights=weights, axis=0)
    # spl = CubicHermiteSpline(x, y, dydx)


    class WLC_model(rv_continuous):
        def __init__(self, **kwargs):
            self.tmax = np.amax([r.x[-1] for r in rates])
            super().__init__(**kwargs)

        def _get_support(self, N, tp):
            a = 0
            b = inverse_wlc(self.tmax, tp)
            return a, b

        def _cdf(self, t, N, tp):
            T = np.sqrt(wlc(t, tp))
            F = spl(T) / A
            return 1 - (1-F)**N

        # def _pdf(self, t, N, tp):
        #     T = np.sqrt(wlc(t, tp))
        #     F = spl(T) / A
        #     rate = spl.derivative()(T) / A
        #     return N * (1-F)*(N-1) * rate *  (tp/T) * (1 - np.exp(-(t/tp)))

    RV = WLC_model(momtype=0)
    return RV

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

if __name__ == "__main__":
    # test WLC inversion
    r = 50 * np.random.random(100)
    lp = 50 * np.random.random(50)
    rnew = np.expand_dims(r, axis=-1) * np.ones_like(lp)
    L = inverse_wlc(r, lp)
    plt.scatter(rnew.flatten(), np.sqrt(wlc(L, lp)).flatten())
    plt.show()
