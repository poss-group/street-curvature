import numpy as np
from utils_network import *
from scipy.stats import rv_continuous
from scipy.interpolate import CubicHermiteSpline
from volume_growth import *
from scipy.optimize import newton, least_squares, brentq

def poly_convolve(A, B):
    """
    Calculate product of two piecewise polynomials by convolution.

    Parameters
    ----------
    A, B : 2-D arrays
        Coefficient arrays. The sizes of the first axes represent the respective
        degrees. The size of the second axes must agree, and is the number of segments
        of the piecewise functions.

    Returns
    -------
    C : 2-D array
        Coefficient array of product of A and B.
    """
    kA = A.shape[0]
    kB = B.shape[0]
    m = A.shape[1]
    A_ = np.zeros((m, kA+kB-1), dtype=A.dtype)
    A_[:, :kA] = A.T
    B_ = np.zeros((m, kA+kB-1), dtype=B.dtype)
    B_[:, :kB] = B.T
    C_fft = np.fft.fft(A_, axis=1) * np.fft.fft(B_, axis=1)

    return np.real(np.fft.ifft(C_fft)).T

def mean_pconst(rate):
    """
    Mean of a piecewise constant pdf.

    Parameters
    ----------
    rate : scipy.interpolate.PPoly object
        The pdf as a piecewise polynomial. Must have order 1.

    Returns
    -------
    E : float
        The mean of the distribution.
    """
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

def wlc_derivative_wrt_lp(L, lp):
    return (np.exp(-L/lp)*(L + 2*lp) + L - 2*lp) / np.sqrt(wlc(L, lp))

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

def lsq_quantile_fit(wt_data, RV, N, Ninter, q='all', truncate=None):
    # sample model CDF for interpolation
    if truncate is None:
        t = np.linspace(0, RV.tmax, Ninter)
        cdf = RV.cdf(t, N)
    else:
        t = np.linspace(0, truncate, Ninter)
        cdf = RV.cdf(t, N)
        cdf /= cdf[-1]

    # data quantiles: sorted array
    if q == 'all':
        data_quantiles = np.sort(wt_data)
        n = wt_data.size
        model_quantiles = np.interp(np.arange(1, n+1)/n, cdf, t)

    else:
        data_quantiles = np.quantile(wt_data, q)
        model_quantiles = np.interp(q, cdf, t)

    # # only retain positive quantiles
    # mask = data_quantiles > 0
    # positive_quantiles = data_quantiles[mask]
    # npositive = np.argmax(mask)

    # set up residual function and its Jacobian
    def f(tp):
        return np.sqrt(wlc(data_quantiles, tp)) - model_quantiles
    def jac(tp):
        return wlc_derivative_wrt_lp(data_quantiles, tp)[:,np.newaxis]

    # intial guess: tp that matches medians
    # model_median = np.interp(0.5, cdf, t)
    # med = np.median(wt_data)
    # x0 = newton(lambda tp : np.sqrt(wlc(med, tp))-model_median,
    #             np.mean(wt_data),
    #             fprime = lambda tp : wlc_derivative_wrt_lp(med, tp))
    x0 = RV.t0 / 2

    return least_squares(f, x0, jac=jac, method='lm'), data_quantiles, model_quantiles

def get_upper_limit(tmax, tp):
    f = lambda t : wlc(t, tp) - tmax**2
    fprime = lambda t : 2*tp*(1-np.exp(-t/tp))
    return newton(f, tmax**2, fprime=fprime)

def draw_distances(rate, size):
    """
    Draw distances according to a piecewise constant pdf.

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
    """
    Calculate mean waiting time from a piecewise polynomial pdf formed
    from a piecewise constant volume growth rate.

    Parameters
    ----------
    r : scipy.interpolate.PPoly
        Volume growth rate as a piecewise constant function.
    A : float
        Total volume used for normalization.
    Nmax : int
        Maximum number of buses for which to calculate the mean waiting time consecutively.

    Returns
    -------
    mean_wt : list with length Nmax
        Mean waiting time for 1, 2, ... , Nmax buses.

    Notes
    -----
    This method becomes numerically unstable for larger Nmax (around 10).
    Use sampling methods in RV_minimal instead.
    """
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
    """
    Average mean waiting times for a list of volume growth rates.

    Parameters
    ----------
    rates : list of scipy.interpolate.PPoly
        Volume growth rates as a piecewise constant functions.
    A : float
        Total volume used for normalization.
    Nmax : int
        Maximum number of buses for which to calculate the mean waiting time consecutively.
    weights : optional, list
        Weights used for averaging. If `weights=None`, then uniform weights are used.

    Returns
    -------
    mean_wt : list with length Nmax
        Mean waiting time for 1, 2, ... , Nmax buses.

    Notes
    -----
    This method becomes numerically unstable for larger Nmax (around 10).
    Use sampling methods in RV_minimal instead.
    """
    return np.average([get_single_mean_waiting_times_minimal(r, A, Nmax) for r in rates], weights=weights, axis=0)


def RV_minimal(rates, weights=None):
    """
    Build a minimal model waiting time random variable from a set of volume growth rates.

    Parameters
    ----------
    rates : list of scipy.interpolate.PPoly instances
        The volume growth rates as piecewise polynomials.
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

    # save total volumes
    total_volumes = [v.c[-1,-1] for v in volumes]

    # find maximum waiting time
    tmax = np.amax([r.x[-1] for r in rates])

    class minimal_model(rv_continuous):
        def __init__(self, **kwargs):
            self.tmax = np.amax([r.x[-1] for r in rates])
            self.t0 = np.average([mean_pconst(r)/A
                                  for r, A in zip(rates, total_volumes)], weights=weights)
            super().__init__(**kwargs)

        def custom_rvs(self, N, size=1):
            # get total number of samples
            Nsamples = np.prod(size)

            # choose pick-up locations randomly
            locations = np.random.choice(len(rates), size=Nsamples, p=weights)
            u, counts = np.unique(locations, return_counts=True)

            # draw distances
            distances = np.concatenate([draw_distances(rates[i], (s, N))
                                        for i, s in zip(u, counts)], axis=0)

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

        def fancy_ppf(self, q, N):
            f = lambda t : self._cdf(t, N) - q
            fder = lambda t : self._pdf(t, N)
            return newton(f, np.linspace(0, self.tmax, q.size), fprime=fder)

        def _cdf(self, t, N):
            return np.average([1 - (1-v(t)/A)**N for v, A in zip(volumes, total_volumes)],
                              axis=0, weights=weights)

        def _pdf(self, t, N):
            return np.average([N * (1-v(t)/A)**(N-1) * r(t) / A for r, v, A in zip(rates, volumes, total_volumes)],
                              axis=0, weights=weights)

    RV = minimal_model(momtype=0, a=0, b=tmax)
    return RV

def RV_WLC(rates, weights=None):
    """
    Build a WLC model waiting time random variable from a set of volume growth rates.

    Parameters
    ----------
    rates : list of scipy.interpolate.PPoly instances
        The volume growth rates as piecewise polynomials.
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

    # save total volumes
    total_volumes = [v.c[-1,-1] for v in volumes]

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
            return np.average([1 - (1-v(T)/A)**N for v, A in zip(volumes, total_volumes)],
                              axis=0, weights=weights)

        def _pdf(self, t, N, tp):
            T = np.sqrt(wlc(t, tp))
            return np.average([N * (1-v(T)/A)**(N-1) * (r(T)/A) *  (tp/T) * (1 - np.exp(-(t/tp)))
                               for r, v, A in zip(rates, volumes, total_volumes)]
                              , axis=0, weights=weights)

    RV = WLC_model(momtype=0)
    return RV

def MLE_2par_weibull(x):
    # precalculate stuff
    logx = np.log(x)
    logx2 = logx * logx
    def g(nu):
        p = x**nu
        return np.mean(logx) + 1/nu - np.sum(p*logx) / np.sum(p)

    def gprime(nu):
        p = x**nu
        return (-1/nu**2 - np.sum(p*logx2) / np.sum(p)
                + (np.sum(p*logx)/np.sum(p))**2)

    # # find bracketing interval
    # a = 1
    # step = 0.5
    # b = a + step
    # ga = g(a)
    # while g(b)*ga > 0:
    #     b += step
    # nu = brentq(g, a, b)
    nu = newton(g, 2, fprime=gprime, maxiter=100)

    # plug into the other equation
    b = 1 / np.mean(x**nu)

    return nu, b




# no need to implement this, it is a Weibull distribution
# class MFT_minimal(rv_continuous):
#     def __init__(self, **kwargs):
#         super().__init__(a=0, **kwargs)

#     def _cdf(self, t, b, nu):
#         return 1 - np.exp(-b * t**nu)

#     def _pdf(self, t, b, nu):
#         return b * nu * t**(nu-1) * np.exp(-b * t**nu)

#     def _sf(self, t, b, nu):
#         return np.exp(-b * t**nu)

#     def _ppf(self, q, b, nu):
#         return (-np.log(1-q) / b)**(1/nu)

#     def _logsf(self, t, b, nu):
#         return -b * t**nu

#     def _isf(self, q, b, nu):
#         return (-np.log(q) / b)**(1/nu)

#     def _logpdf(self, t, b, nu):
#         return -b * t**nu + np.log(b*nu) + (nu-1) * np.log(t)


if __name__ == "__main__":
    from volume_growth import *
    from osm import *
    import osmnx as ox
    G = ox.graph_from_place("Broitzem", network_type="drive")
    prepare_graph(G)
    A = np.sum([w for u, v, w in G.to_undirected().edges(data='travel_time')])
    Nsamples = 200
    edge_pos = equally_spaced_edge_position_sample(G, Nsamples, 'length')
    rates = []
    for edge, p in edge_pos.items():
        rates += growth_rate_at_edge_positions(G, edge, p, 'travel_time')
    RV = RV_minimal(rates)

    # # test WLC inversion
    # r = 50 * np.random.random(100)
    # lp = 50 * np.random.random(50)
    # rnew = np.expand_dims(r, axis=-1) * np.ones_like(lp)
    # L = inverse_wlc(r, lp)
    # plt.scatter(rnew.flatten(), np.sqrt(wlc(L, lp)).flatten())
    # plt.show()

    # # test quantile function
    # N = 4
    # q = np.linspace(0.1, 0.9, 9)
    # plt.plot(q, RV.fancy_ppf(q, N))

    # # test quantiles fitting

    # # test data
    # np.random.seed(seed=44)
    # N = 5
    # Ninter = 1000
    # size = 5000
    # Npar = 100
    # noise_level = 0.02 * RV.t0
    # tpvec = RV.t0 * np.logspace(-0.5, 0.5, Npar)
    # x = [np.amin(tpvec), np.amax(tpvec)]
    # tp_fitted = []
    # for tp in tpvec:
    #     wt = RV.custom_rvs(N, size=size)
    #     # wt += noise_level * np.random.randn(size)
    #     wt = inverse_wlc(wt, tp)
    #     res = lsq_quantile_fit(wt, RV, N, Ninter)
    #     tp_fitted.append(res.x[0])
    # plt.figure()
    # plt.title("LS Quantile Function")
    # plt.scatter(tpvec, tp_fitted)
    # plt.plot(x, x, 'k--')
    # plt.show()

    # test creation of MFT RV
    RV = MFT_minimal()
    print(RV.a)
    t = np.linspace(0, 5, 300)
    plt.figure()
    plt.plot(t, RV.pdf(t, 1, 2))
    plt.show()
