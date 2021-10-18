import numpy as np
from scipy.interpolate import PPoly, interp1d
from scipy.optimize import curve_fit
from scipy.stats import linregress
import networkx as nx
import multiprocessing as mp
from utils_network import equally_spaced_edge_position_sample, uniform_edge_positions_sample

def boxcar(t, left, right):
    """
    Vectorized boxcar (rectangle) function.

    Parameters
    ----------
    t : ndarray
        The evaluation points.
    left : ndarray
        The left bounds of the rectangles. Must have the same shape as t.
    right : ndarray
        The right bounds of the rectangles. Must have the same shape as t.

    Returns
    -------
    x : ndarray
       Boolean, `True` where the evaluation point lies within an interval.
    """
    return (t >= left) * (t < right)

def get_total_volume(G, weight):
    """
    Get sum of edge weights in a MultiDiGraph.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The graph. Must have a `oneway` edge attribute. If `oneway = False`,
        the edge weight is only counted half.
    weight : string
        The edge weight to sum up.

    Returns
    -------
    A : float
        The sum of weights.
    """
    w = np.array(list(nx.get_edge_attributes(G, weight).values()))
    oneway = np.array(list(nx.get_edge_attributes(G, 'oneway').values()))

    return np.sum(oneway * w + 0.5 * ~oneway * w)

def growth_rate_at_node(G, A, weight, speed_key='speed_kph', speed_unit='kph'):
    """
    Get the volume growth rate at node A of network G.

    Parameters
    ----------
    G : nx.networkx.MultiDiGraph
       The network, must allow directed and multi-edges. Needs to have
       a `oneway` edge attribute specifying for which edges the reversed edge
       is also present.
    A : int
        Node ID
    weight : string
        Travel time edge weight.
    speed_key : string, optional
        Name of edge speed attribute.
    speed_unit : string, optional
        Units of the speed attribute.

    Returns
    -------
    rate : scipy.interpolate.PPoly
        The volume growth rate as a piecewise constant function.
        Units are meters per [weight].
    """
    # calculate shortest path lengths
    distfromA = dict(nx.shortest_path_length(G, source=A, weight=weight))

    # unit conversion for speeds
    conv_factor = 1
    if speed_unit == "kph":
        conv_factor /= 3.6

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    speeds = []
    for u, v, k, ddict in G.edges(data=True, keys=True):
        du = distfromA[u]
        w = ddict[weight]
        speeds.append(ddict[speed_key])
        if ddict['oneway']:
            left.append(du)
            right.append(du + w)
        else:
            dv = distfromA[v]
            dmax = 0.5 * (du + dv + w)
            left.append(du)
            right.append(dmax)
    times = np.sort(np.unique(left+right))
    left = np.array(left)
    right = np.array(right)
    speeds = np.array(speeds) * conv_factor
    t = np.expand_dims(times, axis=-1)
    rates = np.sum(boxcar(t, left, right) * speeds, axis=-1)
    coeffs = np.array([np.zeros_like(rates), rates])
    breakpoints = np.concatenate((times, [times[-1]+1]))

    return PPoly(coeffs, breakpoints)

def growth_rate_at_edge_positions(G, edge, positions, weight,
                                  speed_key='speed_kph', speed_unit='kph'):
    """
    Get the volume growth rate at edge positions of network G.

    Parameters
    ----------
    G : nx.networkx.MultiDiGraph
       The network, must allow directed and multi-edges. Needs to have
       a `oneway` edge attribute specifying for which edges the reversed edge
       is also present.
    edge : tuple (u, v, k)
        The edge identifier.
    positions : ndarray, 1-D
        The positions along edge, represented as float in [0,1].
    weight : string
        Travel time edge weight.
    speed_key : string, optional
        Name of edge speed attribute.
    speed_unit : string, optional
        Units of the speed attribute.

    Returns
    -------
    rates : list of scipy.interpolate.PPoly
        The volume growth rates as piecewise constant functions.
        Units are meters per [weight].
    """
    edge_data = G.get_edge_data(*edge)
    A = edge[0]
    B = edge[1]

    # unit conversion for speeds
    conv_factor = 1
    if speed_unit == "kph":
        conv_factor /= 3.6

    # calculate shortest path lengths
    distfromB = dict(nx.shortest_path_length(G, source=B, weight=weight))
    wAB = G.get_edge_data(*edge)[weight]

    # build array of left and right borders of ramp-like functions
    left = []
    right = []
    speeds = []
    if edge_data['oneway']:
        for u, v, k, ddict in G.edges(data=True, keys=True):
            du = distfromB[u] + wAB*(1-positions)
            if edge != (u, v, k):
                w = ddict[weight]
                if ddict['oneway']:
                    left.append(du)
                    right.append(du + w)
                    speeds.append(ddict[speed_key])
                else:
                    dv = distfromB[v] + wAB*(1-positions)
                    dmax = 0.5 * (du + dv + w)
                    left.append(du)
                    right.append(dmax)
                    speeds.append(ddict[speed_key])
            else:
                left.append(np.zeros_like(positions))
                right.append(wAB*(1-positions))
                speeds.append(ddict[speed_key])
                left.append(du)
                right.append(distfromB[A]+np.ones_like(positions)*wAB)
                speeds.append(ddict[speed_key])
    else:
        distfromA = dict(nx.shortest_path_length(G, source=A, weight=weight))
        for u, v, k, ddict in G.edges(data=True, keys=True):

            if edge != (u, v, k) and edge != (v, u, k):
                du = np.minimum(distfromA[u] + wAB*positions,
                            distfromB[u] + wAB*(1-positions))
                w = ddict[weight]
                if ddict['oneway']:
                    left.append(du)
                    right.append(du + w)
                    speeds.append(ddict[speed_key])
                else:
                    dv = np.minimum(distfromA[v] + wAB*positions,
                            distfromB[v] + wAB*(1-positions))
                    dmax = 0.5 * (du + dv + w)
                    left.append(du)
                    right.append(dmax)
                    speeds.append(ddict[speed_key])
        left.append(np.zeros_like(positions))
        left.append(np.zeros_like(positions))
        right.append(wAB*positions)
        right.append(wAB*(1-positions))
        speeds.append(edge_data[speed_key])
        speeds.append(edge_data[speed_key])
    interpolators = []
    left = np.array(left)
    right = np.array(right)
    speeds = np.array(speeds) * conv_factor
    for j in range(positions.size):
        l = left[:,j]
        r = right[:,j]
        times = np.sort(np.unique(np.concatenate((l,r))))
        t = np.expand_dims(times, axis=-1)
        rates = np.sum(boxcar(t, l, r) * speeds, axis=-1)
        coeffs = np.array([np.zeros_like(rates), rates])
        breakpoints = np.concatenate((times, [times[-1]+1]))
        interpolators.append(PPoly(coeffs, breakpoints))

    return interpolators

def volume_growth(G, weight, Npos, pos_weight=None,
                  speed_key='speed_kph', speed_unit='kph',
                  sample_func=equally_spaced_edge_position_sample):
    """
    Do a volume growth analysis of a network.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The network. Must have a `oneway` edge attribute.
    weight : string
        Travel time edge weight.
    Npos : int
        The number of edge positions for which to calculate volume growth.
    pos_weight : str, optional
        The edge weight used for the equally spaced edge position sample.
        If `pos_weight=None`, the `weight` is used.
    speed_key : string, optional
        Name of edge speed attribute.
    speed_unit : string, optional
        Units of the speed attribute.

    Returns
    -------
    rates : list of scipy.interpolate.PPoly
        The volume growth rates as piecewise constant functions.
        Units are meters per [weight].
    """
    # get edge positions
    w = weight if pos_weight is None else pos_weight
    edge_pos = sample_func(G, Npos, w)

    # set up multiprocessing
    cpus = mp.cpu_count() - 1
    pool = mp.Pool(cpus)
    params = [(G, edge, pos, weight, speed_key, speed_unit) for edge, pos in edge_pos.items()]
    sma = pool.starmap_async(growth_rate_at_edge_positions, params)
    results = sma.get()
    pool.close()
    pool.join()

    return sum(results, [])

def get_mean_volume_interpolant(G, weight, Npos, Ninter, pos_weight=None,
                  speed_key='speed_kph', speed_unit='kph'):
    """
    Calculates the mean volume growth.

    Parameters
    ----------
    G : networkx.MultiDiGraph
        The network. Must have a `oneway` edge attribute.
    weight : string
        Travel time edge weight.
    Npos : int
        The number of edge positions for which to calculate volume growth.
    Ninter : int
        The number of interpolation points for the volume growth curve.
    pos_weight : str, optional
        The edge weight used for the equally spaced edge position sample.
        If `pos_weight=None`, the `weight` is used.
    speed_key : string, optional
        Name of edge speed attribute.
    speed_unit : string, optional
        Units of the speed attribute.

    Returns
    -------
    F : scipy.interpolate.interp1d
        The mean volume growth interpolator.
        Units are normalized to the total edge length.
    """
    A = get_total_volume(G, 'length')
    rates = volume_growth(G, weight, Npos, pos_weight=pos_weight,
                          speed_key=speed_key, speed_unit=speed_unit)
    tmax = np.amax([r.x[-1] for r in rates])
    t = np.linspace(0, tmax, Ninter)
    v = np.average([r.antiderivative()(t)/A for r in rates],
                   axis=0)
    return interp1d(t, v, bounds_error=False, fill_value=(0, 1))

def get_interp_derivative(F):
    """
    Get the derivative of a mean volume growth curve.

    Parameters
    ----------
    F : scipy.interpolate.interp1d
        The mean volume growth interpolator.
        Units are normalized to the total edge length.

    Returns
    -------
    Fprime : scipy.interpolate.interp1d
        The mean volume growth rate interpolator.
        Units are [time]^-1.
    """
    rate = np.gradient(F.y, F.x)
    return interp1d(F.x, rate, bounds_error=False, fill_value=(0, 0))

def get_scaling(F):
    """
    Get scaling parameters from a mean volume interpolant.

    Parameters
    ----------
    F : scipy.interpolate.interp1d
        The mean volume growth interpolator.
        Units are normalized to the total edge length.

    Returns
    -------
    nu : float
        The effective dimension.
    c : float
        The power law coefficient.

    Notes
    -----
    The fit minimizes the residuals ``F.y - cu * F.x ** nu``.
    """
    # first obtain exponent
    y = np.log(F.y[1:])
    x = np.log(F.x[1:])
    grad = np.gradient(y, x)
    nu = np.amax(grad)

    # then fit power law (scaling x axis to [0,1])
    stop = np.argmax(grad)+1
    t = F.x[:stop]
    v = F.y[:stop]
    func = lambda x, c: c * x**nu
    popt, _ = curve_fit(func, t/t[-1], v)

    return nu, popt[0]/(t[-1]**nu)

def get_scaling_variant(F):
    """
    Get scaling parameters from a mean volume interpolant.

    Parameters
    ----------
    F : scipy.interpolate.interp1d
        The mean volume growth interpolator.
        Units are normalized to the total edge length.

    Returns
    -------
    nu : float
        The effective dimension.
    c : float
        The power law coefficient.

    Notes
    -----
    The fit minimizes the residuals ``F.y - cu * F.x ** nu``.
    """
    # first obtain exponent
    y = np.log(F.y[1:])
    x = np.log(F.x[1:])
    grad = np.gradient(y, x)
    nu = np.amax(grad)
    region = (nu-grad)/nu < 0.1

    # then fit a line on a loglog plot
    fit = linregress(x[region], y[region])

    return fit.slope, np.exp(fit.intercept)

def power_fit(rates, Nsamples=1000, max_rate=0.5, weights=None):
    """
    Fit a power law to the mean volume growth defined by a set
    of volume growth rates.

    Parameters
    ----------
    rates : list of scipy.interpolate.PPoly
        The volume growth rates as piecewise constant functions.
    Nsamples : int, optional
        The number of samples used for evaluating the volume growth curves.
    max_rate : float, optional
        The maximum mean growth rate defining the right boundary of the fitting
        region. Given as a fraction of the maximum mean growth rate.
    weights : optional, list
        Weights used for averaging. If `weights=None`,
        then uniform weights are used.

    Returns
    -------
    popt : array
        Optimal parameters c, nu so that the sum of squared residuals of
        ``c * x**nu - mean_rate(x)`` is minimized.
    pcov : 2-D array
        The estimated covariance of popt.
    """
    # calculate mean growth rate
    t = np.linspace(0, np.amax([r.x[-1] for r in rates]), Nsamples)
    mean_rate = np.average([r.antiderivative()(t) for r in rates],
                           weights=weights, axis=0)

    # define fitting region
    end = np.argmax(mean_rate > max_rate*np.amax(mean_rate))
    x = t[:end]
    y = mean_rate[:end]

    func = lambda x, c, nu : c * x**nu
    return curve_fit(func, x, y)

def quad_fit(rates, Nsamples=1000, max_rate=0.5, weights=None):
    """
    Fit a quadratic growth to the mean volume growth defined by a set
    of volume growth rates.

    Parameters
    ----------
    rates : list of scipy.interpolate.PPoly
        The volume growth rates as piecewise constant functions.
    Nsamples : int, optional
        The number of samples used for evaluating the volume growth curves.
    max_rate : float, optional
        The maximum mean growth rate defining the right boundary of the fitting
        region. Given as a fraction of the maximum mean growth rate.
    weights : optional, list
        Weights used for averaging. If `weights=None`,
        then uniform weights are used.

    Returns
    -------
    popt : array
        Optimal parameter c so that the sum of squared residuals of
        ``c * x**2 - mean_rate(x)`` is minimized.
    pcov : 2-D array
        The estimated covariance of popt.
    """
    # calculate mean growth rate
    t = np.linspace(0, np.amax([r.x[-1] for r in rates]), Nsamples)
    mean_rate = np.average([r.antiderivative()(t) for r in rates],
                           weights=weights, axis=0)

    # define fitting region
    end = np.argmax(mean_rate > max_rate*np.amax(mean_rate))
    x = t[:end]
    y = mean_rate[:end]

    func = lambda x, c : c * x**2
    return curve_fit(func, x, y)
