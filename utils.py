import numpy as np

def heron(a, b, c):
    """
    Calculate the area of a triangle by Heron's formula.

    Parameters
    ----------
    a, b, c : array_like or floats
        The sides of the triangle

    Returns
    -------
    ndarray or float
        The area

    Notes
    -----
    This function does not check if the side lengths define
    a valid planar triangle.
    """
    s = (a+b+c) / 2

    return np.sqrt(s*(s-a)*(s-b)*(s-c))

def interior_angle(c1, c2, d):
    """
    Calculate the interior angle opposite of d, using the cosine law.

    Parameters
    ----------
    c1, c2, d : array_like or floats
        The sides of the triangle

    Returns
    -------
    ndarray or float
        The angle

    Notes
    -----
    This function does not check if the side lengths define
    a valid planar triangle.
    """
    arg = (c1**2 + c2**2 - d**2) / (2*c1*c2)

    return np.arccos(arg)

def mercator(coordinates):
    """
    Calculate mercator projection

    Parameters
    ----------
    coordinates : array, shape (M, 2)
        The geographical coordinates of M points, in the order
        [longitude, latitude], in degrees

    Returns
    -------
    ndarray, shape (M, 2)
        The Mercator coordinates of M points, in the order
        [x, y]
    """
    longitudes = coordinates.T[0]
    latitudes = coordinates.T[1]
    x = longitudes * (np.pi/180)
    y = np.log(np.tan(np.pi/4 + latitudes*(np.pi/360)))

    return np.array([x,y]).T

def geographical_to_spherical(coordinates):
    """
    Transform from geographical to spherical coordinates.

    Parameters
    ----------
    coordinates : array, shape (M, 2)
        The geographical coordinates of M points, in the order
        [longitude, latitude], in degrees

    Returns
    -------
    ndarray, shape (M, 2)
        The spherical coordinates of M points, in the order
        [theta, phi], in radians
    """
    longitudes = coordinates.T[0]
    latitudes = coordinates.T[1]
    theta = (90-latitudes) * (np.pi/180)
    phi = longitudes * (np.pi/180)

    return np.array([theta, phi]).T

def spherical_to_geographical(coordinates):
    """
    Transform from spherical to geographical coordinates.

    Parameters
    ----------
    coordinates : array, shape (M, 2)
        The spherical coordinates of M points, in the order
        [theta, phi], in radians

    Returns
    -------
    ndarray, shape (M, 2)
        the geographical coordinates of M points, in the order
        [longitude, latitude], in degrees
    """
    theta = coordinates.T[0]
    phi = coordinates.T[1]
    longitudes = phi * (180/np.pi)
    latitudes = (np.pi/2-theta) * (180/np.pi)

    return np.array([longitudes, latitudes]).T

def rotate_to_geolocation(location, coordinates):
    """
    Rotate the coordinates to the given geolocation.

    Parameters
    ----------
    location : array, shape (2)
        The geographical coordinates of the location, in order
        [longitude, latitude], in degrees
    coordinates : array, shape (M, 2)
        The spherical coordinates of M points, in the order
        [theta, phi], in radians

    Returns
    -------
    ndarray, shape (M, 2)
        The geographical coordinates of the M rotated points, in the order
        [longitude, latitude], in degrees
    """
    location_spherical = geographical_to_spherical(location)
    theta = coordinates.T[0]
    phi = coordinates.T[1]
    # transform into qubit
    psi_0 = np.cos(theta/2)
    psi_1 = np.exp(1j*phi) * np.sin(theta/2)
    # rotate qubit
    psi_rotated_0 = (np.cos(location_spherical[0]/2) * psi_0
                     - (np.exp(-1j*location_spherical[1])
                        * np.sin(location_spherical[0]/2)
                        * psi_1))
    psi_rotated_1 = ((np.exp(1j*location_spherical[1])
                      * np.sin(location_spherical[0]/2)
                      * psi_0)
                     + np.cos(location_spherical[0]/2) * psi_1)
    # transform back to spherical coordinates
    theta = 2 * np.arctan(np.abs(psi_rotated_1)/np.abs(psi_rotated_0))
    phi = np.angle(psi_rotated_1) - np.angle(psi_rotated_0)

    return spherical_to_geographical(np.array([theta, phi]).T)

def construct_polygon(N, R, location, offset=0):
    """
    Construct a regular polygon around a location.

    Parameters
    ----------
    N : int
        The number of edges of the polygon.
    R : float
        The circumradius of the polygon, in kilometers
    location : array, shape (2)
        The geographical coordinates of the location, in order
        [longitude, latitude], in degrees
    offset : float, optional
        The longitude of the first (unrotated) vertex, in radians

    Returns
    -------
    ndarray, shape (N, 2)
        The geographical coordinates of the polygon vertices, in order
        [longitude, latitude], in degrees
    """
    # calculate azimuthal angle corresponding to R
    EARTH_RADIUS = 6371
    theta = R/EARTH_RADIUS
    # array of evenly spaced longitudes
    M = np.floor(((offset+np.pi)*N) / (2*np.pi))
    phi = 2*np.pi*(np.arange(N)/N) + offset - 2*np.pi*(M/N)
    coordinates = np.array([theta*np.ones(N), phi]).T

    return rotate_to_geolocation(location, coordinates)

def get_triplength(A, B, router, dimension='duration'):
    """
    Return the triplength between A and B obtained with OSRM.

    Parameters
    ----------
    A : array, shape (2)
        The geographical coordinates of point A, in order
        [longitude, latitude], in degrees
    B : array, shape (2)
        The geographical coordinates of point B, in order
        [longitude, latitude], in degrees
    router : osrm.Client instance
        The routing client
    dimension : {'duration', 'distance'}, optional
        The dimension of the returned triplength. If 'duration',
        the unit is minutes, if 'distance', the unit is kilometers.

    Returns
    -------
    float
        The triplength
    """
    response = router.route(coordinates=np.array([A,B]))
    if dimension == "duration": # trip length in minutes
        return response['routes'][0]['legs'][0]['duration']/60
    if dimension == 'distance': # trip length in km
        return response['routes'][0]['legs'][0]['distance']/1000

def get_tripmeasures(A, B, router):
    """
    Return the duration and distance of the trip from A and B
    obtained with OSRM.

    Parameters
    ----------
    A : array, shape (2)
        The geographical coordinates of point A, in order
        [longitude, latitude], in degrees
    B : array, shape (2)
        The geographical coordinates of point B, in order
        [longitude, latitude], in degrees
    router : osrm.Client instance
        The routing client

    Returns
    -------
    float
        The duration in minutes
    float
        The distance in kilometers
    """
    response = router.route(coordinates=np.array([A,B]))['routes'][0]['legs'][0]
    return np.array([response['duration']/60, response['distance']/1000])

def measure_polygon(A, B, router, dimension='duration', meanR=False):
    """
    Measure angles and areas of the street network polygon.

    Parameters
    ----------
    A : array, shape (2)
        The geographical coordinates of the center point, in order
        [longitude, latitude], in degrees
    B : array, shape (2)
        The geographical coordinates of the vertices, in order
        [longitude, latitude], in degrees
    router : osrm.Client instance
        The routing client
    meanR : bool, optional
        If True, return the mean circumradius.

    Returns
    -------
    ndarray, shape (N, 2)
        The angles at A. First column is calculated using duration in minutes,
        second column using distance in kilometers.
    ndarray, shape (N, 2)
        The areas of the subtriangles. First column is calculated using
        duration in minutes, second column using distance in kilometers.
    ndarray, shape (2,)
        Present only when meanR = True. The average circumradius of the polygon.
    """
    N = B.shape[0]
    d = np.zeros((N, 2))
    c = np.zeros((N, 2))
    for i in range(N):
        d[i] = get_tripmeasures(B[i], B[(i+1)%N], router)
        c[i] = get_tripmeasures(A, B[i], router)
    mask1 = d > (c + np.roll(c, 1, axis=0))
    mask2 = np.abs(c - np.roll(c, 1 ,axis=0)) > d
    angles = interior_angle(c, np.roll(c, 1, axis=0), d)
    areas = heron(c, np.roll(c, 1, axis=0), d)
    angles[mask1] = np.pi
    angles[mask2] = 0
    areas[mask1] = 0
    areas[mask2] = 0

    if meanR:
        return angles, areas, np.average(c, axis=0)
    else:
        return angles, areas

def asymmetry_parameter(angles, axis=0):
    """
    Calculate the asymmetry parameter of a polygon.

    Parameters
    ----------
    angles : array
        The angles at the center, in radians

    Returns
    -------
    float
        Asymmetry parameter
    """
    D = np.ones_like(angles)
    dot = np.sum(D*angles, axis=axis) / (np.linalg.norm(D, axis=axis)
                                         *np.linalg.norm(angles, axis=axis))

    return np.sqrt(1 - dot**2)

def orthodromic_distance(long1, lat1, long2, lat2, R):
    """
    Calculate the element-wise shortest distance on a sphere.

    Parameters
    ----------
    long1 : array
        The longitudes of the first set of points, in degrees
    lat1 : array
        The latitudes of the first set of points, in degrees
    long2 : array
        The longitudes of the second set of points, in degrees
    lat2 : array
        The latitudes of the second set of points, in degrees
    R : float
        The radius of the sphere, in kilometers

    Returns
    -------
    ndarray
        The orthodromic distance, in kilometers
    """
    # convert to radians
    long1_rad = long1 * (np.pi/180)
    lat1_rad = lat1 * (np.pi/180)
    long2_rad = long2 * (np.pi/180)
    lat2_rad = lat2 * (np.pi/180)
    delta_long = long2_rad - long1_rad
    s1 = np.sin(lat1_rad)
    s2 = np.sin(lat2_rad)
    c1 = np.cos(lat1_rad)
    c2 = np.cos(lat2_rad)
    central_angle = np.arctan(np.sqrt((c2*np.sin(delta_long))**2
                                      + (c1*s2-s1*c2*np.cos(delta_long))**2)
                              / (s1*s2+c1*c2*np.cos(delta_long)))

    return R * central_angle
