import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import Delaunay
from routingpy.routers import OSRM

from utils import *

class Mesh(object):
    """

    Parameters
    ----------
    points : (M, 2) array of float
        The coordinates of mesh points. Either geographical coordinates
        in order [longitude, latitude], in degrees, or Cartesian coordinates
    triangles : (ntri, 3) array of int, optional
        For each triangle, the indices of the three points that make
        up the triangle, ordered in an anticlockwise manner.  If not
        specified, the Delaunay triangulation is calculated.
    boundary : array of int, optional
        The indices of the boundary points. If not specified, the
        convex hull is calculated.
    """
    def __init__(self, points, triangles=None, boundary=None):
        self.points = points
        self.tri = Triangulation(points[:,0], points[:,1], triangles=triangles)

        if boundary is None:
            t = Delaunay(points)
            self.boundary = np.unique(t.convex_hull.flatten())
        else:
            self.boundary = boundary

        self.distances = None
        self.defects = None
        self.G = None

    @classmethod
    def pmesh_at_geolocation(cls, location, size1, a, size2=None, b=None, beta=np.pi/3, offset=0):
        """
        Construct a parallelogram mesh at a geographical location.

        Parameters
        ----------
        location : (2,) array of float
            Location in geographical coordinates [longitude, latitude].
        size1 : int
            Size in direction of first basis vector.
        a : float
            Length of first basis vector, in km.
        size2 : int, optional
            Size in direction of first basis vector. If not specified,
        it is taken equal to size1.
        b : float, optional
            Length of first basis vector, in km.  If not specified,
        it is taken equal to a.
        theta : float, optional
            Angle between basis vectors, in radians.
        offset : float, optional
            Angle between first basis vector and horizontal axis (equator).

        Notes
        -----
        The mesh is first constructed at the equator and then rotated to the
        geolocation. Therefore, offset=0 does not mean that the first basis
        vector points eastward.
        """
        # handle kwargs
        if size2 is None:
            size2 = size1
        if b is None:
            b = a

        # transform lengths to angular distances
        R = 6371
        a /= R
        b /= R

        # construct mesh points and triangles
        triangles = []
        m = np.arange(-size1, size1+1)
        n = np.arange(-size2, size2+1)
        M, N = np.meshgrid(m, n)
        M = M.flatten()
        N = N.flatten()
        for k, m in enumerate(M):
            n = N[k]
            if n<size2:
                if m<size1:
                    triangles.append([k, k+1, k+2*size1+1])
                if m>-size1:
                    triangles.append([k, k+2*size1, k+2*size1+1])
        phi = a*np.cos(offset)*M + b*np.cos(beta+offset)*N
        theta = a*np.sin(offset)*M + b*np.sin(beta+offset)*N
        points = np.array([phi, theta]).T

        # rotate mesh points
        points = geographical_to_spherical(points*(180/np.pi))
        points = equator_to_geolocation(location, points)

        # define boundary
        B = (np.abs(M) == size1) + (np.abs(N) == size2)
        boundary = np.where(B)[0]

        return cls(points, triangles=triangles, boundary=boundary)


    def distances_from_metric(self, metric, args=()):
        """
        Calculate the distances between mesh points with the provided metric.

        Parameters
        ----------
        metric : callable(A, B, ...)
            Callable that gives the metric distances between x and y, where
            x and y are arrays of the same shape.
        args : tuple, optional
            Extra arguments to pass to metric function.
        """
        E = self.tri.edges
        d = metric(self.points[E[:, 0], :], self.points[E[:, 1], :], *args)

        self.fill_distances(d)

    def durations_from_router(self, router, **kwargs):
        """
        Calculate the distances between mesh points using a the durations
        returned by arouting machine.

        Parameters
        ----------
        router : Router object from routingpy

        **kwargs : tuple, optional
            Additional kwargs are passed to router's directions method.
        """
        d = []
        counter = 0
        for e in self.tri.edges:
            route = router.directions(locations=self.points[e,:], **kwargs)
            d.append(route.duration/60)
            counter += 1
            print("{} of {} edges calculated.".format(counter, self.tri.edges.shape[0]))

        self.fill_distances(np.array(d))

    def distances_from_router(self, router, **kwargs):
        """
        Calculate the distances between mesh points using a the durations
        returned by arouting machine.

        Parameters
        ----------
        router : Router object from routingpy

        **kwargs : tuple, optional
            Additional kwargs are passed to router's directions method.
        """
        d = []
        counter = 0
        for e in self.tri.edges:
            route = router.directions(locations=self.points[e,:], **kwargs)
            d.append(route.distance/1000)
            counter += 1
            print("{} of {} edges calculated.".format(counter, self.tri.edges.shape[0]))

        self.fill_distances(np.array(d))

    def fill_distances(self, dlist):
        """
        Fill the (ntri, 3) distances array using distances for each edge.

        Parameters
        ----------
        dlist : (nedges,) array of float
            Distances associated with the edges
        """
        edge_indices = np.zeros_like(self.tri.triangles)
        E = np.sort(self.tri.edges, axis=1)
        for i, t in enumerate(self.tri.triangles):
            for j in range(3):
                e = np.sort([t[j], t[(j+1)%3]])
                edge_indices[i][j] = np.where(np.sum(E == e, axis=1) == 2)[0][0]
        self.distances = dlist[edge_indices]

    def apply_defect_scheme(self, scheme='simple'):
        """
        Apply angular defect scheme.

        Parameters
        ----------
        scheme : string, optional
            Angular defect scheme to use, specifying the geometric factor
           that the defect is divided by:

                * 'simple' (default) : 1/3 of the area of the star
                  around a vertex.
                * 'module' : The module of the mesh.
        """
        if self.distances is None:
            raise ValueError("Distances not calculated yet")

        # calculate angles and triangle areas
        angles = interior_angle(self.distances,
                            np.roll(self.distances, 1, axis=1),
                            np.roll(self.distances, 2, axis=1))
        areas = heron(self.distances[:, 0], self.distances[:, 1],
                      self.distances[:, 2])

        # check for degenerate cases
        distances_sorted = np.sort(self.distances, axis=-1)
        distances_check = (distances_sorted[:,2] - distances_sorted[:,1]
                       - distances_sorted[:,0])
        mask0 = np.where(np.logical_or(distances_check > 0,
                                   np.abs(distances_check) < 0))
        if np.sum(mask0[0] > 0):
            for s in mask0:
                half_angle = (np.argmin(self.distances[s,:]) + 1) % 3
                angles[s,:] = 0
                angles[s,half_angle] = np.pi
                areas[s] = 0

        # calculate defects and geometric factors
        defects = []
        G = []
        for k in self.interior:
            mask = np.array(self.tri.triangles == k)
            alpha = angles[mask]
            defects.append(2*np.pi - np.sum(alpha))
            F = areas[np.sum(mask,axis=1) == 1]
            if scheme == 'simple':
                G.append(np.sum(F)/3)
            if scheme == 'module':
                d = self.distances[np.roll(mask, 1, axis=1)]
                G.append(0.5 * np.sum(F)
                         - 0.125 * np.sum(d**2/np.tan(alpha)))

        self.defects = np.array(defects)
        self.G = np.array(G)

    @property
    def interior(self):
        """
        Interior vertices (array of int)
        """
        return np.setdiff1d(np.arange(self.tri.x.size), self.boundary)

    @property
    def curvatures(self):
        """
        Approximated curvatures (array of float)
        """
        if self.defects is None or self.G is None:
            raise ValueError("Defects and areas not calculated yet")

        return self.defects / self.G

if __name__ == "__main__":
    # test if Euclidean metric gives zero curvature
    def euclidean_metric(A, B):
        return np.linalg.norm(A-B, axis=-1)
    M = 40
    points = np.random.random((M, 2))
    mesh = Mesh(points)
    mesh.distances_from_metric(euclidean_metric)
    mesh.apply_defect_scheme()
    print(mesh.curvatures)

    # test pmesh construction
    mesh = Mesh.pmesh_at_geolocation(np.array([0, 30]), 5, 10)
    plt.axis("equal")
    plt.triplot(mesh.tri.x, mesh.tri.y, mesh.tri.triangles)
    plt.show()

    # test if GCDs gives earth curvature
    from metrics import GCD
    mesh.distances_from_metric(GCD, args=(6371,))
    mesh.apply_defect_scheme()
    print(mesh.curvatures)

    # test OSRM routing
    client = OSRM(base_url='http://134.76.24.136/osrm')
    mesh = Mesh.pmesh_at_geolocation(np.array([9.939, 51.5364]), 10, 10)
    mesh.durations_from_router(client, profile='car')
    mesh.apply_defect_scheme()
    plt.axis("equal")
    plt.scatter(mesh.tri.x[mesh.interior], mesh.tri.y[mesh.interior], c=mesh.curvatures)
    plt.colorbar()
    plt.show()