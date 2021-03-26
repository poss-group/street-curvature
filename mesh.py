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
        self.turning_angles = None
        self.G = None
        self.snapped_points = np.zeros_like(self.points)

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

        # get pmesh
        points, triangles, boundary = pmesh(size1, a, size2, b,
                                            beta, offset)

        # move to geolocation
        points = geographical_to_spherical(points*(180/np.pi))
        points = equator_to_geolocation(location, points)

        return cls(points, triangles=triangles, boundary=boundary)

    @classmethod
    def hmesh_at_geolocation(cls, location, size, a, offset=0):
        """
        Construct a hexagonal mesh at a geographical location.

        Parameters
        ----------
        location : (2,) array of float
            Location in geographical coordinates [longitude, latitude].
        size : int
            Size of the mesh (number of rings around centre).
        a : float
            Lattice spacing
        offset : float, optional
            Angle between first basis vector and horizontal axis (equator).

        Notes
        -----
        The mesh is first constructed at the equator and then rotated to the
        geolocation. Therefore, offset=0 does not mean that the first basis
        vector points eastward.
        """
        # transform lengths to angular distances
        R = 6371
        a /= R

        # get hmesh
        points, triangles, boundary = hmesh(size, a, offset)

        # move to geolocation
        points = geographical_to_spherical(points*(180/np.pi))
        points = equator_to_geolocation(location, points)

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
            self.snapped_points[e[0]] = np.array(route.raw['waypoints'][0]['location'])
            self.snapped_points[e[1]] = np.array(route.raw['waypoints'][-1]['location'])
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
            self.snapped_points[e[0]] = np.array(route.raw['waypoints'][0]['location'])
            self.snapped_points[e[1]] = np.array(route.raw['waypoints'][-1]['location'])
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

        # calculate turning angles
        turning_angles = []
        for k in self.boundary:
            mask = np.array(self.tri.triangles == k)
            alpha = angles[mask]
            turning_angles.append(np.pi-np.sum(alpha))
        self.turning_angles = np.array(turning_angles)

    def get_boundary_polygon(self):
        """
        Get the congruent polygon as a new Mesh istance.

        Notes
        -----
        Only works for parallelogram meshes.
        """
        # find middle points
        barycentre = np.mean(self.points, axis=0)
        Aidx = np.argmin(np.linalg.norm(self.points-barycentre, axis=1))

        # find polygon vertices
        idx = [Aidx,]
        for k in self.boundary:
            mask = np.array(self.tri.triangles == k)
            if np.sum(mask) == 2:
                idx.append(k)
        idx = np.array(idx)

        # create new Mesh instance
        points = self.points[idx]
        return Mesh(points)

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

    def save_numpy(self, path='./'):
        """
        Save data as numpy binary files.

        Parameters
        ----------
        path : str, optional
            Directory where data should be saved at.
        """
        np.save(path+"points.npy", self.points)
        np.save(path+"triangles.npy", self.tri.triangles)
        np.save(path+"interior.npy", self.interior)
        np.save(path+"curvatures.npy", self.curvatures)
        np.save(path+"distances.npy", self.distances)

if __name__ == "__main__":
    # # test if Euclidean metric gives zero curvature
    # def euclidean_metric(A, B):
    #     return np.linalg.norm(A-B, axis=-1)
    # M = 40
    # points = np.random.random((M, 2))
    # mesh = Mesh(points)
    # mesh.distances_from_metric(euclidean_metric)
    # mesh.apply_defect_scheme()
    # print(mesh.curvatures)

    # # test pmesh construction
    # mesh = Mesh.pmesh_at_geolocation(np.array([0, 30]), 5, 10)
    # plt.axis("equal")
    # plt.triplot(mesh.tri.x, mesh.tri.y, mesh.tri.triangles)
    # plt.show()

    # # test if GCDs gives earth curvature
    # from metrics import GCD
    # mesh.distances_from_metric(GCD, args=(6371,))
    # mesh.apply_defect_scheme()
    # print(mesh.curvatures)

    # # test OSRM routing
    # client = OSRM(base_url='http://134.76.24.136/osrm')
    # mesh = Mesh.pmesh_at_geolocation(np.array([9.939, 51.5364]), 3, 10)
    # mesh.durations_from_router(client, profile='car')
    # mesh.apply_defect_scheme()
    # plt.axis("equal")
    # plt.scatter(mesh.tri.x[mesh.interior], mesh.tri.y[mesh.interior],
    #             c=mesh.curvatures)
    # plt.colorbar()
    # plt.show()

    # # test hmesh construction
    # mesh = Mesh.hmesh_at_geolocation(np.array([0, 30]), 5, 10)
    # plt.axis("equal")
    # plt.triplot(mesh.tri.x, mesh.tri.y, mesh.tri.triangles)
    # B = mesh.boundary
    # plt.scatter(mesh.tri.x[B], mesh.tri.y[B], c='red')
    # plt.show()

    # # test Karlsruhe metric
    # from metrics import karlsruhe
    # points, triangles, boundary = hmesh(10, 1, 0)
    # mesh = Mesh(points, triangles=triangles, boundary=boundary)
    # mesh.distances_from_metric(karlsruhe)
    # mesh.apply_defect_scheme()
    # plt.axis("equal")
    # plt.scatter(mesh.tri.x[mesh.interior], mesh.tri.y[mesh.interior],
    #             c=mesh.curvatures)
    # plt.colorbar()
    # plt.show()

    # # test snapping recording
    # client = OSRM(base_url='http://134.76.24.136/osrm')
    # mesh = Mesh.hmesh_at_geolocation(np.array([9.939, 51.5364]), 3, 4)
    # mesh.durations_from_router(client, profile='car')
    # plt.figure()
    # plt.axis("equal")
    # plt.scatter(mesh.tri.x[mesh.interior], mesh.tri.y[mesh.interior],
    #             c='k')
    # plt.scatter(mesh.snapped_points[mesh.interior, 0],
    #             mesh.snapped_points[mesh.interior, 1],
    #             c='r')
    # plt.show()

    # test boundary polygon creation
    client = OSRM(base_url='http://134.76.24.136/osrm')
    mesh = Mesh.hmesh_at_geolocation(np.array([9.939, 51.5364]), 3, 4)
    poly = mesh.get_boundary_polygon()
    plt.figure()
    plt.axis("equal")
    plt.scatter(mesh.tri.x, mesh.tri.y, c='k')
    plt.scatter(poly.tri.x, poly.tri.y, c='r')
    plt.show()
