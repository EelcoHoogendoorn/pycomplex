"""Implements a geodesic solver on triangular meshes

References
----------
[1] http://www.multires.caltech.edu/pubs/GeodesicsInHeat.pdf

"""
import numpy as np
import numpy_indexed as npi
import scipy.sparse

from pycomplex.math import linalg
from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3


class MyComplex(ComplexTriangularEuclidian3):
    """Subclass that implements the divergence and gradient operators specific to the paper 'Geodesics In Heat'"""

    def triangle_edge_vectors(self, oriented=True):
        """Compute geometrical edge vectors of each edge of all triangles

        Parameters
        ----------
        oriented : bool
            If true, edge vectors on the same edge have opposing signs on both triangles

        Returns
        -------
        ndarray, [n_triangles, 3, 3], float

        """
        grad = self.topology.matrices[0].T
        vecs = grad * self.vertices
        B21 = self.topology._boundary[1]
        O21 = self.topology._orientation[1]
        vecs = vecs[B21]
        if oriented:
            vecs *= O21[..., None]
        return vecs

    def compute_gradient(self, field):
        """Compute gradient of scalar function on vertices, evaluated on faces

        Parameters
        ----------
        field : ndarray, [n_vertices], float
            primal 0-form

        Returns
        -------
        gradient : ndarray, [n_triangles, 3], float
            a vector in 3-space on each triangle
        """
        E20 = self.topology.incidence[2, 0]

        vecs = self.triangle_edge_vectors()
        normals, triangle_area = linalg.normalized(self.triangle_normals(), return_norm=True)
        gradient = (field[E20][:, :, None] * np.cross(normals[:, None, :], vecs)).sum(axis=1)
        return gradient / (2 * triangle_area[:, None])

    def compute_divergence(self, field):
        """Compute divergence of vector field at faces at vertices

        Parameters
        ----------
        field : ndarray, [n_triangles, 3], float
            a flux vector in 3-space on each triangle

        Returns
        -------
        ndarray, [n_vertices], float
            Corresponding notion of divergence on each vertex
        """
        T01, T12 = self.topology.matrices
        div = T01

        vecs = self.triangle_edge_vectors()
        primal_tangent_flux = linalg.dot(vecs, field[:, None, :])   # [n_triangles, 3]

        cotan = 1 / np.tan(self.compute_triangle_angles)
        dual_normal_flux = self.remap_boundary_N(primal_tangent_flux * cotan) / 2
        return div * dual_normal_flux

    def geodesic(self, seed, scale=None):
        """Compute geodesic distance map

        Parameters
        ----------
        seed : ndarray, [n_vertices], float
            primal 0-form; indicating starting point

        Returns
        -------
        ndarray, [n_vertices], float
            primal 0-form; distance from the seed point

        Notes
        -----
        http://www.multires.caltech.edu/pubs/GeodesicsInHeat.pdf

        """
        # diffuse seed to get gradient map of. paper uses implicit method, but explicit is fine here for demonstration
        from examples.diffusion.explicit import Diffusor
        D = Diffusor(self)
        if scale is None:
            scale = np.linalg.norm(np.diff(self.box, axis=0)) / 10  # pretty sensible default guess
        diffused = D.integrate_explicit_sigma(seed * 1.0, scale)

        # now try and find a potential that has the same normalized gradient as the diffused seed
        gradient = -linalg.normalized(self.compute_gradient(diffused))
        rhs = self.compute_divergence(gradient)
        phi = scipy.sparse.linalg.minres(D.laplacian, rhs)[0]
        return phi - phi.min()


if __name__ == '__main__':
    from examples.subdivision.letter_a import create_letter
    import matplotlib.pyplot as plt

    # create an interesting shape to compute geodesics over
    letter = create_letter(3).as_23().subdivide_simplicial().as_3()
    letter = MyComplex(vertices=letter.vertices, topology=letter.topology)#.optimize_weights()

    seed = letter.topology.chain(0, dtype=np.float)
    # pick a point on the surface and give it a seed
    idx = np.argmin(np.linalg.norm(letter.vertices - [2, 2, -3], axis=1))
    seed[idx] = 1
    geo = letter.geodesic(seed)

    letter = letter.copy(vertices = np.dot(letter.vertices, linalg.power(linalg.orthonormalize(np.random.randn(3, 3)), 0.2)))
    letter.plot_3d(plot_dual=False, backface_culling=True, plot_vertices=False)
    letter.plot_primal_0_form(geo)
    plt.show()
