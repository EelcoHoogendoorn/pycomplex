"""Implements a geodesic solver

References
----------
http://www.multires.caltech.edu/pubs/GeodesicsInHeat.pdf

"""
import numpy as np
import numpy_indexed as npi
import scipy.sparse

from pycomplex.math import linalg
from pycomplex.complex.simplicial import ComplexTriangularEuclidian3


class MyComplex(ComplexTriangularEuclidian3):

    # def edges(self):
    #     E20 = self.topology.incidence[2, 0]
    #     E10 = self.topology.incidence[1, 0]
    #     E21 = self.topology.incidence[2, 1]
    #
    #     edges = E10[E21]
    #     # edges = edges.reshape(-1, 3, 2)
    #     return edges

    def remap_edges(self, field):
        """given a quantity computed on each triangle-edge, sum the contributions from each adjecent triangle

        Parameters
        ----------
        field : ndarray, [n_triangles, 3], float

        Returns
        -------
        field : ndarray, [n_edges], float
        """
        B12 = self.topology._boundary[1]   # [n_triangles, 3], edge indices
        _, field = npi.group_by(B12.flatten()).sum(field.flatten())
        return field

    def hodge_edge(self):
        """Compute edge hodge based on cotan formula; corresponds to circumcentric calcs"""
        cotan = 1 / np.tan(self.compute_face_angles)
        return self.remap_edges(cotan) / 2

    def laplacian_vertex(self):
        """Compute cotan/area gradient based laplacian, mapping primal 0-forms to dual n-forms"""
        hodge = self.hodge_edge()
        hodge = scipy.sparse.dia_matrix((hodge, 0), shape=(len(hodge),) * 2)
        grad = self.topology.matrices[0].T
        div = grad.T
        return div * hodge * grad

    def compute_gradient(self, field):
        """compute gradient of scalar function on vertices, evaluated on faces

        Parameters
        ----------
        field : ndarray, [n_vertices], float
            primal 0-form

        Returns
        -------
        gradient : ndarray, [n_triangles, 3], float
            triplet of dual 0-forms
        """
        E20 = self.topology.incidence[2, 0]
        E21 = self.topology.incidence[2, 1]

        grad = self.topology.matrices[0].T
        vecs = grad * self.vertices
        vecs = vecs[E21]

        normals, triangle_area = linalg.normalized(self.triangle_normals(), return_norm=True)
        gradient = (field[E20][:, :, None] * np.cross(normals[:, None, :], vecs)).sum(axis=1)
        return gradient / (2 * triangle_area[:, None])

    def compute_divergence(self, field):
        """Compute divergence of vector field at faces at vertices

        Parameters
        ----------
        field : ndarray, [n_faces, 3], float

        Returns
        -------
        ndarray, [n_vertices], float

        """
        E21 = self.topology.incidence[2, 1]
        T01, T12 = self.topology.matrices

        grad = T01.T
        div = grad.T

        vecs = grad * self.vertices
        vecs = vecs[E21]    # [n_faces, 3, 3]
        inner = linalg.dot(vecs, field[:, None, :])
        cotan = 1 / np.tan(self.compute_face_angles)

        return div * self.remap_edges(inner * cotan) / 2

    def geodesic(self, seed, m=1):
        """Compute geodesic distance map

        Parameters
        ----------
        seed : ndarray, [n_vertices], float
            primal 0-form indicating starting point

        Returns
        -------
        ndarray, [n_vertices], float
            distance from the seed point

        Notes
        -----
        http://www.multires.caltech.edu/pubs/GeodesicsInHeat.pdf
        """
        laplacian = self.laplacian_vertex()
        mass = self.vertex_areas()
        t = self.edge_lengths().mean() ** 2 * m
        heat = lambda x : mass * x - laplacian * (x * t / 1000)
        operator = scipy.sparse.linalg.LinearOperator(shape=laplacian.shape, matvec=heat)

        diffused = scipy.sparse.linalg.minres(operator, seed.astype(np.float64), tol=1e-12)[0]
        print(diffused.min(), diffused.max())
        return diffused
        gradient = -linalg.normalized(self.compute_gradient(diffused))
        # self.plot(facevec=gradient)
        rhs = self.compute_divergence(gradient)
        phi = scipy.sparse.linalg.minres(laplacian, rhs)[0]
        return phi - phi.min()


from examples.subdivision.letter_a import create_letter

letter = create_letter().as_23().to_simplicial().as_3()
assert letter.topology.is_oriented

letter = MyComplex(vertices=letter.vertices, topology=letter.topology)

seed = letter.topology.chain(0)
seed[0] = 1
geo = letter.geodesic(seed)

# letter.plot_3d(plot_dual=False, backface_culling=True, plot_vertices=False)
letter.plot_primal_0_form(geo)