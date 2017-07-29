"""Implements a geodesic solver on triangular meshes

References
----------
[1] http://www.multires.caltech.edu/pubs/GeodesicsInHeat.pdf

"""
import numpy as np
import numpy_indexed as npi
import scipy.sparse

from pycomplex.math import linalg
from pycomplex.complex.simplicial import ComplexTriangularEuclidian3


class MyComplex(ComplexTriangularEuclidian3):
    """Subclass that implements the divergence and gradient operators specific to the paper"""

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

    def triangle_edge_vectors(self, O):
        """
        Parameters
        ----------


        Returns
        -------
        ndarray, [n_triangles, 3, 3], float
        """
        grad = self.topology.matrices[0].T
        vecs = grad * self.vertices
        B21 = self.topology._boundary[1]
        O21 = self.topology._orientation[1]
        vecs = vecs[B21]
        if O:
            vecs *= O21[..., None]
        return vecs

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

        vecs = self.triangle_edge_vectors(True)

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
        T01, T12 = self.topology.matrices
        div = T01

        vecs = self.triangle_edge_vectors(False)
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

        diffused = seed * 1.0
        for i in range(1000):
            diffused -= laplacian * diffused / mass / 2000
        print(diffused.min(), diffused.max())
        # return diffused

        # heat = lambda x : mass * x - laplacian * (x * t /1000)
        # operator = scipy.sparse.linalg.LinearOperator(shape=laplacian.shape, matvec=heat)
        # diffused = scipy.sparse.linalg.minres(operator, diffused.astype(np.float64), tol=1e-12)[0]
        print(diffused.min(), diffused.max())
        # return np.log(diffused + 1e-9)

        gradient = -linalg.normalized(self.compute_gradient(diffused))
        # self.plot(facevec=gradient)
        rhs = self.compute_divergence(gradient)
        phi = scipy.sparse.linalg.minres(laplacian, rhs)[0]
        return phi - phi.min()


from examples.subdivision.letter_a import create_letter

letter = create_letter(3).as_23().to_simplicial().as_3()
assert letter.topology.is_oriented

letter = MyComplex(vertices=letter.vertices, topology=letter.topology)

seed = letter.topology.chain(0, fill=0, dtype=np.float)
idx = np.argmin(np.linalg.norm(letter.vertices - [2, 2, -3], axis=1))
print(idx)
seed[idx] = 10
geo = letter.geodesic(seed)

# letter.plot_3d(plot_dual=False, backface_culling=True, plot_vertices=False)
letter.plot_primal_0_form(geo)
