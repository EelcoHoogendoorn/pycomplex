"""A composite complex is a set of subcomplexes stitches together

These subcomplexes might be of a differing type eventually;
for now, focus on getting the triangular case working smoothly

should be sufficiently flexible to be used for escheresque purposes,
yet simple enough to be easily tested

ideally, compositeComplex and Complex share the same interface, as much as possible
"""

import numpy as np
import numpy_indexed as npi
from cached_property import cached_property


class CompositeComplex(object):
    """Composite complex repeating a single patch"""

    def __init__(self, base, topology):
        """

        Parameters
        ----------
        base : Complex
            complex to be tiled
        topology :
            describes how the different patches are connected
        """
        self.base = base
        self.topology = topology

    @property
    def n_elements(self):
        self.topology.n_elements

    def form(self, n, fill=None, dtype=np.float):
        f = np.empty()

    @cached_property
    def boundary_info(self):
        """Return terms describing how the patch boundary stitches together

        Returns
        -------
        vertices : ndarray, [n_terms], int
            the vertex index this boundary term applies to
            single number for edge vertices; multiple entries for corner vertices
        patch_i : ndarray, [n_terms], int
            relative element in quotient group to reach opposing element
            how current index relates to other side of the term
        patch_j : ndarray, [n_terms], int
            relative element in quotient group to reach opposing element
            how current index relates to other side of the term
        sub : ndarray, [n_terms], int
            relative subgroup transform.
            only needed by normal transformation so far to get transformations
        """
        # NOTE: need to build this up in a manner that does not depend on escheresque particulars,
        # yet is still compatible with them

        # #
        # vi = self.group.vertex_incidence            # [n_vertex_entries, 4]
        # ei = self.group.edge_incidence              # [n_edge_entries, 4]
        #
        # # these are the vertex indices for all edges and corners of the triangle
        # bv = self.triangle.boundary_vertices        # [3, 1]
        # be = self.triangle.boundary_edge_vertices   # [3, n_boundary_edges]
        #
        # def broadcast(a, b):
        #     shape = len(b), len(a), 3
        #     a = np.broadcast_to(a[None], shape)
        #     b = np.broadcast_to(b[:, None], shape[:-1])
        #     return np.concatenate([b.reshape(-1, 1), a.reshape(-1, 3)], axis=1)
        #
        # v = [broadcast(a, b) for a, b in zip(npi.group_by(vi[:, 0]).split(vi[:, 1:]), bv)]
        # e = [broadcast(a, b) for a, b in zip(npi.group_by(ei[:, 0]).split(ei[:, 1:]), be)]
        #
        # return np.concatenate(v + e, axis=0)

    @cached_property
    def stitcher_d2_flat(self):
        """Compute sparse matrix that applies stitching to d2 forms
        acts on flattened d2-form

        Returns
        -------
        sparse matrix
        """
        info = self.boundary_info
        info = info[~np.logical_and(info[:, 1] == info[:, 2], info[:, 3] == 0)]   # remove diagonal
        # transform to flat idx
        r = self.ravel(info[:, 1], info[:, 0])
        c = self.ravel(info[:, 2], info[:, 0])
        import scipy.sparse
        def sparse(r, c):
            n = np.prod(self.shape_p0)
            return scipy.sparse.coo_matrix((np.ones_like(r), (r, c)), shape=(n, n))
        return sparse(r, c)

    def stitcher_d2(self):
        """

        Returns
        -------
        callable (d2) -> (d2)
            function that stitches d2 forms
        """

    def boundary(self):
        """not sure yet how to represent"""

if __name__ == "__main__":
    from pycomplex import synthetic
    quad = synthetic.n_cube(2)
    from pycomplex.topology.simplicial import TopologyTriangular
    tris = [
        [0, 1, 2],
        [1, 2, 3]
    ]
    topology = TopologyTriangular.from_simplices(tris)
    triangle = synthetic.n_simplex(2, equilateral=False)
    composite = CompositeComplex(triangle, topology)