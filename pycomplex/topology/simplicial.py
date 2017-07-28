
from functools import lru_cache

import numpy as np
import numpy_indexed as npi
from cached_property import cached_property

from pycomplex.topology import topology_matrix, sign_dtype, index_dtype, transfer_matrix
from pycomplex.topology.primal import *


def generate_simplex_boundary(simplices):
    """

    Parameters
    ----------
    simplices : ndarray, [n_simplices, n_corners], index_dtype

    Returns
    -------
    parity : ndarray, [n_corners], sign_dtype
    boundary : ndarray, [n_simplices, n_corners, n_dim], index_dtype

    Notes
    -----
    Only in uneven ndim is a roll not a parity change!

    """
    n_simplices, n_corners = simplices.shape
    n_dim = n_corners - 1
    parity = np.ones(n_corners, dtype=sign_dtype) * (n_dim % 2)
    b = np.empty((n_simplices, n_corners, n_dim), dtype=simplices.dtype)
    for c in range(n_corners):
        b[:, c] = np.roll(simplices, -c, axis=-1)[:, 1:]
        parity[c] *= c % 2
    return parity, b


@lru_cache()
def permutation_map(n_dim):
    """Generate permutations map relative to a canonical n-simplex

    Parameters
    ----------
    n_dim : int

    Returns
    -------
    parity : ndarray, [n_combinations], sign_dtype
    permutation : ndarray, [n_combinations, n_corners], sign_dtype
    """
    n_corners = n_dim + 1
    l = list(np.arange(n_corners))
    from pycomplex.math.combinatorial import permutations
    par, perm = zip(*permutations(l))
    par = np.array(par, dtype=sign_dtype)
    perm = np.array(perm, dtype=sign_dtype)
    return par, perm


def relative_simplex_parity(simplices):
    """Compute parity for a set of simplices, relative to a canonical simplex

    Parameters
    ----------
    simplices : ndarray, [n_simplices, n_corners], sign_dtype
        permutation of a set of simplices relative to a canonical simplex

    Returns
    -------
    parity : ndarray, [n_simplices], sign_dtype, {0, 1}
        The parity of each n-simplex, relative to a canonical one
    """
    simplices = np.asarray(simplices)
    n_simplices, n_corners = simplices.shape
    n_dim = n_corners - 1
    parity, permutation = permutation_map(n_dim)
    return parity[npi.indices(permutation, simplices.astype(permutation.dtype))]


class TopologySimplicial(PrimalTopology):
    """Simplicial topology of arbitrary dimension

    """
    def boundary_type(self):
        return TopologySimplicial

    @classmethod
    def from_elements(cls, elements):
        return cls.from_simplices(elements)

    @classmethod
    def from_simplices(cls, simplices):
        """Construct topology from simplex description

        Parameters
        ----------
        simplices : ndarray, [n_simplices, n_dim + 1], int
            EN0 array; simplices given in terms of their vertex indices

        Returns
        -------
        TopologySimplicial
        """
        EN0 = np.asarray(simplices)

        def construct_lower(EN0):
            n_simplices, n_corners = EN0.shape
            n_dim = n_corners - 1
            b_corners = n_corners - 1

            if n_dim == 1:
                # FIXME: does this special case add much at all?
                EnN = EN0
                En0 = np.unique(EN0).reshape(-1, 1)
                EnN = EnN.reshape((n_simplices,) + (n_dim, 2))
                parity = np.zeros_like(EnN)
                parity = np.logical_xor(parity, [[[0, 1]]])
                orientation = parity * 2 - 1
                return En0.astype(index_dtype), EnN.astype(index_dtype), orientation.astype(sign_dtype)

            b_parity, full_boundary = generate_simplex_boundary(EN0)
            boundary_corners = full_boundary.reshape(n_simplices * n_corners, b_corners)
            sorted_boundary_corners, permutation = sort_and_argsort(boundary_corners, axis=1)
            permutation = permutation.astype(sign_dtype)

            index = npi.as_index(sorted_boundary_corners)

            # this is the easy one; identical sorted corners are a single boundary
            ENn = index.inverse
            ENn = ENn.reshape(n_simplices, n_corners).astype(index_dtype)

            # pick a convention for the boundary element for each group of identical corners
            En0 = boundary_corners[index.index]
            canonical_permutation = permutation[index.index]

            # broadcast that picked value to the group
            canonical_permutation = canonical_permutation[index.inverse]
            # find relative permutation of vertices of each element
            relative_permutation = relative_permutations(permutation, canonical_permutation)
            # derive relative parity of each n-cube to the one picked as defining neutral convention in En0
            relative_parity = relative_simplex_parity(relative_permutation)

            parity = relative_parity.reshape(ENn.shape)
            parity = np.logical_xor(b_parity[None, :], parity)

            orientation = parity * 2 - 1
            return En0, ENn, orientation.astype(sign_dtype)


        n_simplices, n_pts = EN0.shape
        n_dim = n_pts - 1

        E = np.zeros((n_pts), dtype=np.object)
        E[...] = None
        E[n_dim] = EN0
        O = np.zeros(n_dim, dtype=object)
        B = np.zeros(n_dim, dtype=object)

        # p = simplex_parity(EN0)
        while True:
            EN0, ENn, ONn = construct_lower(EN0)
            n_simplices, n_pts = EN0.shape
            n_dim = n_pts - 1
            N = n_dim
            E[N] = EN0
            O[N] = ONn
            B[N] = ENn

            if n_dim == 0:
                break

        return cls(elements=E, orientation=O, boundary=B)

    def fix_orientation(self):
        """Try to find a consistent orientation for all simplices

        Returns
        -------
        oriented : topology
            same topology but with oriented simplices

        Raises
        ------
        ValueError
            if the topology is not orientable
        """
        orientation = self.relative_orientation()
        E = self.elements[-1]
        return type(self).from_simplices(np.where(orientation, E, E[:, ::-1]))


class TopologyTriangular(TopologySimplicial):

    @classmethod
    def from_simplices(cls, triangles):
        """Construct topology from triangle description as sets of vertex indices

        Note that this is quite a bit simpler than the general case

        Also, we enforce a relation between I21 and I20;
        that is, edges and vertices are always opposite

        seek to do similar things for tets
        """
        E20 = np.asarray(triangles, dtype=index_dtype)
        if not E20.ndim==2 and E20.shape[1] == 3:
            raise ValueError('Expected integer triples')

        E00 = np.unique(E20).reshape(-1, 1)

        L = np.roll(E20, -1, axis=1)
        R = np.roll(E20, +1, axis=1)
        # I210 is triangles expressed as edges expressed as vertex indices; [n_triangles, 3, 2]
        E210 = np.concatenate([L[..., None], R[..., None]], axis=-1)
        E210 = np.sort(E210, axis=-1)

        E10, E21 = npi.unique(E210.reshape(-1, 2), return_inverse=True)
        E21 = E21.reshape(-1, 3).astype(index_dtype)

        # special case rule for triangle orientations
        O21 = ((L < R) * 2 - 1).astype(sign_dtype)
        O10 = np.ones((len(E10), 2), sign_dtype)
        O10[:, 0] *= -1

        # construct grid of all element representations
        E = [E00, E10, E20]
        B = [E10, E21]
        O = [O10, O21]

        return cls(elements=E, orientation=O, boundary=B)

    def subdivide(coarse):
        """Loop subdivision on triangles

        Returns
        -------
        TopologyTriangular
            loop-subdivided topology

        Notes
        -----
        This function assumes the presence of tri-edge and tri-vertex incidence matrices,
        ordered such that corresponding columns denote opposite edges and vertices

        It uses the from_simplices constructor; this means edge orientation is not preserved

        """
        I21 = coarse.incidence[2, 1]
        I20 = coarse.incidence[2, 0]

        # edge edge spawns a new vertex; each face a new face
        I20c = I21 + coarse.n_vertices
        I20s = -np.ones((coarse.n_triangles, 4, 3), index_dtype)
        I20s[:, 0] = I20c                           # middle triangle
        I20s[:, 1:, 0] = I20                        # for other three, first vert is parent vert
        I20s[:, 1:, 1] = np.roll(I20c, +1, axis=1)  # use opposing vert-edge relation in I21/I20 to fill in the rest
        I20s[:, 1:, 2] = np.roll(I20c, -1, axis=1)

        fine = type(coarse).from_simplices(I20s.reshape(-1, 3))

        # # build up transfer operators; only edge-edge has some nontrivial logic
        # I10f = np.sort(fine.corners[1], axis=1)
        # # filter down edges for those having connection with original vertices
        # fine_idx = np.flatnonzero(I10f[:, 0] < coarse.n_elements[0])
        # I10f = I10f[fine_idx]
        # # highest vertex; translate to corresponding n-cube on the coarse level
        # coarse_idx = I10f[:, -1] - coarse.n_elements[0]
        #
        # transfers = [
        #     (coarse.range(0), coarse.range(0)),
        #     (fine_idx, coarse_idx),
        #     (fine.range(2), np.repeat(coarse.range(2), 4)),
        # ]
        # # transfers is a list of arrays
        # # where each entry i is an ndarray, [sub.n_elements[i]], index_dtype, referring to the parent element
        fine.transfer_matrices = coarse.subdivide_transfer(fine)
        fine.parent = coarse

        return fine

    def subdivide_transfer(coarse, fine):
        # build up transfer operators; only edge-edge has some nontrivial logic
        I10f = np.sort(fine.corners[1], axis=1)
        # filter down edges for those having connection with original vertices
        fine_idx = np.flatnonzero(I10f[:, 0] < coarse.n_elements[0])
        I10f = I10f[fine_idx]
        # highest vertex; translate to corresponding n-cube on the coarse level
        coarse_idx = I10f[:, -1] - coarse.n_elements[0]

        transfers = [
            (coarse.range(0), coarse.range(0)),
            (fine_idx, coarse_idx),
            (fine.range(2), np.repeat(coarse.range(2), 4)),
        ]
        # transfers is a list of arrays
        # where each entry i is an ndarray, [sub.n_elements[i]], index_dtype, referring to the parent element
        transfer_matrices = [transfer_matrix(*t, shape=(fine.n_elements[n], coarse.n_elements[n]))
                                 for n, t in enumerate(transfers)]
        return transfer_matrices

    def subdivide_direct(self):
        """Subdivide triangular topology in a direct manner, without a call to from_simplices

        This allows us to impart a little more structure to the subdivision,
        leading to particularly simple forms of the transfer operators between levels,
        and should be more efficient as well

        Returns
        -------
        TopologyTriangular
            loop-subdivided topology
        """
        E21 = self.incidence[2, 1]
        E20 = self.incidence[2, 0]
        E10 = self.incidence[1, 0]
        O21 = self.orientation[1]
        O10 = self.orientation[0]
        N0, N1, N2 = self.n_elements

        # positive and negatively rolled E21 info is needed several times; compute once
        E21p = np.roll(E21, +1, axis=-1)
        O21p = np.roll(O21, +1, axis=-1)
        E21m = np.roll(E21, -1, axis=-1)
        O21m = np.roll(O21, -1, axis=-1)

        # create edge edges; two for each parent edge
        eE10  = np.empty((N1, 2, 2), index_dtype)
        eO10  = np.empty((N1, 2, 2), sign_dtype)
        new_edge_vertex = np.arange(N1, dtype=index_dtype) + N0
        eE10[:,0,0] = E10[:,0]
        eE10[:,0,1] = new_edge_vertex
        eE10[:,1,0] = new_edge_vertex
        eE10[:,1,1] = E10[:,1]
        # edge-edge sign info
        eO10[1,:,0,0] = +O10[:,0]
        eO10[1,:,0,1] = -O10[:,0]
        eO10[1,:,1,0] = -O10[:,1]
        eO10[1,:,1,1] = +O10[:,1]

        # 3 new edges per face
        fE10 = np.empty((N2, 3, 2), index_dtype) # edge-vertex info added as consequence of faces
        fO10 = np.empty((N2, 3, 2), sign_dtype)
        fE10[:,:,0] = E21m + N0     # edge-vertices can be found by rotating and offsetting parent edges
        fE10[:,:,1] = E21p + N0
        # orientation part
        fO10[:,:,0] = -1  # this is a sign convention we are free to choose; neg-to-pos edges are maintained globally
        fO10[:,:,1] = +1

        # 4 new faces per face
        fE21  = np.empty((N2, 4, 3), index_dtype)
        fO21  = np.empty((N2, 4, 3), sign_dtype)

        # add middle (zero) tri
        # add middle connectivity; three new edges for each face
        fE21[:,0,:] = np.arange(3*N2, dtype=index_dtype).reshape((N2, 3)) + N1*2
        fO21[:,0,:] = 1                                  # orientation of middle tri is given by convention

        # first edge of each outer tri is connected to the middle; copy index, invert sign
        fE21[:,1:,0] =  fE21[:,0,:]
        fO21[:,1:,0] = -fO21[:,0,:]

        # now do hard part; hook up corner triangles
        plusverts = E10[E21,1]
        # getting the right indices requires some logic; look up if triangle vertex is on the plus end of edge or not;
        # if so, it has the higher (uneven) index
        fE21[:,1:,+1] = E21m*2 + (np.roll(plusverts, -1, axis=1) == E20)*1
        fE21[:,1:,-1] = E21p*2 + (np.roll(plusverts, +1, axis=1) == E20)*1
        # weights are simply inherited, using the same roll logic
        fO21[:,1:,+1] = O21m
        fO21[:,1:,-1] = O21p

        # E20 is implied by E21; but much easier to calc by subdivision too
        sE20 = np.empty((N2, 4, 3), index_dtype)
        sE20[:,0 , :] = E21[0] + N0  # middle tri; translate parent edge index to edge-vertex
        sE20[:,1:, 0] = E20          # corner tri; inherit from parent
        sE20[:,1:,+1] = E21p[0] + N0 # middle edge connections; same as fE10; rotate parent edge, translate to edge-vert
        sE20[:,1:,-1] = E21m[0] + N0

        sE20 = sE20.reshape(-1, 3)
        sE21 = fE21.reshape(-1, 3)
        sO21 = fO21.reshape(-1, 3)

        sE10 = np.concatenate((
            eE10.reshape(-1, 2),
            fE10.reshape(-1, 2)), axis=0)
        sO10 = np.concatenate((
            eO10.reshape(-1, 2),
            fO10.reshape(-1, 2)), axis=0)

        O = np.zeros_like(self.orientation)
        O[0] = sO10
        O[1] = sO21
        E = np.zeros_like(self.elements)
        E[...] = None
        E[2, 0] = sE20
        E[2, 1] = sE21
        E[1, 0] = sE10

        return TopologyTriangular(elements=E, orientation=O)

    @staticmethod
    def subdivide_direct_transfer(coarse, fine):
        """Transfer operators belonging to direct subdivision logic; note their simplicity"""
        transfers = [
            (fine.range(0)[:coarse.n_elements[0]],   coarse.range(0)),
            (fine.range(1)[:coarse.n_elements[1]*2], np.repeat(coarse.range(1), 2)),
            (fine.range(2),                          np.repeat(coarse.range(2), 4)),
        ]

    def to_cubical(self):
        """Convert the triangular complex into a cubical complex,
        by forming 3 quads from each triangle"""
        N0, N1, N2 = self.n_elements
        I20 = self.incidence[2, 0]
        I21 = self.incidence[2, 1]

        Q20 = -np.ones((N2, 3, 2, 2), dtype=index_dtype)
        Q20[:, :, 0, 0] = self.range(2)[:, None] + N0 + N1
        Q20[:, :, 0, 1] = np.roll(I21, -1, axis=1) + N0
        Q20[:, :, 1, 0] = np.roll(I21, +1, axis=1) + N0
        Q20[:, :, 1, 1] = I20

        from pycomplex.topology.cubical import TopologyCubical2
        return TopologyCubical2.from_cubes(Q20.reshape(N2 * 3, 2, 2))

    # some convenient named accessors
    def face_edge(self):
        return self.T12
    def edge_vertex(self):
        return self.T01

    # simplices described in terms of vertex indices
    @property
    def vertices(self):
        return self.elements[0]
    @property
    def edges(self):
        return self.elements[1]
    @property
    def triangles(self):
        return self.elements[2]

    # number of simplices of each dimension
    @property
    def n_vertices(self):
        return self.n_elements[0]
    @property
    def n_edges(self):
        return self.n_elements[1]
    @property
    def n_triangles(self):
        return self.n_elements[2]


class TopologyTetrahedral(TopologySimplicial):

    @classmethod
    def from_simplices(cls, simplices):
        """Construct topology from tetrahedral description as sets of vertex indices

        Note that this is quite a bit simpler than the general case

        Also, we enforce a relation between I32 and I30;
        that is, faces and vertices are always opposite
        """
        E30 = np.asarray(simplices, dtype=index_dtype)
        if not E30.ndim==2 and E30.shape[1] == 4:
            raise ValueError('Expected [n, 4] array')

        E00 = np.unique(E30).reshape(-1, 1)

        # P3, E320 = generate_boundary(E30, n=3, axis=-1)
        P3, E320 = generate_simplex_boundary(E30)
        # A, B, C, D = [np.roll(E30, i+1, axis=1) for i in range(4)]
        # E320 = np.concatenate([A[..., None], B[..., None], C[..., None], D[..., None]], axis=-1)

        P = relative_simplex_parity(E320.reshape(-1, 3))
        E320 = np.sort(E320, axis=-1)


        L = np.roll(E20, -1, axis=1)
        R = np.roll(E20, +1, axis=1)
        # I210 is triangles expressed as edges expressed as vertex indices; [n_triangles, 3, 2]
        E210 = np.concatenate([L[..., None], R[..., None]], axis=-1)
        E210 = np.sort(E210, axis=-1)

        E10, E21 = npi.unique(E210.reshape(-1, 2), return_inverse=True)
        E21 = E21.reshape(-1, 3).astype(index_dtype)

        # special case rule for triangle orientations
        O21 = ((L < R) * 2 - 1).astype(sign_dtype)
        O10 = np.ones((len(E10), 2), sign_dtype)
        O10[:, 0] *= -1

        # construct grid of all element representations
        E = [E00, E10, E20]
        B = [E10, E21]
        O = [O10, O21]

        return cls(elements=E, orientation=O, boundary=B)
