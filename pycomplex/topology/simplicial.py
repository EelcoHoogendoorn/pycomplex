
import numpy as np
import numpy_indexed as npi
from cached_property import cached_property

from pycomplex.topology import topology_matrix, sign_dtype, index_dtype, transfer_matrix
from pycomplex.topology.topology import PrimalTopology


def combinations(arr, n, axis):
    from pycomplex.math.combinatorial import combinations

    arr = np.asarray(arr)
    idx = np.arange(arr.shape[axis], dtype=index_dtype)
    parity, comb = zip(*combinations(idx, n))
    comb = np.asarray((list(comb)))
    return np.asarray(parity), arr.take(comb, axis=axis)


def simplex_parity(simplices):
    """Compute parity for a set of simplices
    result is 0 for sorted indices, 1 for a permutation thereof
    """
    n_simplices, n_dim = simplices.shape
    from pycomplex.math.combinatorial import permutations_map
    parity, permutation = permutations_map(n_dim)
    sorter = np.argsort(simplices, axis=-1)
    return parity[npi.indices(permutation.astype(sign_dtype), sorter.astype(sign_dtype))]


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
            n_simplices, n_pts = EN0.shape
            n_dim = n_pts - 1

            b_parity, full_boundary = combinations(EN0, n_dim, axis=-1)
            full_boundary = full_boundary.reshape(-1, n_dim)

            sorted_boundary = np.sort(full_boundary, axis=-1)
            index = npi.as_index(sorted_boundary)

            En0 = index.unique
            ENn = index.inverse
            ENn = ENn.reshape(-1, n_pts).astype(index_dtype)

            # if n_dim == 1:
            #     orientation = [-1, +1] * n_simplices
            # else:
            s_parity = simplex_parity(EN0)
            s_parity = np.zeros_like(s_parity)
            parity = np.logical_xor(b_parity[None, :], s_parity[:, None])
            # parity = b_parity

            # b_parity = simplex_parity(full_boundary).reshape(parity.shape)
            # parity = np.logical_xor(parity, b_parity)

            # from pycomplex.combinatorial import permutations_map
            # parity, permutation = permutations_map(n_dim)
            # orientation = parity[npi.indices(permutation.astype(np.int8), sorter.astype(np.int8))]
            # orientation = np.logical_xor(b_parity, np.repeat(s_parity, n_pts)) * 2 - 1
            # orientation = b_parity * 2 - 1

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

    # @cached_property
    # def boundary(self):
    #     """Return n-1-topology representing the boundary"""
    #     b_idx = self.boundary_indices()
    #     if len(b_idx) == 0:
    #         return None
    #     # construct boundary
    #     B = type(self).from_simplices(self.elements[-2][b_idx])
    #     # compute correspondence idx for all parent and boundary elements
    #     B.parent_idx = self.find_correspondence(B)
    #     B.parent = self
    #     return B

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
            raise ValueError('Expectect integer triples')

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
        fine.transfers = transfers
        fine.transfer_matrices = [transfer_matrix(*t, shape=(fine.n_elements[n], coarse.n_elements[n]))
                                 for n, t in enumerate(transfers)]
        fine.parent = coarse

        return fine

    @staticmethod
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
