
from functools import lru_cache

import numpy as np
import numpy_indexed as npi
from cached_property import cached_property

from pycomplex.topology import topology_matrix, sign_dtype, index_dtype, transfer_matrix, generate_boundary_indices, sparse_to_elements
from pycomplex.topology.primal import *


def generate_simplex_boundary(simplices):
    """Generate a full set of oriented boundary simplices for each input simplex

    Parameters
    ----------
    simplices : ndarray, [n_simplices, n_corners], index_dtype

    Returns
    -------
    boundary : ndarray, [n_simplices, n_corners, n_dim], index_dtype
        for each simplex, an oriented boundary of simplices is generated

    Notes
    -----
    If a roll represents a parity change depends on even-ness of n-dim!

    """
    n_simplices, n_corners = simplices.shape
    n_dim = n_corners - 1
    assert n_dim > 0

    b = np.empty((n_simplices, n_corners, n_dim), dtype=simplices.dtype)
    for c in range(n_corners):
        b[:, c] = np.roll(simplices, -c, axis=-1)[:, 1:]
        if c % 2 and n_dim % 2:
            b[:, c, [0, 1]] = b[:, c, [1, 0]]   # change the parity by swapping two vertices
    return b


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
                # This can be simpler and faster than the general case, but
                # parity of edges is indeed a genuine special case, as is connectivity of dual edges
                # since we cannot signal boundary orientation by permutations here
                ENn = EN0
                En0 = np.unique(EN0).reshape(-1, 1)
                parity = np.zeros_like(ENn)
                parity[:, 1] = 1
                return En0.astype(index_dtype), ENn.astype(index_dtype), parity_to_orientation(parity)

            full_boundary = generate_simplex_boundary(EN0)
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

            return En0, ENn, parity_to_orientation(parity)


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
        F = E.copy()
        F[:, [1, 0]] = F[:, [0, 1]]
        return type(self).from_simplices(np.where(orientation, E, F))

    def as_2(self):
        if not self.n_dim == 2:
            raise ValueError
        return TopologyTriangular(elements=self.elements, boundary=self._boundary, orientation=self._orientation)

    def fundamental_domains(self):
        """Generate fundamental domains

        Returns
        -------
        domains : ndarray, [n_simplices, n_corners, n_corners - 1 ..., 2, n_dim + 1], index_dtype
            n-th entry of last index refers to n-element indices
            0 and n entry are kinda trivial

        Notes
        -----
        Should implement this for cubes too
        """
        dim = list(range(2, self.n_dim + 2))[::-1]

        shape = np.asarray([self.n_elements[-1]] + dim + [self.n_dim + 1])

        domains = -np.ones(shape, dtype=index_dtype)
        domains[..., -1] = self.range(-1).reshape((-1,) + (1,)*self.n_dim) # they all refer to their parent primal simplex
        IN0 = self.elements[-1]

        for i in range(2, self.n_dim + 1):
            s = shape[:-1].copy()
            s[i:] = 1
            IN0 = generate_simplex_boundary(IN0.reshape(-1, IN0.shape[-1])).reshape(IN0.shape + (-1, ))
            domains[..., -i] = generate_boundary_indices(self.elements[-i], IN0).reshape(s)
        domains[..., 0] = IN0

        return domains

    def subdivide_fundamental(self):
        """Perform a subdivision into fundamental domains"""
        offset = np.cumsum([0] + self.n_elements)[:-1]
        # subdivision is essentially just remapping fundamental-domain n-simplex indices to 0-simplex indices
        simplices = self.fundamental_domains() + offset
        # flip the mirrored side to preserve orientation;
        # FIXME: in n-dim > 3 we still need a fix-orientation somehow...
        simplices[..., 0, :] = np.flip(simplices[..., 0, :], axis=-1)
        return type(self).from_simplices(simplices.reshape(-1, self.n_dim + 1))

    def subdivide_cubical(self):
        """Perform a subdivision of n-simplices into n+1 cubical domains

        Returns
        -------
        TopologyCubical
        """
        N = self.n_elements

        cube_shape = (2, ) * self.n_dim
        cubes = -np.ones((N[-1], self.n_dim + 1) + cube_shape, dtype=index_dtype)
        corners = np.indices(cube_shape).reshape(self.n_dim, -1).T.astype(sign_dtype)

        def idx(c):
            return (Ellipsis, ) + tuple(c)

        domains = self.fundamental_domains()

        cubes[idx(corners[0])] = self.incidence[-1, 0]       # 0 corner is attached to 0-simplices
        cubes[idx(corners[-1])] = self.range(-1)[..., None]  # -1 corner is attached to n-simplices

        if self.n_dim == 2:
            I21 = self.incidence[2, 1]
            cubes[:, :, 1, 0] = np.roll(I21, 1, axis=1)
            cubes[:, :, 0, 1] = np.roll(I21, 2, axis=1)
        elif self.n_dim == 3:
            # work out 3d case and see if it generalizes to n-d
            I32 = self.incidence[3, 2]
            # sum of corners == 2; connect to 2-simplices
            cubes[:, :, 0, 1, 1] = np.roll(I32, 1, axis=1)
            cubes[:, :, 1, 0, 1] = np.roll(I32, 2, axis=1)
            cubes[:, :, 1, 1, 0] = np.roll(I32, 3, axis=1)
            # sum of corners == 1; connect to 1-simplices
            # faces above induce an ordering
            #[1, 0, 0] edge is bordered by [1, 1, 0] and [1, 0, 1] faces
            # can we easily find edge incident to these two faces?
            # of course we have vertex id too
            # group domains by 0 2 and 3-index
            # should get groups of two
            # a, b, c ,d = [domains[:, i] for i in range(4)]

            # from each vertex take the opposing ones by rolling
            # then find the corresponding edge by looking up the pair

            # alternate relation; [1, 0, 0] edge is edge opposite [0, 1, 1] face
            # so find the edge that does not share a particular face; is that easier?

            def edge_containing_faces(l, r):
                # find the edge sharing these two faces
                T = self.matrix(1, 2).tocoo()
                e, f = T.row, T.col
                E = sparse_to_elements(T.T)
                print()

            if True:
                domains = self.fundamental_domains()
                domains_ = domains.reshape(-1, 4)
                _domains = npi.group_by(domains_[:, [0, -1]]).split_array_as_array(domains_)
                edges = generate_simplex_boundary(self.elements[2])
                B = self._boundary[-1]
                Q = edges[B]

                for n in range(len(cubes)):
                    for i in range(4):
                        cube = cubes[n, i]
                        ds = domains[n, i]
                        E = Q[n, i]


                        o = cubes[n, i, 0, 1, 1]
                        l = cubes[n, i, 1, 0, 1]
                        r = cubes[n, i, 1, 1, 0]

                        cubes[n, i, 1, 0, 0] = 0

                        # edge_containing_faces(l, r)

            if False:
                I30 = self.elements[-1]
                for i in range(4):
                    q = np.roll(I30, i, axis=1)
                    c0 = q[:, 0:1]
                    o0 = q[:, 1:]
                    c0 = np.repeat(c0, 3, axis=1)
                    e0 = np.concatenate([c0[..., None], o0[..., None]], axis=2)
                    e0 = e0.reshape(-1, 2)
                    I = npi.indices(np.sort(self.elements[1], axis=1), np.sort(e0, axis=1))
                    I = I.reshape(-1, 3)

                    cubes[:, i, 1, 0, 0] = I[:, 0]
                    cubes[:, i, 0, 1, 0] = I[:, 1]
                    cubes[:, i, 0, 0, 1] = I[:, 2]

            # I31 = self.incidence[3, 1]  # this doesnt exist yet, actually
            # cubes[:, :, 1, 0, 0] = np.roll(I31, 1, axis=1)
        else:
            # in 4d case, we get 5 4-cubes, with 16 corners
            # having a 1-4-6-4-1 distribution
            raise NotImplementedError

        offset = np.cumsum([0] + self.n_elements)
        for c in corners:
            idx = (Ellipsis, ) + tuple(c)
            cubes[idx] += offset[c.sum()]

        from pycomplex.topology.cubical import TopologyCubical
        return TopologyCubical.from_cubes(cubes.reshape((-1,) + cube_shape))


class TopologyTriangular(TopologySimplicial):

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

    def subdivide_cubical(self):
        return super(TopologyTriangular, self).subdivide_cubical().as_2()

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
