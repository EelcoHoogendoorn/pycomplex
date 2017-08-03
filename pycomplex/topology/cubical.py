from functools import lru_cache

import numpy as np
import numpy_indexed as npi

import pycomplex.math.combinatorial
from pycomplex.topology import index_dtype, sign_dtype, transfer_matrix, generate_boundary_indices
from pycomplex.topology.primal import *


def generate_cube_boundary(cubes, degree=1, return_corners=False):
    """Given a set of n-cubes, construct overcomplete boundary set of n-d-cubes
    
    every n-cube is bounded on n-dim dimensions; but the d-boundary has permutations given by pascals triangle
    f.i., 4-cube has 6 types of 2-cubes incident to it; every unique combination of 2 axes grabbed from those 4

    Parameters
    ----------
    cubes : ndarray, [n_elements, (2,) * n], int
        Vertex indices of n_elements n-cubes
    degree : int
        degree of the boundary
        dimension of the returned cubes is lowered by this number relative to the input cubes

    Returns
    -------
    boundary : ndarray, [n_elements, n_combinations, (2,)**d, (2,)**b]
        Vertex indices of 2**d * n_elements (n-d=b)-cubes
        if degree == 1, the boundary of each n-cube is an oriented set of n-1-cubes
    """
    cubes = np.asarray(cubes)
    n_dim = cubes.ndim - 1
    b_dim = n_dim - degree
    n_elements = cubes.shape[0]

    # axes to pull forward to generate all incident n-d-cubes
    axes_parity, axes_list = zip(*pycomplex.math.combinatorial.combinations(np.arange(n_dim), degree))
    n_combinations = len(axes_list)

    boundary = np.empty((n_elements, n_combinations) + (2,) * degree + (2,) * (b_dim), dtype=cubes.dtype)
    corners = []
    for i, (p, axes) in enumerate(zip(axes_parity, axes_list)):
        s_view = cubes
        corner = [0] * n_dim
        for j, axis in enumerate(axes): # permute the relevant subrange of axes
            s_view = np.moveaxis(s_view, axis + 1, j + 1)
            corner[axis] = 1
        corners.append(tuple(corner))
        if p:
            s_view = np.flip(s_view, axis=-1)
        boundary[:, i] = s_view

    if degree == 1:
        # flip the parity of elements on one side of the cube
        boundary[:, :, 1] = np.flip(boundary[:, :, 1], axis=-1)

    if return_corners:
        return boundary, corners
    else:
        return boundary


# def generate_cube_boundary_alt(cubes):
#     """Given a set of n-cubes, construct overcomplete boundary set of n-d-cubes
#
#     every n-cube is bounded on n-dim dimensions; but the d-boundary has permutations given by pascals triangle
#     f.i., 4-cube has 6 types of 2-cubes incident to it; every unique combination of 2 axes grabbed from those 4
#
#     Parameters
#     ----------
#     cubes : ndarray, [n_elements, (2,) * n], int
#         Vertex indices of n_elements n-cubes
#
#     Returns
#     -------
#     boundary : ndarray, [n_elements, (2,) * n, (2,)**(n-1)]
#         Vertex indices of 2**d * n_elements (n-1)-cubes
#         the boundary of each n-cube is an oriented set of n-1-cubes
#     """
#     cubes = np.asarray(cubes)
#     n_dim = cubes.ndim - 1
#     b_dim = n_dim - degree
#     n_elements = cubes.shape[0]
#
#     # axes to pull forward to generate all incident n-d-cubes
#     axes_parity, axes_list = zip(*pycomplex.math.combinatorial.combinations(np.arange(n_dim), degree))
#     n_combinations = len(axes_list)
#
#     boundary = np.empty((n_elements, n_combinations) + (2,) * degree + (2,) * (b_dim), dtype=cubes.dtype)
#
#     for i, (p, axes) in enumerate(zip(axes_parity, axes_list)):
#         s_view = cubes
#         for j, axis in enumerate(axes):  # permute the relevant subrange of axes
#             s_view = np.moveaxis(s_view, axis + 1, j + 1)
#         if p:
#             s_view = np.flip(s_view, axis=-1)
#         boundary[:, i] = s_view
#
#     if degree == 1:
#         # flip the parity of elements on one side of the cube
#         boundary[:, :, 1] = np.flip(boundary[:, :, 1], axis=-1)
#
#     return boundary


@lru_cache()
def permutation_map(n_dim, rotations=True):
    """Generate cubical permutation map

    Parameters
    ----------
    n_dim : int
    rotations : bool
        if False, only mirror symmetries are included in the map, and rotated cubes will not be recognized as valid cubes
        This will make a huge efficiency difference for high dimensional regular grids, that will not contain
        any rotations anyway

    Returns
    -------
    parity : ndarray, [n_combinations], sign_dtype
    permutation : ndarray, [n_combinations, n_corners], sign_dtype

    """
    n_corners = 2 ** n_dim
    assert n_corners < 128  # if not, we need to increase our dtype; but yeah...
    # create canonical n-cube
    cube = np.arange(n_corners, dtype=sign_dtype).reshape((2,) * n_dim)
    import pycomplex.math.combinatorial
    permutation = list(range(n_dim))
    if rotations:
        permutations = pycomplex.math.combinatorial.permutations(permutation)
    else:
        permutations = [(0, permutation)]
    # flip around each axis and make all possible axes swaps;
    # xorred combination of flips and transposes is parity
    def flip(arr, axes):
        for i in np.flatnonzero(axes):
            arr = np.flip(arr, axis=i)
        return arr

    parity = []
    permutation = []
    for permutation_parity, permutation_axes in permutations:
        # apply permutation to cube
        permuted_cube = np.transpose(cube, permutation_axes)
        # loops over the corners of a cube
        for c in cube.flatten():
            flip_axes = np.array(np.unravel_index(c, cube.shape))
            # apply flips along all indicated axes
            flipped_cube = flip(permuted_cube, flip_axes)
            flip_parity = flip_axes.sum()

            permutation.append(flipped_cube.flatten())
            parity.append((permutation_parity + flip_parity) % 2)
    return np.array(parity, dtype=sign_dtype), np.array(permutation, dtype=sign_dtype)


def relative_cube_parity(cubes):
    """Find the parity of a set of n-cubes

    Parameters
    ----------
    cubes : ndarray, [n_cubes] + [2] * ndim, sign_dtype
        permutation of a set of n-cubes relative to canonical cube

    Returns
    -------
    parity : ndarray, [n_cubes], sign_dtype, {0, 1}
        The parity of each n-cube, relative to a canonical one
    """
    cubes = np.asarray(cubes)
    n_dim = cubes.ndim - 1
    n_cubes = len(cubes)
    corners = cubes.reshape(n_cubes, -1)

    parity, permutation = permutation_map(n_dim)

    try:
        return parity[npi.indices(permutation, corners.astype(permutation.dtype))]
    except KeyError as e:
        raise ValueError("Ceci n'est pas une n-cube")


class TopologyCubical(PrimalTopology):
    """N-dimensional regular topology, where each element is an n-cube"""

    def boundary_type(self):
        return TopologyCubical

    @classmethod
    def from_elements(cls, elements):
        return cls.from_cubes(cubes=elements)

    @classmethod
    def from_cubes(cls, cubes):
        """

        Parameters
        ----------
        cls : type(TopologyCubical)
        cubes : ndarray, [n_elements, (2,) * n_dim], index_type
            vertex indices defining the n_cubes

        Returns
        -------
        TopologyCubical

        """
        def lower(EN0):
            """Construct n-1 boundary"""
            n_dim = EN0.ndim - 1
            b_dim = n_dim - 1
            n_elements = EN0.shape[0]
            b_shape = (2,) * b_dim
            b_corners = 2 ** b_dim

            if n_dim == 1:
                # special case for E10
                EnN = EN0
                En0 = np.unique(EN0)
                EnN = EnN.reshape((n_elements, n_dim, 2))
                parity = np.zeros_like(EnN)
                parity[..., 1] = 1
                return En0.astype(index_dtype), EnN.astype(index_dtype), parity_to_orientation(parity)

            # generate boundary elements
            En0_all = generate_cube_boundary(EN0, degree=1)
            En0_all = En0_all.reshape((-1,) + b_shape)

            # get mapping to unique set of boundary elements, by considering sorted corners
            corners = En0_all.reshape((-1, b_corners))
            sorted_corners, permutation = sort_and_argsort(corners, axis=1)
            permutation = permutation.astype(sign_dtype)

            index = npi.as_index(sorted_corners)

            # boundary operator is always the same simple matter
            EnN = index.inverse
            EnN = EnN.reshape((n_elements, n_dim, 2))   # retain some structure in boundary operator

            # pick one element from each group of corner-identical elements as reference;
            # note: this takes first; need not be canonical in any sense!
            En0 = En0_all[index.index]
            canonical_permutation = permutation[index.index]
            # broadcast that picked value to the entire set
            canonical_permutation = canonical_permutation[index.inverse]
            # find relative permutation of vertices
            relative_permutation = relative_permutations(permutation, canonical_permutation)
            # derive relative parity of each n-cube to the one picked as defining neutral convention in En0
            relative_parity = relative_cube_parity(relative_permutation.reshape((-1,) + b_shape))

            # parity relative to canonical boundary element
            parity = relative_parity.reshape(EnN.shape)

            return En0.astype(index_dtype), EnN.astype(index_dtype), parity_to_orientation(parity)

        EN0 = np.asarray(cubes)
        n_dim = EN0.ndim - 1
        if not EN0.shape[1:] == (2,) * n_dim:
            raise ValueError

        E = np.zeros((n_dim + 1), dtype=np.object)
        E[...] = None
        E[n_dim] = EN0

        B = [None] * n_dim
        O = [None] * n_dim


        while True:
            if n_dim == 0:
                break
            EN0, ENn, ONn = lower(EN0)

            n_dim = EN0.ndim - 1
            N = n_dim

            E[N] = EN0
            O[N] = ONn
            B[N] = ENn


        return cls(elements=E, boundary=B, orientation=O)

    def subdivide(self):
        """Cubical topology subdivision; general n-d case

        Returns
        -------
        TopologyCubical
            subdivided topology where each n-cube has been split into n**2 n_cubes

        Notes
        -----
        currently, the transfer operators are set as a side effect of calling this function; not super happy with that
        """
        E = self.elements
        cubes = E[-1]
        n_dim = self.n_dim

        # construct all levels of boundaries of cubes, plus their indices
        B = [generate_cube_boundary(cubes, d) for d in reversed(range(self.n_dim + 1))]
        # FIXME: do we always want to be doing this during construction to fill out our incidence matrix?
        I = [pycomplex.topology.generate_boundary_indices(e, b) for e, b in zip(E, B)]

        # convert boundary element indices to new vertex indices on the refined level
        # account for the fact that new verts will be concatted, so index in next level shifts up
        O = np.cumsum([0] + [len(i) for i in E])
        for i, o in zip(I, O):
            i += o

        new_cubes = self.subdivide_specialized(I)
        new_cubes = new_cubes.reshape((-1,) + (2,) * n_dim)
        sub = type(self).from_cubes(new_cubes)

        sub.transfers = self.subdivide_transfers(sub, O)
        sub.transfer_matrices = [transfer_matrix(*t, shape=(sub.n_elements[n], self.n_elements[n]))
                                 for n, t in enumerate(sub.transfers)]
        sub.parent = self

        return sub

    def subdivide_transfers(coarse, fine, O):
        """build up transfer operators

        Returns
        -------
        transfers : list of tuple of arrays
            where each entry i is a tuple of ndarray, index_dtype, referencing (fine.elements[i], coarse.elements[i])
        """
        transfers = []
        for c, o in zip(fine.corners, O):
            q = np.sort(c, axis=1)
            # filter down e for those having connection with original vertices
            idx = np.flatnonzero(q[:, 0] < coarse.n_elements[0])
            q = q[idx]
            # highest vertex; translate to corresponding n-cube on the coarse level
            q = q[:, -1] - o
            transfers.append((idx, q))
        return transfers

    def subdivide_specialized(self, I):
        """Construct subdivided cubes from list of boundary indices

        Parameters
        ----------
        I : list of ndarray
            list of length n_dim+1
            the i-th entry of I contains the indices of the i-cubes incident to the n-cubes to be subdivided

        Returns
        -------
        new_cubes : ndarray, [n_cubes + cube_shape + cube_shape], index_type
            for each n-cube in self, we get an n-cube of subdivided n-cubes
        """

        n_cubes = self.n_elements[-1]
        n_dim = self.n_dim
        cube_shape = (2,) * n_dim
        new_cubes = -np.ones(((n_cubes,) + cube_shape + cube_shape), index_dtype)

        # zeros and ones describing an n-cube
        corners = np.indices(cube_shape).reshape(n_dim, -1).T

        # FIXME: come up with a method to loop over elements of I, for any n-dim
        # FIXME: perhaps hardcode the axis permutations for each ndim?
        # for d in range(n_dim+1):
        #     axes_list = list(zip(*dec.combinatorial.combinations(np.arange(n_dim), d)))[1]
        #     print(axes_list)
        # FIXME: subdivision fails with alternative generate_boundaries
        # quite unsure yet why

        if n_dim == 1:
            for c in corners:
                C = 1 - c   # corner and Complement
                corner = new_cubes[:, c[0]]
                corner[..., c[0]] = I[0][:, 0, c[0]]
                corner[..., C[0]] = I[1][:, 0]

        elif n_dim == 2:
            for c in corners:
                C = 1 - c   # corner and Complement
                corner = new_cubes[:, c[0], c[1]]
                # verts
                corner[..., c[0], c[1]] = I[0][:, 0, c[0], c[1]]
                # edges
                corner[..., c[0], C[1]] = I[1][:, 0, c[0]]  # edges along 0-axis
                corner[..., C[0], c[1]] = I[1][:, 1, c[1]]
                # faces
                corner[..., C[0], C[1]] = I[2][:, 0]

        elif n_dim == 3:
            for c in corners:
                C = 1 - c   # corner and Complement
                corner = new_cubes[:, c[0], c[1], c[2]]
                # verts
                corner[..., c[0], c[1], c[2]] = I[0][:, 0, c[0], c[1], c[2]]
                # edges
                corner[..., C[0], c[1], c[2]] = I[1][:, 2, c[1], c[2]]  # why the reverse indices?
                corner[..., c[0], C[1], c[2]] = I[1][:, 1, c[0], c[2]]
                corner[..., c[0], c[1], C[2]] = I[1][:, 0, c[0], c[1]]
                # faces
                corner[..., c[0], C[1], C[2]] = I[2][:, 0, c[0]]
                corner[..., C[0], c[1], C[2]] = I[2][:, 1, c[1]]
                corner[..., C[0], C[1], c[2]] = I[2][:, 2, c[2]]
                # cubes
                corner[..., C[0], C[1], C[2]] = I[3][:, 0]

        else:
            raise NotImplementedError

        return new_cubes

    def fundamental_domains(self):
        """Generate fundamental domains

        Returns
        -------
        domains : ndarray, [n_cubes] + cube_shape + cube_shape, index_dtype
            each cube generates a cube of fundamental domains,
            each consisting of a cube of cube indices

        """
        cube_shape = [2] * self.n_dim
        corners = np.indices(cube_shape).reshape(self.n_dim, -1).T

        shape = np.asarray([self.n_elements[-1]] + cube_shape + cube_shape)

        # 2d shape: [n, 2, 2,  2, 2]
        # 4 domains, 4 edges. each domain partakes in two edges
        # 3d shape: [n, 6, 4, 2,  2, 2]
        # 8 domains, 6 faces.
        #   each face is shared by 4 domains; each domain has 3 faces
        # 8 domains, 12x2 or 4x6 edges.
        #   each edge is shared by two domains; each domain has 3 edges
        # instead of n_combinations, need to have boundaries in a form that emphasises their orientation
        # corner vector indicating which dimension have been projected out?
        # faces is [1, 0, 0], [0, 1, 0], [0, 0, 1], and two sides of all
        # edges is adding another projection, and another mirror
        # this is a general pattern; add another axis to squeeze, and another mirror
        # if we can return this corner-structure from generate_boundaries, I think we are good

        domains = -np.ones(shape, dtype=index_dtype)
        # c = Ellipsis, 0, 0
        # domains[c] = self.range(-1)
        E = self.elements[-1]

        # FIXME: treat 0 and n as special cases? seems to work fine this way
        for d in range(0, self.n_dim + 1):
            IN0, corners = generate_cube_boundary(E, degree=d, return_corners=True)
            for i, c in enumerate(corners):
                c = (Ellipsis, ) + c
                domains[c] = IN0[:, i]

        # IN0, corners = generate_cube_boundary(
        #     E.reshape((-1,) + subcube), degree=2, return_corners=True)
        # c = (Ellipsis,) + corners[0]
        # domains[c] = IN0

        return domains

    def product(self, other):
        """Construct product topology of self and other

        Returns
        -------
        TopologyCubical
            topology of dimension self.n_dim + other.n_dim
        """

        if not np.array_equiv(self.elements[0], np.arange(self.n_elements[0], dtype=index_dtype)):
            raise ValueError('This breaks the assumptions made in this function')
        if not np.array_equiv(other.elements[0], np.arange(other.n_elements[0], dtype=index_dtype)):
            raise ValueError('This breaks the assumptions made in this function')

        self_cubes = self.elements[-1]
        other_cubes = other.elements[-1]

        sl, ss = self_cubes.shape[:1], self_cubes.shape[1:]
        ol, os = other_cubes.shape[:1], other_cubes.shape[1:]
        # some broadcasting magic to construct our new cubes
        cubes = self_cubes.reshape(sl + (1,) + ss + (1,) * other.n_dim) + \
                other_cubes.reshape((1,) + ol + (1,) * self.n_dim + os) * self.n_elements[0]

        return TopologyCubical.from_cubes(cubes.reshape((-1,) + cubes.shape[2:]))

    def fix_orientation(self):
        """Try to find a consistent orientation for all cubes

        Returns
        -------
        oriented : TopologyCubical
            same topology but with oriented simplices

        Raises
        ------
        ValueError
            if the topology is not orientable
        """
        E = self.elements[-1]
        orientation = self.relative_orientation()
        orientation = orientation.reshape((len(E),) + (1,) * self.n_dim)
        # FIXME: this type of edge flipping only works if from_cubes does internal sorting
        return type(self).from_cubes(np.where(orientation, E, E[:, ::-1]))

    def as_2(self):
        return TopologyCubical2(elements=self._elements, boundary=self._boundary, orientation=self._orientation)


class TopologyCubical2(TopologyCubical):

    def to_simplicial(self):
        """Convert the cubical topology into a simplicial topology,
        by forming 4 tris from each quad

        Returns
        -------
        TopologyTriangular
        """
        Q20 = self.elements[2]
        n_e = self.n_elements

        T20 = -np.ones((n_e[2], 4, 3), dtype=index_dtype)
        T20[:, 0, :2] = Q20[:, 0, ::+1]
        T20[:, 1, :2] = Q20[:, 1, ::-1]
        T20[:, 2, :2] = Q20[:, ::-1, 0]
        T20[:, 3, :2] = Q20[:, ::+1, 1]
        # all tris connect to a vertex inserted at each quad
        T20[:, :, 2] = self.range(2)[:, None] + n_e[0]

        from pycomplex.topology.simplicial import TopologyTriangular
        simplicial = TopologyTriangular.from_simplices(T20.reshape(-1, 3))
        self.to_simplicial_transfer(simplicial)
        return simplicial

    def to_simplicial_transfer(cubical, simplicial):
        """Construct transfer operators from cubical to simplicial

        The n-th operator in this list by right-multiplication with a cubical n-chain
        will result in the corresponding simpicial n-chain
        """
        import scipy.sparse
        # FIXME: implement edge operator
        simplicial.transfer_operators = [
            scipy.sparse.vstack([
                cubical.matrix(0, 0),
                cubical.matrix(2, 0).T / 4
            ]),
            None,
            transfer_matrix(
                simplicial.range(2),
                simplicial.range(2) // 4,
                shape=(simplicial.n_elements[2], cubical.n_elements[2])
            )
        ]
