import numpy as np
import numpy_indexed as npi
from fastcache import clru_cache
from cached_property import cached_property

import pycomplex.math.combinatorial
from pycomplex.topology import sign_dtype, index_dtype
from pycomplex.topology.primal import PrimalTopology
from pycomplex.topology.util import parity_to_orientation, sort_and_argsort, relative_permutations, transfer_matrix, \
    element_indices


def generate_cube_boundary(cubes, degree=1, return_corners=False, mirror=True):
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
        if p and mirror:
            s_view = np.flip(s_view, axis=-1)
        boundary[:, i] = s_view

    if degree == 1 and mirror:
        # flip the parity of elements on one side of the cube
        boundary[:, :, 1] = np.flip(boundary[:, :, 1], axis=-1)

    if return_corners:
        return boundary, np.asarray(corners)
    else:
        return boundary


@clru_cache()
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

    @cached_property
    def cube_shape(self):
        """Shape of a unit cube"""
        return (2, ) * self.n_dim

    @cached_property
    def cube_corners(self):
        """Ndarray, [n_corners, n_dim], int"""
        return np.indices(self.cube_shape).reshape(self.n_dim, -1).T.astype(sign_dtype)

    @classmethod
    def from_elements(cls, elements, mirror=True):
        return cls.from_cubes(cubes=elements, mirror=mirror)

    @classmethod
    def from_cubes(cls, cubes, mirror=True):
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
            En0_all = generate_cube_boundary(EN0, degree=1, mirror=mirror)
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

        EN0 = np.asarray(cubes, dtype=index_dtype)
        n_dim = EN0.ndim - 1
        if not EN0.shape[1:] == (2,) * n_dim:
            raise ValueError

        E = np.zeros((n_dim + 1), dtype=object)
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

    def subdivide_cubical(self):
        """Cubical topology subdivision; general n-d case

        Returns
        -------
        TopologyCubical
            subdivided topology where each n-cube has been split into an n_cube of n_cubes

        Notes
        -----
        currently, the transfer operators are set as a side effect of calling this function; not super happy with that
        transfer operators still incomplete

        note that the divided topology is created using from_cubes
        this complicates relating vector-forms between fine and coarse,
        since their relationship is implicit; all n-cubes are encoded by their vert-idx,
        and related back into fine intermediate cubes in from_cubes

        Perhaps implementing a subdivide_cubical_direct could mitigate those concerns
        That isnt easy either though
        """
        offset = np.cumsum([0] + self.n_elements, dtype=index_dtype)
        new_cubes = self.cubical_domains().copy()

        # subdivision is essentially just remapping fundamental-domain n-cube indices to 0-cube indices
        for c in self.cube_corners:
            idx = (Ellipsis, ) + tuple(c)
            new_cubes[idx] += offset[c.sum()]
        # perform mirrors according to corners, to preserve orientation
        for c in self.cube_corners:
            idx = (slice(None), ) + tuple(c)
            for i, b in enumerate(c):
                if b == 1:
                    new_cubes[idx] = np.flip(new_cubes[idx], axis=1 + i)

        new_cubes = new_cubes.reshape((-1,) + self.cube_shape)

        sub = type(self).from_cubes(new_cubes)
        sub.parent = self

        sub.transfers = self.subdivide_cubical_transfers(sub)
        sub.transfer_matrices = [transfer_matrix(*t, shape=(sub.n_elements[n], self.n_elements[n]))
                                 for n, t in enumerate(sub.transfers)]

        return sub

    def subdivide_cubical_relations(coarse, fine):
        """Find the relationship between subdivided cubes and their parents

        This is mostly inverting computation in subdivide_cubical...
        This function only assumes fine vertices were generated in accordance with `offsets`,
        with vertices inserted on course 0-cubes followed by vertices on coarse 1-cubes, and so on

        Returns
        -------
        List[ndarray, [], uint8]
            list corresponding to each cube elements array
            where the array encodes the order of the parent element for each fine element
        List[ndarray, [], index_dtype]
            list corresponding to each cube elements array
            where the array encodes the index of the parent element for each fine element
        """
        offsets = np.cumsum([0] + coarse.n_elements, dtype=index_dtype)
        order = []
        parent = []
        for e in fine.elements:
            o = (e[..., None] >= offsets[1:]).sum(axis=-1, dtype=np.uint8)
            order.append(o)
            p = e - offsets[o]
            parent.append(p)
        return order, parent

    def subdivide_cubical_transfers(coarse, fine):
        """Build up transfer operators
        These operators only encode relationships between purely coincident n-cubes;
        only between those that share a direct ancestry relationship

        Parameters
        ----------
        coarse : Complex
        fine : Complex

        Returns
        -------
        transfers : list of tuple of arrays
            where each entry i is a tuple of ndarray, index_dtype, referencing (fine.elements[i], coarse.elements[i])
        """
        # FIXME: how to derive relative orientation? or do we not need to? is it inherited by construction? test suggests so

        order, parent = coarse.subdivide_cubical_relations(fine)
        transfers = []
        for n, (o, p) in enumerate(zip(order, parent)):
            i = np.nonzero(o == 0)  # elements connected to coarse vertices
            f = i[0]
            j = np.nonzero(o[f] == n)  # connections to coarse n-elements
            c = p[f][j]
            transfers.append((f, c))
        return transfers

    def subdivide_octohedral(self):
        """Subdivision by inserting a new 0-cube at each n-cube, and creating a new 'octahedron'
        at each n-1-cube. This requires the addition of a wholly new topology type however, where each element
        has 2 * n_dim vertices.
        At the boundary, we would only get a half-element, so only elegant on closed topologies really

        Also, edges do not map to edges, so thats bad for crease modelling.

        Cute, but not much point to it
        """
        assert self.is_closed
        raise NotImplementedError

    def subdivide_fundamental(self):
        """Subdivide cubical topology into simplicial topology,
        by means of splitting each n-cube into its fundamental domains

        Returns
        -------
        sub: TopologySimplicial
        """
        offset = np.cumsum([0] + self.n_elements, dtype=index_dtype)[:-1]
        # subdivision is essentially just remapping fundamental-domain n-simplex indices to 0-simplex indices
        simplices = self.fundamental_domains() + offset
        # flip the mirrored side to preserve orientation;
        simplices[..., 0, [0, -1]] = simplices[..., 0, [-1, 0]]
        from pycomplex.topology.simplicial import TopologySimplicial
        sub = TopologySimplicial.from_simplices(simplices.reshape(-1, self.n_dim + 1))
        sub.parent = self
        return sub

    def fundamental_domains(self):
        """Form fundamental domain simplices for each n-cube by connecting corners of all degrees

        Notes
        -----
        This is very similar to simplex fundamental domain subdivision logic

        Returns
        -------
        ndarray, [n_cubes, 2*n ... 2, n_dim + 1], index_dtype
            all fundamental domain simplices
        """
        n_simplex_corners = self.n_dim + 1
        dim = ((np.arange(1, n_simplex_corners)) * 2)[::-1]

        shape = np.asarray([self.n_elements[-1]] + list(dim) + [n_simplex_corners])

        domains = -np.ones(shape, dtype=index_dtype)
        domains[..., -1] = self.range(-1).reshape((-1,) + (1,)*self.n_dim) # they all refer to their parent primal simplex

        IN0 = self.elements[-1]
        cube_shape = self.cube_shape
        for i in range(2, self.n_dim + 1):
            s = shape[:-1].copy()
            s[i:] = 1
            IN0 = generate_cube_boundary(IN0.reshape((-1,)+cube_shape), mirror=True)
            cube_shape = cube_shape[1:]
            IN0 = IN0.reshape(tuple(s[:i]) + cube_shape)
            domains[..., -i] = element_indices(self.elements[-i], IN0).reshape(s)
        domains[..., 0] = IN0

        return domains

    def cubical_domains(self):
        """Generate cubical fundamental domains

        Returns
        -------
        domains : ndarray, [n_cubes + cube_shape + cube_shape], index_dtype
            each cube generates a cube of fundamental domain cubes,
            each consisting of a cube of cube indices

        Notes
        -----
        These are cubical fundamental domains, not true fundamental domains
        that is, these subdomains do not reflect the diagonal symmetry of the cube
        """
        shape = np.asarray((self.n_elements[-1],) + self.cube_shape + self.cube_shape)

        domains = -np.ones(shape, dtype=index_dtype)
        c = (Ellipsis,) + (1,) * self.n_dim
        domains[c] = self.range(-1).reshape((-1,) + (1,) * self.n_dim)
        E = self.elements[-1]
        c = (Ellipsis,) + (0,) * self.n_dim
        domains[c] = E

        for d in range(1, self.n_dim):
            IN0, corners = generate_cube_boundary(E, degree=d, return_corners=True)

            for i, c in enumerate(corners):
                idx = element_indices(self.elements[-(d + 1)], IN0[:, i])
                shape = (-1, ) + tuple(1 + c)   # select broadcasting pattern
                c = (Ellipsis, ) + tuple(1 - c) # indexing with this leave shape [n] + cube_shape; picks the type of n-cube to write to
                domains[c] = idx.reshape(shape)

        return domains

    def product(self, other):
        """Construct product topology of self and other

        Returns
        -------
        TopologyCubical
            topology of dimension self.n_dim + other.n_dim
        """

        # if not np.array_equiv(self.elements[0], np.arange(self.n_elements[0], dtype=index_dtype)):
        #     raise ValueError('This breaks the assumptions made in this function')
        # if not np.array_equiv(other.elements[0], np.arange(other.n_elements[0], dtype=index_dtype)):
        #     raise ValueError('This breaks the assumptions made in this function')

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
            same topology but with oriented cubes

        Raises
        ------
        ValueError
            if the topology is not orientable
        """
        E = self.elements[-1]
        parity = self.relative_parity()
        # broadcast parity to all cube corners
        parity = parity.reshape((len(parity),) + (1,) * self.n_dim)
        # FIXME: this type of edge flipping only works if from_cubes does internal sorting
        return type(self).from_cubes(np.where(parity, E, E[:, ::-1]))

    def as_2(self):
        return TopologyCubical2(elements=self._elements, boundary=self._boundary, orientation=self._orientation)


class TopologyCubical2(TopologyCubical):

    def subdivide_simplicial(self):
        """Convert the cubical topology into a simplicial topology,
        by forming 4 tris from each quad

        Returns
        -------
        TopologyTriangular

        Notes
        -----
        There does not appear to be a way to generalize this to nd

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
        self.subdivide_simplicial_transfer(simplicial)
        return simplicial

    def subdivide_simplicial_transfer(cubical, simplicial):
        """Construct transfer operators from cubical to simplicial

        The n-th operator in this list by right-multiplication with a cubical n-chain
        will result in the corresponding simpicial n-chain
        """
        import scipy.sparse
        # FIXME: implement edge operator
        simplicial.transfer_operators = [
            scipy.sparse.vstack(cubical.averaging_operators_0[::2]),
            None,
            transfer_matrix(
                simplicial.range(2),
                simplicial.range(2) // 4,
                shape=(simplicial.n_elements[2], cubical.n_elements[2])
            )
        ]
        simplicial.parent = cubical
