import numpy as np
import numpy_indexed as npi
import scipy
import scipy.sparse
from cached_property import cached_property

import pycomplex.math.combinatorial
from pycomplex.topology.topology import BaseTopology
from pycomplex.topology import index_dtype, sign_dtype, transfer_matrix


def generate_boundary(cubes, degree=1):
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
    """
    cubes = np.asarray(cubes)
    n_dim = cubes.ndim - 1
    b_dim = n_dim - degree
    n_elements = cubes.shape[0]

    # axes to pull forward to generate all incident n-d-cubes
    axes_list = list(zip(*pycomplex.math.combinatorial.combinations(np.arange(n_dim), degree)))[1]
    n_combinations = len(axes_list)

    boundary = np.empty((n_elements, n_combinations) + (2,) * degree + (2,) * (b_dim), dtype=cubes.dtype)

    for i, axes in enumerate(axes_list):
        s_view = cubes
        for j, axis in enumerate(axes):
            s_view = np.moveaxis(s_view, axis + 1, j + 1)
        boundary[:, i] = s_view
    return boundary


class TopologyCubical(BaseTopology):
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

            En0_all = generate_boundary(EN0, degree=1)
            En0_all = En0_all.reshape((-1,) + (2,) * (b_dim))

            # get mapping to unique set
            # FIXME: we need a sort to find uniqueness. see examples/subdivision_surface
            # not required if we stick to geometrically regular grids; add a flag?
            index = npi.as_index(En0_all)
            En0 = index.unique
            EnN = index.inverse
            EnN = EnN.reshape((n_elements,) + (n_dim, 2))

            # set up connection parity
            parity = np.zeros_like(EnN)
            # alternate orientation on both ends
            parity[..., 1] = 1
            # alternate orientation over the axes
            # FIXME: cant defend this choice and works only up to ndim=3
            parity = np.logical_xor(parity, (np.arange(n_dim)[None, :, None] % 2))

            # create topology matrix
            orientation = parity * 2 - 1
            return En0, EnN, orientation.astype(sign_dtype)

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
            EN0, ENn, ONn = lower(EN0)

            n_dim = EN0.ndim - 1
            N = n_dim

            E[N] = EN0
            O[N] = ONn
            B[N] = ENn

            if N == 0:
                break

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
        B = [generate_boundary(cubes, d) for d in reversed(range(self.n_dim + 1))]
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

                corner[..., c[0], c[1]] = I[0][:, 0, c[0], c[1]]
                corner[..., c[0], C[1]] = I[1][:, 0, c[0]]  # edges along 0-axis
                corner[..., C[0], c[1]] = I[1][:, 1, c[1]]
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
        return TopologyTriangular.from_simplices(T20.reshape(-1, 3))

    def to_simplicial_transfer(cubical, simplicial):
        pass
        # need to be able to interpolate 0-forms for many purposes... stack E00 and E20
