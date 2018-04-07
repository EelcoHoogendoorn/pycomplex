
import numpy as np
import numpy.testing as npt
import numpy_indexed as npi
import scipy.sparse

from cached_property import cached_property

from pycomplex.topology import index_dtype, sign_dtype, topology_matrix, selection_matrix
from pycomplex.topology.base import BaseTopology


def parity_to_orientation(parity):
    return ((np.asarray(parity) * 2) - 1).astype(sign_dtype)


def orientation_to_parity(parity):
    return np.asarray(parity) < 0


def indices(shape, dtype):
    """mem-efficient version of np.indices"""
    n_dim = len(shape)
    idx = [np.arange(s, dtype=dtype) for s in shape]
    for q, (i, s) in enumerate(zip(idx, shape)):
        strides = [0] * n_dim
        strides[q] = i.strides[0]
        idx[q] = np.ndarray(buffer=i.data, shape=shape, strides=strides, dtype=i.dtype)
    return idx


def sort_and_argsort(arr, axis):
    """Potentially faster method to sort and argsort; hasnt been profiled though

    Parameters
    ----------
    arr : ndarray, [shape]
    axis : int
        axis to sort along

    Returns
    -------
    sorted : ndarray, [shape]
    argsort : ndarray, [shape], int
        indices along axis of arr
    """
    argsort = np.argsort(arr, axis=axis)
    I = indices(arr.shape, index_dtype)
    I[axis] = argsort
    return arr[I], argsort


def relative_permutations(self, other):
    """Combine two permutations of indices to get relative permutation

    Parameters
    ----------
    self : ndarray, [n, m], int
    other : ndarray, [n, m], int

    Returns
    -------
    relative : ndarray, [n, m], int
    """
    assert self.shape == other.shape
    assert self.ndim == 2
    I = np.indices(self.shape)
    relative = np.empty_like(self)
    relative[I[0], other] = self
    return relative


class PrimalTopology(BaseTopology):
    """Stuff common to all primal topology objects in all dimensions

    This deals with their construction from element arrays
    """

    @classmethod
    def from_elements(cls, elements):
        raise NotImplementedError

    def __init__(self, elements, boundary, orientation, strict=False):
        """

        Parameters
        ----------
        elements : list of ndarrays
            element representation in terms of vertex indices; details depend on subtype
            n-th entry describes n-elements in terms of the indices of the 0-elements
            elements of the list are of type index_dtype
        boundary : list of ndarrays
            list of boundary index information
            n-th entry describes n+1-elements in terms of the indices of its n-element boundary
            elements of the list are of type index_dtype
        orientation : list of ndarrays
            list of orientation sign information
            n-th entry describes n+1-elements in terms of the relative orientation of its n-element boundary
            elements of the list are of type sign_dtype
        strict : bool, optional
            if True, strict checking for manifoldness is performed
        """
        if not np.array_equiv(elements[0].flatten(), np.arange(len(elements[0]))):
            raise ValueError('Vertices need to form a contiguous range starting at 0')

        self.strict = strict
        self._elements = elements
        self._orientation = orientation
        self._boundary = boundary
        self.n_dim = len(orientation)

        assert(len(boundary) == len(orientation))
        assert(len(elements) == self.n_dim + 1)
        for b, o in zip(boundary, orientation):
            if not b.shape == o.shape:
                raise ValueError
            if not b.dtype == index_dtype:
                raise ValueError
            if not o.dtype == sign_dtype:
                raise ValueError
        for e in elements:
            if not e.dtype == index_dtype:
                raise ValueError

    @cached_property
    def matrices(self):
        """
        Returns
        -------
        array_like, [n_dim], sparse matrix
        """
        # FIXME: this should go to baseclass
        return [self.matrix(i) for i in range(self.n_dim)]

    @cached_property
    def matrices_original(self):
        """For symmetry with dual, where there is a distinction"""
        return self.matrices

    @cached_property
    def matrices_blocked(self):
        """Topology matrices, with i/p split

        Does this even make sense? would need to reorder all unknowns to group boundary;
        possible but not very efficient. leave block construction to derived equation objects?
        """
        raise NotImplementedError
        return

    @cached_property
    def elements(self):
        return self._elements

    @cached_property
    def corners(self):
        return [e.reshape(n, -1) for e, n in zip(self._elements, self.n_elements)]

    @cached_property
    def n_elements(self):
        """List of ints; n-th entry is number of n-elements"""
        return [len(e) for e in self._elements]

    @cached_property
    def incidence(self):
        """Incidence relations between topological elements"""
        I = np.zeros((self.n_dim + 1, self.n_dim + 1), dtype=np.object)
        for i, e in enumerate(self.elements):
            I[i, 0] = e
        for i, b in enumerate(self._boundary):
            I[i+1, i] = b
        return I

    # FIXME: add caching here?
    def matrix(self, n, m=None):
        """Construct topological relations between elements as sparse matrix"""
        if n < 0:
            n = self.n_dim + n
        if m is None:
            m = n + 1
        elif m == 0:
            i = self._elements[n]
            o = np.ones_like(i)

        if m == n + 1:
            i = self._boundary[n]
            o = self._orientation[n]
        if m == n:
            i = np.arange(self.n_elements[n], dtype=index_dtype).reshape(-1, 1)
            o = np.ones_like(i)

        return topology_matrix(i, o)

    @cached_property
    def dual(self):
        """Return dual topology object, that closes all boundaries"""
        from pycomplex.topology.dual import Dual, ClosedDual
        if not self.is_oriented:
            raise ValueError('Dual of non-oriented manifold is not supported')
        if self.is_closed:
            return ClosedDual(self)
        else:
            if not self.boundary.is_oriented:
                raise ValueError('Dual of non-oriented boundary is not supported')
            return Dual(self)

    def find_correspondence(self, other, mapping):
        """

        Parameters
        ----------
        self : BaseTopology
        other : BaseTopology
            subset of self
        mapping : ndarray, [other.n_elements[0]], index_type
            indices of self.vertices for each of other.vertices

        Returns
        -------
        idx : list of ndarray
            ith entry of the list contains the index of other in self of the i-elements
        """
        idx = []
        for C, c in zip(self.corners, other.corners):
            c = np.sort(mapping[c], axis=1)
            C = np.sort(C, axis=1)
            # FIXME: find and do something with relative orientation here?
            idx.append(npi.indices(C, c))
        return idx

    def select_subset(self, n_chain):
        """Select a subset of the topology

        Parameters
        ----------
        n_chain : ndarray, [n_elements[-1], sign_type
            boolean chain indicating which elements to select

        Returns
        -------
        type(self)
            Topology of the indicated subset,
            including a precomputed mapping back to the original topology
        """
        n_chain = np.asarray(n_chain)
        if not len(n_chain) == self.n_elements[-1]:
            raise ValueError

        elements = self.elements[-1][np.flatnonzero(n_chain)]   # FIXME: discarding sign info of chain here...

        mapping, inverse = np.unique(elements.flatten(), return_inverse=True)
        elements = inverse.reshape(elements.shape).astype(index_dtype)

        subset = type(self).from_elements(elements)
        subset.parent = self
        subset.parent_idx = self.find_correspondence(subset, mapping)
        return subset

    @cached_property
    def boundary(self):
        """Return n-1-topology representing the boundary

        Returns
        -------
        self.boundary_type
            Boundary topology, with attribute parent_idx referring back to the parent elements
        """
        # FIXME: rather than using from_elements, a direct subset selection would be preferable
        # shouldnt be too hard; just find chains on all elements, select subsets, and remap vertex indices
        # FIXME: not sure boundary orientation is even preserved in current implementation
        if not self.is_oriented:
            raise ValueError('Cannot get the boundary of a non-oriented manifold')

        chain_N = self.chain(-1, fill=1)
        chain_n = self.matrix(-1) * chain_N
        b_idx = np.flatnonzero(chain_n)
        if len(b_idx) == 0:
            # topology is closed
            return None

        # construct boundary
        boundary_elements = self.elements[-2][b_idx]

        # mapping is a list of parent vertex indices, for each boundary vertex
        mapping, inverse = np.unique(boundary_elements.flatten(), return_inverse=True)
        boundary_elements = inverse.reshape(boundary_elements.shape).astype(index_dtype)

        boundary = self.boundary_type().from_elements(boundary_elements)
        boundary.parent_idx = self.find_correspondence(boundary, mapping)
        boundary.parent = self
        return boundary

    @cached_property
    def is_closed(self):
        return self.boundary is None

    def split_connected_components(self):
        """Split the topology in disjoint connected components

        Returns
        -------
        components : list of topology objects
            all disjoint components, sorted from most to least n-elements

        """
        n_components, labels = self.label_connections()
        components = [self.select_subset(labels == i) for i in range(n_components)]
        return sorted(components, key=lambda c: -c.n_elements[-1])

    @cached_property
    def selector(self):
        """Operators to select interior, or to strip boundary elements

        Returns
        -------
        selectors : list of len self.n_dim + 1
            selectors mapping all elements to the interior subset
            first element of this list is square; maps dual n-forms to primal 0-forms, which are one-to-one
        """
        def s(i):
            d = self.chain(n=i, fill=1, dtype=sign_dtype)
            try:
                d[self.boundary.parent_idx[i]] = 0
            except:
                pass
            return selection_matrix(d)

        return [s(i) for i in range(self.n_dim + 1)]

    @cached_property
    def selector_b(self):
        """Operators to select boundary, or to strip interior elements

        Returns
        -------
        selectors : list of len self.n_dim + 1
            selectors mapping all elements to the boundary subset
            first element of this list is trivial; maps dual n-forms to primal 0-forms, which have no boundary terms
        """
        def s(i):
            d = self.chain(n=i, fill=0, dtype=sign_dtype)
            try:
                d[self.boundary.parent_idx[i]] = 1
            except:
                pass
            return selection_matrix(d)

        return [s(i) for i in range(self.n_dim + 1)]
