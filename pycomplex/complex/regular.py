"""A regular oomplex is a cubical complex with the restriction that all cubes are axis-aligned

"""

import numpy as np
import numpy_indexed as npi
import scipy.spatial

from pycomplex.complex.cubical import *
from pycomplex.geometry import regular
from cached_property import cached_property


class ComplexRegularMixin(object):

    @cached_property
    def positive_dual_metric(self):
        """Returns true if all dual metrics are positive"""
        return True

    @cached_property
    def is_well_centered(self):
        """Test that all circumcenters are inside each simplex"""
        return True

    @cached_property
    def is_pairwise_delaunay(self):
        """Test that adjacent circumcenters do not cross eachother, or that dual 1-metric is positive"""
        return True

    @cached_property
    def weighted_average_operators(self):
        """No need for this fancy averaging nonsense on a regular grid"""
        return self.topology.dual.averaging_operators_0

    @cached_property
    def pick_primal_precomp(self):
        """Precompute things for efficient picking of primal regular n-cubes

        Returns
        -------
        tree : scipy.spatial.cKDTree
            tree built around centroid of each primal cube.
            querying the centroid will give us the index of the closest cube
        basis : ndarray, [n_domains, 2, n_dim], float
            first component is the first corner of the n-cube
            second component is the diagonal to the opposite corner
            this allows for efficient computation of barycentric coordinates within the cube

        """
        cubes = self.topology.elements[-1]
        # basis is just first vert of cube and diagonal of cube
        corners = self.vertices[cubes]
        corners = corners.reshape(-1, np.prod(self.topology.cube_shape), self.n_dim)
        basis = np.empty((len(corners), 2, self.n_dim))
        basis[:, 0] = corners[:, 0]
        basis[:, 1] = corners[:, -1] - corners[:, 0]

        tree = scipy.spatial.cKDTree(corners.mean(axis=1))
        return tree, basis

    def pick_primal(self, points, cube=None):
        """Pick the primal cubes

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick
        cube : ndarray, [n_points], index_dtype, optional
            primal cube indices
            can be given as an initial guess for the cube to be picked

        Returns
        -------
        cubes : ndarray, [n_points], index_dtype
            picked primal cube indices
        baries : ndarray, [n_points, n_dim] float
            barycentric weights corresponding to the domain indices

        """
        tree, basis = self.pick_primal_precomp

        def compute_baries(basis, points):
            r = points - basis[:, 0]
            return r / basis[:, 1]

        def query(points):
            _, cube = tree.query(points)
            b = basis[cube]
            baries = compute_baries(b, points)
            return cube, baries

        if cube is None:
            cube, baries = query(points)
        else:
            baries = compute_baries(basis[cube], points)
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                cube = cube.copy()
                d, b = query(points[update])
                cube[update] = d
                baries[update] = b

        return cube, baries

    @cached_property
    def pick_fundamental_precomp(self):
        """Precompute things for efficient picking of fundamental domain regular n-cubes

        Returns
        -------
        tree : scipy.spatial.cKDTree
            tree built around centroid of each fundamental cube.
            querying the centroid will give us the index of the closest cube
        domains : ndarray, [n_domains + cube_shape], index_dtype
            each n-cube contains references to n-cubes of various degrees
        basis : ndarray, [n_domains, 2, n_dim], float
            first component is the first corner of the n-cube
            second component is the diagonal to the opposite corner
            this allows for efficient computation of barycentric coordinates within the cube
        """

        domains = self.topology.cubical_domains()
        PP = self.primal_position

        cubes = np.ones(domains.shape + (self.n_dim,))
        for c in self.topology.cube_corners:
            idx1 = (Ellipsis, ) + tuple(c) + (slice(None), )
            idx2 = (Ellipsis, ) + tuple(c)
            cubes[idx1] = PP[c.sum()][domains[idx2]]

        cubes = cubes.reshape(-1, np.prod(self.topology.cube_shape), self.n_dim)
        basis = np.empty((len(cubes), 2, self.n_dim))
        basis[:, 0] = cubes[:, 0]
        basis[:, 1] = cubes[:, -1] - cubes[:, 0]
        import scipy.spatial
        tree = scipy.spatial.cKDTree(cubes.mean(axis=1))

        return tree, domains.reshape((-1,) + self.topology.cube_shape), basis

    def pick_fundamental(self, points, domain_idx=None):
        """Pick the fundamental domain cubes

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick
        domain_idx : ndarray, [n_points], index_dtype, optional
            fundamental domain indices
            can be given as an initial guess for the domain to be picked

        Returns
        -------
        domains : ndarray, [n_points + cube_shape], index_dtype
            each n-cube contains references to n-cubes of various degrees
        baries : ndarray, [n_points, n_dim] float
            barycentric weights corresponding to the domain indices

        """
        tree, domains, basis = self.pick_fundamental_precomp

        def compute_baries(basis, points):
            r = points - basis[:, 0]
            return r / basis[:, 1]

        def query(points):
            _, domain = tree.query(points)
            b = basis[domain]
            baries = compute_baries(b, points)
            return domains[domain], baries

        if domain_idx is None:
            domain, baries = query(points)
        else:
            baries = compute_baries(basis[domain_idx], points)
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                domain = domains[domain_idx]
                d, b = query(points[update])
                domain[update] = d
                baries[update] = b

        return domain, baries

    def pick_dual(self, points):
        """Pick the dual cubes

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick

        Returns
        -------
        cubes : ndarray, [n_points], index_dtype
            picked dual cube indices

        """
        domains, _ = self.pick_fundamental(points)
        return domains.reshape(len(domains), -1)[:, 0]

    def average_dual(self, d0):
        """Average a dual 0-form to obtain values on all dual elements

        Parameters
        ----------
        d0 : dual 0-form
            dual 0-form, including boundary values

        Returns
        -------
        list of dual n-forms
            n-th element is a dual n-form

        """
        dual_forms = [a * d0 for a in self.topology.dual.averaging_operators_0]
        return dual_forms

    def sample_dual_0(self, d0, points):
        """Sample a dual 0-form at the given points, using barycentric interpolation over the fundamental domains

        Parameters
        ----------
        d0 : ndarray, [n_dual_vertices, ...], float
            dual 0-form
        points : ndarray, [n_points, n_dim], float
            points to sample at

        Returns
        -------
        ndarray : [n_points, ...], float
            d0 sampled at the given points

        """
        # extend dual 0 form to all other dual elements by averaging
        dual_forms = self.average_dual(d0)[::-1]
        domain, bary = self.pick_fundamental(points)
        # no extrapolation
        bary = np.clip(bary, 0, 1)
        # do interpolation over fundamental domain
        total = 0
        for c in self.topology.cube_corners:
            idx = (Ellipsis, ) + tuple(c)
            B = bary * c + (1-bary) * (1-c)
            B = np.prod(B, axis=1)
            total = total + (dual_forms[c.sum()][domain[idx]].T * B.T).T
        return total

    def sample_primal_0(self, p0, points):
        """Sample a primal 0-form at the given points, using barycentric interpolation over the primal cubes

        Parameters
        ----------
        p0 : ndarray, [n_primal_vertices, ...], float
            primal 0-form
        points : ndarray, [n_points, n_dim], float
            points to sample at

        Returns
        -------
        ndarray : [n_points, ...], float
            p0 sampled at the given points

        """
        element, bary = self.pick_primal(points)
        bary = np.clip(bary, 0, 1)
        IN0 = self.topology.incidence[-1, 0]
        verts = IN0[element]
        # do interpolation over cube
        total = 0
        for c in self.topology.cube_corners:
            idx = (Ellipsis, ) + tuple(c)
            B = bary * c + (1-bary) * (1-c)
            B = np.prod(B, axis=1)
            total = total + (p0[verts[idx]].T * B.T).T
        return total

    def fundamental_domain_postions(self):
        domains = self.topology.cubical_domains()
        PP = self.primal_position

        cubes = np.ones(domains.shape + (self.n_dim,))
        for c in self.topology.cube_corners:
            idx1 = (Ellipsis,) + tuple(c) + (slice(None),)
            idx2 = (Ellipsis,) + tuple(c)
            cubes[idx1] = PP[c.sum()][domains[idx2]]
        return cubes

    @cached_property
    def metric(self):
        """Compute metrics from fundamental domain contributions

        A bit ugly and not super efficient, but it does work in n-d
        """
        topology = self.topology
        assert topology.is_oriented

        domains = self.topology.cubical_domains().reshape((-1,) + self.topology.cube_shape)
        corners = self.topology.cube_corners

        # assemble subdomain cube vertices
        cubes = self.fundamental_domain_postions().reshape((-1, ) + self.topology.cube_shape + (self.n_dim, ))

        PN = topology.n_elements
        DN = PN[::-1]

        # metrics
        PM = [np.zeros(n) for n in PN]
        PM[0][...] = 1
        DM = [np.zeros(n) for n in DN]
        DM[0][...] = 1

        def idx(c):
            return (slice(None),) + tuple(c)

        groups = [npi.group_by(domains[idx(c)]) for c in corners]   # cache groupings since they may get reused

        # do summation of primal terms; feels like there should be a simpler way
        for i, c in list(enumerate(corners))[1:-1]:
            n = c.sum()
            hypervolume = regular.hypervolume(cubes[idx(corners[0])], cubes[idx(c)], n=n)
            g, sums = groups[i].sum(hypervolume)
            PM[n][g] += sums
        # now we divide by the number of primal N-cubes contributing to each primal n-cube
        for n in range(1, self.n_dim):
            PM[n] /= self.topology.degree[n]

        # do summation of dual terms
        for i, c in list(enumerate(corners))[1:-1]:
            n = self.n_dim - c.sum()
            hypervolume = regular.hypervolume(cubes[idx(c)], cubes[idx(corners[-1])], n=n)
            g, sums = groups[i].sum(hypervolume)
            DM[n][g] += sums / (2 ** c.sum())

        # do summation over N-cubes; these are simple
        V = regular.hypervolume(cubes[idx(corners[0])], cubes[idx(corners[-1])])
        PM[-1] = groups[-1].sum(V)[1]
        DM[-1] = groups[+0].sum(V)[1]

        return PM, DM


class ComplexRegular1(ComplexRegularMixin, ComplexCubical1Euclidian1):
    """Regular cubical 1 complex; or line segment"""

    def plot_primal_0_form(self, f0, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        edges = self.topology.elements[1]
        x = self.vertices[:, 0][edges]
        y = f0[edges]

        lines = np.concatenate([x[..., None], y[..., None]], axis=2)

        lc = matplotlib.collections.LineCollection(lines, **kwargs)
        ax.add_collection(lc)

        ax.set_xlim(self.box[:, 0])
        ax.set_ylim(f0.min(), f0.max())


class ComplexRegular2(ComplexRegularMixin, ComplexCubical2Euclidian2):
    """Regular cubical 2-complex"""
    def plot_primal_0_form(self, c0, **kwargs):
        """Plot a primal 0-form

        Parameters
        ----------
        c0 : ndarray, [n_vertices], float
            a primal 0-form

        Notes
        -----
        This functionality is built on top of the primal 0-form plotting of a triangular complex
        If this function is to be called repeatedly it is much preferrable to cache this subdivision
        """
        tris = self.subdivide_simplicial()
        tris.as_2().plot_primal_0_form(tris.topology.transfer_operators[0] * c0, **kwargs)


class ComplexRegular3(ComplexRegularMixin, ComplexCubical3Euclidian3):
    """Regular cubical 3-complex"""


class ComplexRegular4(ComplexRegularMixin, ComplexCubical4Euclidian4):
    """Regular cubical 4-complex"""
