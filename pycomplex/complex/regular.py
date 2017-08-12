
from cached_property import cached_property

from pycomplex.geometry import regular
from pycomplex.complex.cubical import *


def gather(idx, vals):
    """return vals[idx]. return.shape = idx.shape + vals.shape[1:]"""
    return vals[idx]


def scatter(idx, vals, target):
    """target[idx] += vals. """
    np.add.at(target.ravel(), idx.ravel(), vals.ravel())


def pinv(A):
    u, s, v = np.linalg.svd(A)
    s = 1 / s
    # s[:, self.complex.topology.n_dim:] = 0
    return np.einsum('...ij,...j,...jk->...ki', u[..., :s.shape[-1]], s, v)


class ComplexRegularMixin(object):

    @cached_property
    def pick_primal_precomp(self):
        cubes = self.topology.elements[-1]

        # basis is just first vert of cube and diagonal of cube
        corners = self.vertices[cubes]
        corners = corners.reshape(-1, np.prod(self.topology.cube_shape), self.n_dim)
        basis = np.empty((len(corners), 2, self.n_dim))
        basis[:, 0] = corners[:, 0]
        basis[:, 1] = corners[:, -1] - corners[:, 0]

        import scipy.spatial
        tree = scipy.spatial.cKDTree(corners.mean(axis=1))
        return tree, cubes, basis

    def pick_primal(self, points, cube=None):
        """Pick the primal cubes

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick

        Returns
        -------
        cubes : ndarray, [n_points, ], index_dtype
            n-th column corresponds to indices of n-element
        baries : ndarray, [n_points, n_dim] float
            barycentric weights corresponding to the domain indices

        """
        tree, cubes, basis = self.pick_primal_precomp

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
        domains = self.topology.fundamental_domains()
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

    def pick_fundamental(self, points, domain=None):
        """Pick the fundamental domain cubes

        Parameters
        ----------
        points : ndarray, [n_points, n_dim], float
            points to pick

        Returns
        -------
        domains : ndarray, [n_points + cube_shape], index_dtype
            n-th column corresponds to indices of n-element
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

        if domain is None:
            domain, baries = query(points)
        else:
            baries = compute_baries(basis[domain], points)
            update = np.any(baries < 0, axis=1)
            if np.any(update):
                domain = domain.copy()
                d, b = query(points[update])
                domain[update] = d
                baries[update] = b

        return domain, baries

    @cached_property
    def cached_averages(self):
        # for a truly regular grid, there is no need for weighting, unlike simplicial case
        return self.topology.dual.averaging_operators

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
        interior_d0, boundary_d0 = np.split(d0, [self.topology.n_elements[-1]], axis=0)
        dual_forms = [a * interior_d0 for a in self.cached_averages]    # these are duals without boundary
        if len(boundary_d0):
            boundary_forms = [a * boundary_d0 for a in self.boundary.topology.dual.averaging_operators()]
            for i, (d, b, p) in enumerate(zip(dual_forms, boundary_forms, self.boundary.topology.parent_idx)):
                raise Exception('check this logic')
                d[p] = b
        return dual_forms

    def sample_dual_0(self, d0, points):
        # extend dual 0 form to all other dual elements by averaging
        dual_forms = self.average_dual(d0)[::-1]
        domain, bary = self.pick_fundamental(points)
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


#     @cached_property
#     def metric(self):
#         """Compute metrics from fundamental domain contributions"""
#         topology = self.topology
#         assert topology.is_oriented
#         PP = self.primal_position
#         domains = self.topology.fundamental_domains()
#
#         domains = domains.reshape(-1, domains.shape[-1])
#         corners = np.concatenate([p[d][:, None, :] for p, d in zip(PP, domains.T)], axis=1)
#
#         # assemble cube vertices
#         cubes =
#
#         PN = topology.n_elements
#         DN = PN[::-1]
#
#         # metrics
#         PM = [np.zeros(n) for n in PN]
#         PM[0][...] = 1
#         DM = [np.zeros(n) for n in DN]
#         DM[0][...] = 1
#
#         unsigned = regular.unsigned_volume
#         groups = [npi.group_by(c) for c in domains.T]   # cache groupings since they may get reused
#
#         for i in range(1, self.topology.n_dim):
#             n = i + 1
#             d = self.topology.n_dim - i
#             PM[i] = groups[i].mean(unsigned(corners[:, :n]))[1] * factorial(n)
#             DM[i] = groups[d].sum (unsigned(corners[:, d:]))[1] / factorial(d+1)
#
#         V = regular.hypervolume(cubes[:, corners[0]], cubes[:, corners[-1]])
#         PM[-1] = groups[-1].sum(V)[1]
#         DM[-1] = groups[+0].sum(V)[1]
#
#         return (
#             [m * (self.radius ** i) for i, m in enumerate(PM)],
#             [m * (self.radius ** i) for i, m in enumerate(DM)]
#         )


class ComplexRegular1(ComplexCubical1Euclidian1):
    """Regular cubical 1 complex; or line segment"""

    @cached_property
    def metric(self):
        """Calc metric properties and hodges for a 1d regular cubical complex

        Returns
        -------
        primal : list of ndarray
        dual : list of ndarray

        """

        topology = self.topology
        PP = self.primal_position

        PN = topology.n_elements
        DN = PN[::-1]

        #metrics
        PM = [np.zeros(n) for n in PN]
        PM[0][...] = 1
        DM = [np.zeros(n) for n in DN]
        DM[0][...] = 1

        # precomputations
        E10  = topology.incidence[1, 0].reshape(-1, 2)  # [edges, v2]          vertex indices per edge
        PP10  = PP[0][E10]                 # [edges, v2]

        # calc edge lengths
        PM[1] += regular.edge_length(PP10[:, 0, :], PP10[:, 1, :])
        for d1 in range(2):
            scatter(
                E10[:,d1],
                regular.edge_length(PP10[:, d1, :], PP[1]),
                DM[1])

        return PM, DM

    def plot_primal_0_form(self, f0, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.collections

        edges = self.topology.elements[1]
        x = self.vertices[:, 0][edges]
        y = f0[edges]

        lines = np.concatenate([x[..., None], y[..., None]], axis=2)

        lc = matplotlib.collections.LineCollection(lines, **kwargs)
        ax.add_collection(lc)

        ax.set_xlim(self.box[:, 0])
        ax.set_ylim(f0.min(), f0.max())


class ComplexRegular2(ComplexRegularMixin, ComplexCubical2Euclidian2):
    """Regular cubical 2-complex
    makes for easy metric calculations, and guarantees orthogonal primal and dual
    """
    # FIXME: drop dependence on euclidian2; would like to be able to process boundaries of regulars as well
    # FIXME: if we are in 22-space, can we use more efficient plotting?

    @cached_property
    def metric(self):
        """Calc metric properties and hodges for a 2d regular cubical complex

        Returns
        -------
        primal : list of ndarray
        dual : list of ndarray

        Notes
        -----
        Should be relatively easy to generalize to n-dimensions

        """
        topology = self.topology
        PP = self.primal_position

        PN = topology.n_elements
        DN = PN[::-1]

        # metrics
        PM = [np.zeros(n) for n in PN]
        PM[0][...] = 1
        DM = [np.zeros(n) for n in DN]
        DM[0][...] = 1

        # precomputations
        E20  = topology.incidence[2, 0]  # [faces, e2, e2]      vertex indices per face
        E21  = topology.incidence[2, 1]  # [faces, e2, e2]      edge indices per face
        E10  = topology.incidence[1, 0].reshape(-1, 2)  # [edges, v2]          vertex indices per edge

        PP10  = PP[0][E10]                 # [edges, v2, c3]
        PP21  = PP[1][E21]                 # [faces, e2, e2, c3] ; face-edge midpoints
        PP20  = PP[0][E20]                 # [faces, e2, e2, c3]

        # calc areas of fundamental squares
        for d1 in range(2):
            for d2 in range(2):
                # this is the area of one fundamental domain, assuming regular coords
                areas = regular.hypervolume(PP20[:, d1, d2, :], PP[2])
                PM[2] += areas                    # add contribution to primal face
                scatter(E20[:,d1,d2], areas, DM[2])

        # calc edge lengths
        PM[1] += regular.edge_length(PP10[:, 0, :], PP10[:, 1, :])
        for d1 in range(2):
            for d2 in range(2):
                scatter(
                    E21[:,d1,d2],
                    regular.edge_length(PP21[:, d1, d2, :], PP[2]),
                    DM[1])

        return PM, DM
