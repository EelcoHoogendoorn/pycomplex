
from cached_property import cached_property

from pycomplex.geometry import regular
from pycomplex.complex.cubical import *


def gather(idx, vals):
    """return vals[idx]. return.shape = idx.shape + vals.shape[1:]"""
    return vals[idx]


def scatter(idx, vals, target):
    """target[idx] += vals. """
    np.add.at(target.ravel(), idx.ravel(), vals.ravel())


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

    def plot_primal_0_form(self, f0, ax, **kwargs):
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


class ComplexRegular2(ComplexCubical2Euclidian2):
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

    @cached_property
    def primal_lookup(self):
        # FIXME: this could go in a ComplexRegularMixin?
        assert self.topology.n_dim == self.n_dim
        box = self.box
        quad_position = self.primal_position[2]
        edge_length = self.primal_metric[1][0]
        # assert np.allclose(edge_length, edge_length[0]) # rectangles not supported yet
        def to_grid(x):
            x = x - box[0]
            return x / edge_length - 1
        # add a padding gridcell in all directions to handle out of bounds sampling
        # NOTE: could also use scipy.ndimage.map_coordinates and handle out of bound that way
        grid_shape = ((box[1] - box[0]) / edge_length + 2).astype(np.int)
        grid = -np.ones(grid_shape, dtype=index_dtype)

        local = to_grid(quad_position)
        idx = np.floor(local).astype(np.int)
        grid[tuple(idx.T)] = self.topology.range(2)
        return grid, to_grid

    def pick_primal(self, points):
        """

        Parameters
        ----------
        points : ndarray, [n_points, 2], float

        Returns
        -------
        idx : ndarray, [n_points], index_dtype
            Quad index of each picking point. -1 when sampling out of bounds.
        bary : ndarray, [n_points, 2], float

        """
        grid, to_grid = self.primal_lookup
        local = to_grid(points)
        idx = np.floor(local).astype(index_dtype)
        bary = local - idx
        quad_idx = grid[tuple(idx.T)]
        return quad_idx, bary

    def pick_dual(self, points):
        """Could do something analogous to primal here
        but what about boundary quads?

        Concave case is particularly challenging I suppose
        does plead for a method defined in fundamental domains

        yeah have the best feeling about this.




        """
        raise NotImplementedError
