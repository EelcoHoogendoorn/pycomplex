
from cached_property import cached_property

from pycomplex.complex.cubical import *


# class ComplexRegular1(ComplexCubical1Euclidian1):
#     """Regular cubical 1 complex; or line segment"""


class ComplexRegular2(ComplexCubical2Euclidian2):
    """Regular cubical 2-complex.
    makes for easy metric calculations, and guarantees orthogonal primal and dual
    """
    # FIXME: drop dependence on euclidian2; would like to be able to process boundaries of regulars as well

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
        from pycomplex.geometry import regular

        def gather(idx, vals):
            """return vals[idx]. return.shape = idx.shape + vals.shape[1:]"""
            return vals[idx]
        def scatter(idx, vals, target):
            """target[idx] += vals. """
            np.add.at(target.ravel(), idx.ravel(), vals.ravel())

        topology = self.topology
        # dual_vertices, dual_edges, dual_faces = self.dual_position()
        PP = self.primal_position

        #metrics
        PN = topology.n_elements
        DN = PN[::-1]
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
