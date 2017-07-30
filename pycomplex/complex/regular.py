from pycomplex.complex.cubical import *


class ComplexRegular2(ComplexCubical2Euclidian2):
    """Regular cubical 2-complex.
    makes for easy metric calculations, and guarantees orthogonal primal and dual
    """

    def metric(self):
        """Calc metric properties and hodges for a 2d regular cubical complex

        Notes
        -----
        Should be relatively easy to generalize to n-dimensions
        Are we interested in computing dual boundary metric too? Or just leave that to the boundaty complex?
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
        PP0, PP1, PP2 = self.primal_position

        #metrics
        P0, P1, P2 = topology.n_elements
        D0, D1, D2 = P2, P1, P0
        MP0 = np.ones (P0)
        MP1 = np.zeros(P1)
        MP2 = np.zeros(P2)
        MD0 = np.ones (D0)
        MD1 = np.zeros(D1)
        MD2 = np.zeros(D2)

        # precomputations
        E20  = topology.incidence[2, 0]  # [faces, e2, e2]      vertex indices per face
        E21  = topology.incidence[2, 1]  # [faces, e2, e2]      edge indices per face
        E10  = topology.incidence[1, 0].reshape(-1, 2)  # [edges, v2]          vertex indices per edge
        # E210 = E10[E21]                  # [faces, e2, e2, v2]  vertex indices per edge per face

        PP10  = PP0[E10]                 # [edges, v2, c3]
        # PP210 = PP10[E21]                # [faces, e2, e2, v2, c3]
        PP21  = PP1[E21]                 # [faces, e2, e2, c3] ; face-edge midpoints
        PP20  = PP0[E20]                 # [faces, e2, e2, c3]
        # calc areas of fundamental squares
        for d1 in range(2):
            for d2 in range(2):
                # this is the area of one fundamental domain, assuming regular coords
                areas = regular.hypervolume(PP20[:, d1, d2, :], PP2)
                MP2 += areas                    # add contribution to primal face
                scatter(E20[:,d1,d2], areas, MD2)

        # calc edge lengths
        MP1 += regular.edge_length(PP10[:, 0, :], PP10[:, 1, :])
        for d1 in range(2):
            for d2 in range(2):
                scatter(
                    E21[:,d1,d2],
                    regular.edge_length(PP21[:, d1, d2, :], PP2),
                    MD1)

        self.primal_metric = [MP0, MP1, MP2]
        self.dual_metric = [MD0, MD1, MD2]

        self.hodge_from_metric()

    def hodge_from_metric(self):
        MP = self.primal_metric
        MD = self.dual_metric
        #hodge operators
        self.D2P0 = MD[2] / MP[0]
        self.P0D2 = MP[0] / MD[2]

        self.D1P1 = MD[1] / MP[1]
        self.P1D1 = MP[1] / MD[1]

        self.D0P2 = MD[0] / MP[2]
        self.P2D0 = MP[2] / MD[0]
