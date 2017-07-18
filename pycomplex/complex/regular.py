from pycomplex.complex.cubical import *


class ComplexRegular2(ComplexCubical2Euclidian2):
    """Regular cubical 2-complex.
    makes for easy metric calculations, and guarantees orthogonal primal and dual
    """

    def metric(self):
        """calc metric properties and hodges for a 2d regular cubical complex
        sum over subdomains.
        should be relatively easy to generalize to n-dimensions
        """
        from pycomplex.geometry import cubical

        def gather(idx, vals):
            """return vals[idx]. return.shape = idx.shape + vals.shape[1:]"""
            return vals[idx]
        def scatter(idx, vals, target):
            """target[idx] += vals. """
            np.add.at(target.ravel(), idx.ravel(), vals.ravel())

        topology = self.topology
        dual_vertices, dual_edges, dual_faces = self.dual_position()
        primal_vertices, primal_edges, primal_faces = self.primal_position()

        #metrics
        P0, P1, P2 = topology.n_elements
        D0, D1, D2 = P2, P1, P0
        MP0 = np.ones (P0)
        MP1 = np.zeros(P1)
        MP2 = np.zeros(P2)
        MD0 = np.ones (D0)
        MD1 = np.zeros(D1)
        MD2 = np.zeros(D2)

        #precomputations
        E21 = topology.elements[2, 1]     # [faces, e3]
        E10 = topology.elements[1, 0]     # [edges, v2]
        E10P  = self.vertices[E10] # [edges, v2, c3]
        E210P = E10P[E21]          # [faces, e3, v2, c3]
        FEM  = (E210P.mean(axis=2))  # face-edge midpoints; [faces, e3, c3]
        FEV  = E10[E21] # [faces, e3, v2]

        # calc areas of fundamental squares
        for d1 in range(2):
            for d2 in range(2):
                # this is the area of one fundamental domain
                # note that it is assumed here that the primal face center lies within the triangle
                # could we just compute a signed area and would it generalize?
                areas = cubical.area_from_corners(E210P[:, e, 0, :], E210P[:, e, 1, :], primal_faces)
                MP2 += areas                    # add contribution to primal face
                scatter(FEV[:,e,0], areas/2, MD2)

        # calc edge lengths
        MP1 += cubical.edge_length(E10P[:, 0, :], E10P[:, 1, :])
        for e in range(3):
            scatter(
                E21[:,e],
                cubical.edge_length(FEM[:, e, :], primal_faces),
                MD1)

        self.primal_metric = [MP0, MP1, MP2]
        self.dual_metric = [MD0, MD1, MD2]

