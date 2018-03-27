"""Generalized homogeneous poisson; or laplace-beltrami

Use this to test non-scalar multigrid
"""

import numpy as np
import scipy.sparse
from cached_property import cached_property

from examples.multigrid.equation import Equation
from pycomplex.sparse import normalize_l1, inv_diag


class Laplace(Equation):
    """laplace-beltrami on dual k-forms

    A * x = B * y, or laplacian(x) = mass(y)

    Notes
    -----
    need to think of clear interface to extend this to multicomplex as well.
    what do we truly expect as interface from the complex that we bind to here?
    at what level do we translate from possible block system to simple monolithic matrices?
    """

    def __init__(self, complex, k=-2):
        """

        Parameters
        ----------
        complex : BaseComplex
        k : int
            laplace order, referring to primal elements

        """
        # FIXME: add modifier terms for all diagonals
        self.complex = complex
        self.k = k if k > 0 else complex.n_dim + k + 1

    @cached_property
    def operators(self):
        """Construct laplace-beltrami on primal k-forms

        Returns
        -------
        A : Linear operator
            symmetric;
            maps primal k-form to dual k-form
        B : Linear operator
            symmetric diagonal;
            maps primal k-form to dual k-form
        BI :
            inverse of B
        """
        k = self.k
        DP = [scipy.sparse.diags(h) for h in self.complex.hodge_DP]
        PD = [scipy.sparse.diags(h) for h in self.complex.hodge_PD]

        DP = ([0] + DP + [0])[k:][:3]
        PD = ([0] + PD + [0])[k:][:3]

        DPl, DPm, DPr = DP
        PDl, PDm, PDr = PD

        # using the primal implies no fancy boundary treatment for now
        Dl, Dr = ([0] + self.complex.topology.matrices + [0])[k:][:2]
        Pl, Pr = np.transpose(Dl), np.transpose(Dr)

        A = Pl * PDl * Dl + \
            PDm * Dr * DPr * Pr * PDm

        B = DPm
        BI = inv_diag(B)

        return A.tocsr(), B.tocsc(), BI.tocsc()

    # @cached_property
    # def null_space(self):
    #     """Return null space of A"""
    #     A, B, BI = self.operators
    #
    #     return np.ones((B.shape[1], 1))

    def solve(self, y):
        """Solve linear system in eigenbasis

        Parameters
        ----------
        y : ndarray
            k-form

        Returns
        -------
        x : ndarray
            k-form
        """
        A, B, BI = self.operators

        def poisson(y, v):
            x = np.zeros_like(y)
            # poisson linear solve is simple division in eigenspace. skip nullspace
            null_rank = (np.abs(v) / np.max(v) < 1e-9).sum()
            # swivel dimensions to start binding broadcasting dimensions from the left
            x[null_rank:] = (y[null_rank:].T / v[null_rank:].T).T
            return x

        return self._solve_eigen(B * y, poisson)

    # FIXME: mg operators belong to the complexes themselves
    @cached_property
    def transfer(self):
        """Transfer operator of state variables between fine and coarse representation

        Returns
        -------
        sparse matrix
            entry [c, f] is the overlap between fine and coarse dual n-cells

        Notes
        -----
        only available for SphericalTriangularComplex; implementation on regular grids should be easy tho
        """
        try:
            # triangular 0-form special case
            fine = self.complex
            coarse = fine.parent
            return self.complex.multigrid_transfer_dual(coarse, fine).T
        except:
            return self.complex.multigrid_transfers[self.k]

    @cached_property
    def restrictor(self):
        """Maps dual of primal k-form from fine to coarse"""
        A, B, BI = self.operators
        # FIXME: need a link to parent equation; this is hacky. should be coarse.BI
        # convert to dual; then transfer, then back to primal. why again?
        # fine = scipy.sparse.diags(self.complex.primal_metric[self.k])
        # coarse = scipy.sparse.diags(self.complex.parent.hodge_PD[self.k])
        fine = scipy.sparse.diags(1. / self.complex.dual_metric[self.k])
        coarse = scipy.sparse.diags(self.complex.parent.dual_metric[self.k])

        return coarse * normalize_l1(self.transfer.T, axis=0) * fine
    def restrict(self, fine):
        """Restrict solution from fine to coarse"""
        return self.restrictor * fine
    @cached_property
    def interpolator(self):
        """Maps dual of primal k-form from coarse to fine"""
        fine = scipy.sparse.diags(self.complex.dual_metric[self.k])
        coarse = scipy.sparse.diags(1. / self.complex.parent.dual_metric[self.k])
        return fine * normalize_l1(self.transfer, axis=1) * coarse
    def interpolate(self, coarse):
        """Interpolate solution from coarse to fine"""
        return self.interpolator * coarse

    def petrov_galerkin(self):
        """Generate operator appropriate for the parent complex"""
        A, B, BI = self.operators
        return self.restrictor * A * self.interpolator


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    kind = 'regular'

    # if kind == 'sphere':
    #     from pycomplex import synthetic
    #     complex = synthetic.icosphere(refinement=6)
    #     complex = complex.copy(radius=16)

    if kind == 'simplicial':
        from pycomplex import synthetic
        complex = synthetic.delaunay_cube(density=64, n_dim=2)
        complex = complex.copy(vertices=complex.vertices - 0.5).optimize_weights().as_2().as_2()
        # NOTE: no hierarchy so not compatible with geometric mg

    if kind == 'regular':
        from pycomplex import synthetic
        complex = synthetic.n_cube_grid((1, 1)).as_22().as_regular()
        hierarchy = [complex]
        for i in range(6):
            complex = complex.subdivide_cubical()
            hierarchy.append(complex)


    # setup vectorial geometric multigrid solver
    equations = [Laplace(c, k=-2) for c in hierarchy]
    from examples.multigrid import multigrid
    mg_preconditioner = multigrid.as_preconditioner(equations)
    equation = equations[-1]
    print(equation.largest_eigenvalue)


    # test transfer operators
    if True:
        R = equations[-1].restrictor
        I = equations[-1].interpolator
        T = R * I
        # ones isnt in the nullspace of vector laplace
        null = equations[-2].eigen_basis(K=3)[0][:, 0]
        # ones = np.ones((T.shape[1], 1))
        q = T * null
        print(q)
        # FIXME: q / null tends to be around 4 atm
        # assert np.allclose(q, 1)
        T = I * R
        null = equations[-1].eigen_basis(K=3)[0][:, 0]
        # ones = np.ones((T.shape[1], 1))
        q = T * null
        print(q)
        assert np.allclose(q, 1)


    def plot_flux(fd1):
        ax = plt.gca()
        complex.plot_dual_flux(fd1, plot_lines=True, ax=ax)

        ax.set_xlim(*complex.box[:, 0])
        ax.set_ylim(*complex.box[:, 1])
        plt.axis('off')


    # toggle between different outputs
    output = 'modes'
    if output == 'modes':
        # output eigenmodes
        # path = r'../output/seismic_modes_0'
        # from examples.util import save_animation
        from time import clock
        t = clock()
        # FIXME: geometric mg only stable for very low k so far
        # FIXME: equation.restrictor * equations.interpolator far from identity!
        V, v = equation.eigen_basis(K=50, preconditioner=mg_preconditioner, tol=1e-6)
        print('eigen solve time:', clock() - t)
        print(v)
        # quit()
        # for i in save_animation(path, frames=len(v), overwrite=True):
        #     plot_flux(V[:, i] * (r**0) * 1e-2)
