class StencilEquation(object):
    """Based on a slice of full stencil complex

    do we represent forms and operators as flat structures, or as gridded?
    """
    @property
    def diagonal(self):
        """
        open questions: how do we find the diagonal of normal operator?
        evaluate impulse response on checkerboard-type pattern? do this for each form?
        given A.T * A, we need its diagonals. formed by columns of A dotted with themselves
        mult with impulse gives diag on impulse location.
        checkboard evaluation seems like it would work for all systems considered thus far
        at least it should when normal eq only has second order terms

        maybe richardson smoother is easier after all?
        finding largest eig may take more iters than solving the problem though
        """
        raise NotImplementedError