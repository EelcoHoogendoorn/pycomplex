"""Implement convolution stencil based exterior derivatives and grid transfer operators

having these operators present would allow easy testing of vector-laplace friendly mg solvers,
through first order representation and grid transfer with least-squared jacobi-relaxation

should work for both elasticity and stokes; but start with 2d heat transfer
all boundary conditions are of an immersive type for these applications


applications:
    heat solver for wall thickness analysis.
    model mold as higher temperature condition cells, where heat 'leaks away' in proportion to temp?
    with source a func of temp we get a 'shielded poisson' behavior away from surface

    eigen mode solver for stiffness
    use air cell method outside as bc.

    hard to remove support / tight cavities
    stokes with uniform divergece outside the part; check resulting pressure of 'expanding foam'

    heat solver outside the part, to detect tiny hard to cut regions?

    can we do some type of flow analysis inside the part? can we identify
    the point of easiest mold fill, for instance? akin to finding to point most distant from everything,
    in some flow-ease weighted manner


would be neat to create general mechanism that creates code for all stencil operators in all dimensions
may be easier in scipy with wrapping mode. probably a decent start.
given any form, composed of a set of form-maps, we can take derivative wrt each spatial coordinate.
however, need to apply commutation relation before adding fi dxdy and dydx.
with wrap, all feature maps are same size which is nice
otherwise for primal each dx subtracts one from shape in corresp dir,
and for dual we need axis-dependent pad? same as full padding in theano and tf.
doing this with 2-point stencil results in addition of one unknown in that dir, and zero bcs
setting boundary mode to nearest in scipy results in setting gradient bc to zero

note that we are always just adding and subtracting; all mult operators are useless
could make numba kernal that acts on sparse add and sub list?
however, need transfer operators too. tf makes those easy using striding
how to do transfer ops at the boundary tho?

"""

from typing import Tuple
import numpy as np
from scipy import ndimage
from pycomplex.topology.simplicial import permutation_map
import numpy_indexed as npi


def pascal(n, k):
    """Evaluate pascals triangle"""
    if k in (0, n):
        return 1
    return pascal(n - 1, k - 1) + pascal(n - 1, k)


def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a


def generate(ndim):
    """Generate symbols and their derivative relations of anticommutative exterior algebra

    Parameters
    ----------
    ndim : int
        dimension of the space

    Returns
    -------
    symbols : tuple
        describes the interpretation of each component of each form
    terms : tuple
        refers to symbols in the lower forms that derives forms are built from
    axis : tuple
        axis to differentiate any given term to
    parities : tuple
        sign term
    """
    symbols = [(tuple(),)]  # 0-form is represented by empty symbol
    parities = []
    terms = []
    axes = []

    for n in range(ndim):
        par, perm = permutation_map(n)
        p = symbols[n]
        # for new generation of symbols as all previous generations derived wrt all directions
        s = [q + (i,) for i in range(ndim) for q in p if i not in q]
        s = np.array(s)
        ss = np.sort(s, axis=1)
        arg = np.argsort(s, axis=1).astype(par.dtype)
        parity = par[npi.indices(perm, arg)]    # parity of each term relative to sorted order

        idx = npi.as_index(ss)
        u, up = npi.group_by(idx, parity)
        # last dir added to s is the current diff direction. earlier columns are id of terms to diff to reach s
        gs = npi.group_by(idx).split(s)
        gid, gd = gs[..., :-1], gs[..., -1]

        symbols.append(totuple(u))
        parities.append(totuple(up))
        axes.append(totuple(gd))
        terms.append(totuple(gid))

    return tuple(symbols), tuple(terms), tuple(axes), tuple(parities)


class StencilComplex(object):

    """Abstract factory class for stencil based dec operations and mg hierarchies

    The defining distinction of the stencil based approach is that all topology is implicit
    other than its shape it has no associated memory requirements

    make component-first or component-last configurable?
    fix a convention for the meaning of all form components?
    anticommutative algebra works like this: xy = -yx, xx = 0
    at each derivative step, take derivative of each component wrt each dir.
    drop zero terms, aggregate identical symbols

    """

    def __init__(self, shape: Tuple[int], boundary: str = 'periodic'):
        self.boundary = boundary
        self.shape = shape
        self.ndim = len(shape)
        self.symbols, self.terms, self.axes, self.parities = generate(self.ndim)

    def form(self, n, dtype=np.float32):
        """Allocate an n-form"""
        assert self.boundary == 'periodic'
        # with periodic boundaries all forms have the same shape
        components = len(self.symbols[n])
        return np.zeros((components, ) + self.shape, dtype=dtype)

    @property
    def primal(self):
        """Primal derivative operators

        Returns
        -------
        array_like, [ndim-1], operator
            list of stencil operators mapping primal n-form to primal n+1 form
        """
        def conv(*args, **kwargs):
            return ndimage.convolve1d(*args, **kwargs, weights=[-1, +1], mode='wrap')
        def corr(*args, **kwargs):
            return ndimage.correlate1d(*args, **kwargs, weights=[-1, +1], mode='wrap')

        ops = []
        for n, (symbols, terms, axes, parities) in enumerate(zip(self.symbols, self.terms, self.axes, self.parities)):
            def wrapper(n, symbols, terms, axes, parities):
                def foo(f):
                    d = self.form(n + 1)
                    # loop over all components of the new derived form
                    for c, (t, a, p) in enumerate(zip(terms, axes, parities)):
                        # loop over all terms that the new symbol is composed of
                        # could optimize away += for first assignment
                        for T, A, P in zip(t, a, p):
                            i = symbols.index(T)
                            if P:
                                d[c] += conv(f[i], axis=A)
                            else:
                                d[c] -= conv(f[i], axis=A)
                    return d
                return foo

            ops.append(wrapper(n, symbols, terms, axes, parities))
        return ops

    @property
    def dual(self):
        """Dual derivative operators

        Returns
        -------
        array_like, [ndim-1], operator
            list of operators mapping dual n-form to dual n+1 form
        """
        return [c.transpose for c in self.primal]

    @property
    def hodge(self):
        """

        Returns
        -------
        array_like, [ndim], n-form
            for each level of form, an array broadcast-compatible with the domain
        """

    @property
    def smoother(self):
        return smoother(self.ndim)

    @property
    def transfer(self):
        # FIXME: which of two versions? smoothing in all directions, or only those containing zeros?
        raise NotImplementedError

    @property
    def coarsen(self):
        """
        Returns
        -------
        array_like, [ndim], n-form
        """
        return self.transfer

    @property
    def refine(self):
        # assume galerkin transfer operators as a default
        return [c.transpose for c in self.coarsen]


class StencilOperator(object):
    def __init__(self, left: callable, right: callable, shape: Tuple):
        self.left = left
        self.right = right
        self.shape = shape

    @property
    def transpose(self):
        return StencilOperator(
            right=self.left,
            left=self.right,
            shape=(self.shape[1], self.shape[0])
        )

    def __call__(self, *args, **kwargs):
        assert args[0].shape == self.shape[1]
        ret = self.right(*args, **kwargs)
        assert ret.shape == self.shape[0]
        return ret

    def __mul__(self, other):
        assert isinstance(other, type(self))
        assert self.shape[1] == other.shape[0]
        # construct composed operator functions
        def left(*args, **kwargs):
            self.left
        return StencilOperator(
            left=self.left
        )


def binning(arr, steps):
    """inverse of tile"""
    shape = [(a // b, b) for a, b in zip(arr.shape, steps)]
    shape = [c for p in shape for c in p]
    return arr.reshape(shape).sum(axis=tuple(np.arange(len(steps), dtype=np.int) * 2 + 1))


def unbinning():
    """scale a factor two in all directions, but fill some with zeros"""


complex = StencilComplex((2, 3, 4))
f0 = complex.form(0)
f0 = np.arange(24).reshape(1, 2, 3, 4)
f1 = complex.primal[0](f0)
f2 = complex.primal[1](f1)
print(f2)
quit()

class StencilComplex2D(StencilComplex):
    def __init__(self, *args, **kwargs):
        super(self, StencilComplex).__init__(*args, **kwargs)
        assert self.ndim == 2

    def primal(self):
        def conv(*args, **kwargs):
            return ndimage.convolve1d(*args, **kwargs, weights=[-1, +1], mode='wrap')
        def corr(*args, **kwargs):
            return ndimage.correlate1d(*args, **kwargs, weights=[-1, +1], mode='wrap')

        def p01(p0):
            p1 = self.form(1)
            for i in range(2):
                conv(p0[0], output=p1[i], axis=i)
            return p1
        def p12(p1):
            p2 = self.form(2)
            p2[0] += conv(p1[0], axis=1)
            p2[0] -= conv(p1[1], axis=0)
            return p2

        return [p01, p12]

    def coarsen(self):
        """

        Returns
        -------

        Notes
        -----
        conv and corr are the same here, considering symmetric smoothing stencil
        """
        def c0(p0):
            s = ndimage.convolve(p0, weights=self.smoother, mode='wrap')
            return s[:, ::2, ::2]
        def c1(p1):
            s = np.copy(p1)
            ndimage.convolve(p1[0], output=s[0], weights=self.smoother, mode='wrap')
            ndimage.convolve(p1[1], output=s[1], weights=self.smoother, mode='wrap')
            x = s[0, :, ::2]
            y = s[1, ::2, :]
            return np.array([
                binning(x, [2, 1]),
                binning(y, [1, 2]),
            ])
        def c2(p2):
            s = ndimage.convolve(p2[0], weights=self.smoother, mode='wrap')
            return binning(s, [2, 2])

        return [c0, c1, c2]

    def refine(self):
        """

        Returns
        -------

        Notes
        -----
        These are effectively transposes of the coarsen operator
        """
        def r0(p0):
            # tile with zeros inbetween; then convolve
            return ndimage.convolve(s, weights=self.smoother, mode='wrap')

        def r1(p1):
            pass

        def r2(p2):
            s = np.tile(p2, (2, 2))
            return ndimage.convolve(s, weights=self.smoother, mode='wrap')

        return [r0, r1, r2]



class StencilComplex3D(StencilComplex):
    def __init__(self, *args, **kwargs):
        super(self, StencilComplex).__init__(*args, **kwargs)
        assert self.ndim == 3

    def primal(self):
        def conv(*args, **kwargs):
            return ndimage.convolve1d(*args, **kwargs, weights=[-1, +1], mode='wrap')
        def p01(p0):
            p1 = self.form(1)
            for i in range(3):
                conv(p0[0], output=p1[i], axis=i)
            return p1
        def p12(p1):
            p2 = self.form(2)
            p2[0] += conv(p1[1], axis=2)
            p2[0] -= conv(p1[2], axis=1)

            p2[1] += conv(p1[2], axis=0)
            p2[1] -= conv(p1[0], axis=2)

            p2[2] += conv(p1[0], axis=1)
            p2[2] -= conv(p1[1], axis=0)

            return p2
        def p23(p2):
            p3 = self.form(3)
            p3[0] += conv(p2[0], axis=0)
            p3[0] += conv(p2[1], axis=1)
            p3[0] += conv(p2[2], axis=2)
            return p3
        return [p01, p12, p23]


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
        return


import numpy as np
def smoother(ndim):
    """Note: this a seperable kernel. """
    if ndim == 1:
        return np.array([1, 2, 1]) / 4
    return smoother(1) * smoother(ndim-1)[..., None]

print(smoother(2) * 2)
print(smoother(4).sum())
quit()

def numba_gen():
    """try and generate numba code to run our kernels

    advantages; full unrolling, multiplies with ones and zeros optimized away,
    wraparound logic can be selectively emitted at the start of each loop body

    note that for diff operators we only need +1 offset
    can optimize wrap if shape is known at compile time
    transfer ops do require negative offset.
    still, only use -1, 0 and +1
    if using scipy for transfer no negatives needed here either

    can we omit conditional in inner loop entirely?
    maybe require power of two for the contiguous axis, so when we advance indices we get
    wrapping for free using bitmask
    can also perform the wrapping as an unrolled case; plain loop over pure incremental part
    and invoke body again with more expensive indexing logic a few times

    note that curl stencil in 3d has 12 terms, but writes to only 3 different mem locations
    is this 4 time write coalesed automatically? or should i manually do so?
    probably; += requires a read and init. write-once architecture is to be preferred
    if [1/4, 1/2, 1/4] with read step 2 defines coarsening,
    prolongation is write step 2? but write step 2 makes no sense in single-write arch.
    shit. either we need a variable kernel/loop body, or read/write operation
    can do += only when having multiple write points? if write cache can be trusted it should not matter

    however, does make tf seem more attractive again. if perf is to be cometitive mem efficieny is key

    or can we split striding and conv? first write with stride 2, then convolve with restriction filter
    works for 0-form. but 1-form?

    what is nd-transfer operator anyway? maps from parent to child, then apply smoothing kernel?
    note that we do not need any galerkov properties
    or do we only smooth in direction where we have zeros?
    if coarsen is fine smooth followed by mapping to parent, is this still galerkin?

    pure restrict: read step 2, write step 1. nonzero weight in correct hypercube pattern
    pure prolongate; read step 1, write step 2. write to each output pixel once so no  += needed
    could be implemented as pure numpy

    then blur with standard pattern; scipy is fine here due to independence per dir
    also, we typically do 3 smooths per transfer; so smooths dominate performance
    also formulating normal equations doubles work again.
    in total we have at least 2 reads and 2 writes per unknown for elasticity for a single linear system
    so 4 reads and 4 writes per unknown per normal smooth iteration, or 12 for 3 iterations
    2-step prolongation is write-read-write on the finest level, so not much point optimizing it more

    assuming a 50gb/s memory bandwidth and 1gb of state, this gives an idea of peak performance;
    single v-cycle (with pre- and post smooth) will not be faster than 1sec total.
    this is ignoring the fact that in 2d we read all memory banks 2 times, and in 3d 2 times;
    only get a wavefront along last axis in naive implementaion.
    optimal implementaion would tile the first axis and move on wavefronts along it,
    such that entire ndim-1 front fits in cache. tiles can be worked in parralel
    scipy version does not look that efficient either thb.

    note that blur kernel is seperable, and derivatives can all be expressed as 1d operations too
    1d operation is easily done in mem-efficient manner

    in conclusion; plenty of room for optimizations, but for the time being just build a numpy solution

    need only handfull of v-cycles for thermal or foam simulation.
    for elastic eigensolver we should strive for an algo that converges in few outer iterations.

    """
    n_dim = 2
    # read and write strides
    ws = 1
    rs = 1
    assert read.shape * rs == write.shape * ws
    # unroll the loop over the sparse stencil
    body = """
    p = p0, p1, p2
    wp = p * ws
    # where we add the offset, we need to add wrapping logic
    rp = p * rs + o
    write[{wp}, {wl}] += read[rp, rl] * w
    
    """

    body2d = """
    for p0 in range(shape0):
        rp0 = p0 * rs
        wp0 = p0 * ws
        rp0_0 = rp0 + -1; if rp0_0 < 0: rp0_0 += read.shape[0]
        rp0_1 = rp0 + 0
        rp0_2 = rp0 + 1; if rp0_0 >= read.shape[0]: rp0_0 -= read.shape[0]
        for p1 in range(shape1):
        
            for w, ro, wo, rl, wl in stencil:
                write[wp0, wp1, wl] += read[rp0, rp1, rl] * w
    """


def numba():
    """Some numba exeriments; seems flexibility is low; cannot efficiently act on all data"""
    from numba import stencil

    @stencil
    def dx(a):
        return a[0, 0, 0] - a[1, 0, 0]
    @stencil
    def dy(a):
        return a[0, 0, 0] - a[0, 1, 0]
    @stencil
    def dz(a):
        return a[0, 0, 0] - a[0, 0, 1]

    @stencil
    def div(x, y, z):
        return (x[0, 0, 0] - x[1, 0, 0]) + (y[0, 0, 0] - y[0, 1, 0]) + (z[0, 0, 0] - z[0, 0, 1])

    def grad(f):
        shape = np.array(f.shape)
        x = np.empty(shape - [1, 0, 0])
        dx(a, out=x)
        y = np.empty(shape - [0, 1, 0])
        dy(a, out=y)
        z = np.empty(shape - [0, 0, 1])
        dz(a, out=z)
        return x, y, z

    @stencil
    def grad(a):
        return

    import numpy as np
    a = np.arange(2*3*4).reshape(2, 3, 4)


    g = (grad(a))

    print(div(*g))



def theano_3d():
    """using theano, we can apply a gradient filter to a 0-form
    compute intensity of typical 1st order stencil is low; so we want to compute
    different directions in a single pass
    only 2 ops out of a 8-corner cube would be used in a grad op implemented this way

    modern tesla; 900GB/s and 16Tflops 32bit perf; implies we can do 64 fp32 ops for every read
    implies extra multiplies with 0 are fine if it allows us to use optimal conv code

    https://folk.uio.no/marcink/krotkiewski2010_gpu_stencil.pdf
    100gflops out of 1tflop on C2050; 144gbs bandwidth

    https://arxiv.org/pdf/1410.0759.pdf
    cudnn claims 2 out of 4 Tflops on maxwell
    many input/output channels bumps intensity though

    https://github.com/NervanaSystems/nervanagpu#refs
    even 6 tflops claimed; not sure about using neon though

    so with specialized 2.5d kernel we can save op to a factor 9 in flops relative to convolution
    but seems like conserving mem bandwidth is indeed the real challenge.

    single copy of state for a volume set of mode vectors is easily a gb
    50 vecs is 20mb per vec or 5m entries, or 200^3 grid, 0form only. including vector forms only 100^3 still

    maybe do eigen solve on cpu and big ram, and precondition solves on gpu?
    we can interleave solving for individual or batches of vectors to hide mem transfers
    in matrix-free method there is minimal benefit to further vectorizing

    often we may wish for scalar field multiplies right after or before conv;
    would be nice to have lib that supports this

    all well and good but first priority should be making something that works
    note that gpu advantage is limited by compute intensity of a single conv stencil

    """
    import numpy as np
    grad_filters = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)
    grad_filters[0, 0, 0, 0, 0] = -1
    grad_filters[0, 0, 1, 0, 0] = +1
    grad_filters[0, 1, 0, 0, 0] = -1
    grad_filters[0, 1, 0, 1, 0] = +1
    grad_filters[0, 2, 0, 0, 0] = -1
    grad_filters[0, 2, 0, 0, 1] = +1

    print('grad intensity (1/4.0)')
    print(grad_filters.size / np.count_nonzero(grad_filters))
    dual_div_filters = np.swapaxes(grad_filters, 0, 1)

    curl_filters = np.zeros((3, 3, 2, 2, 2), dtype=np.float32)

    curl_filters[1, 0, 0, 0, 0] = +1
    curl_filters[1, 0, 0, 0, 1] = -1
    curl_filters[2, 0, 0, 0, 0] = -1
    curl_filters[2, 0, 0, 1, 0] = +1

    curl_filters[2, 1, 0, 0, 0] = +1
    curl_filters[2, 1, 1, 0, 0] = -1
    curl_filters[0, 1, 0, 0, 0] = -1
    curl_filters[0, 1, 0, 0, 1] = +1

    curl_filters[0, 2, 0, 0, 0] = +1
    curl_filters[0, 2, 0, 1, 0] = -1
    curl_filters[1, 2, 0, 0, 0] = -1
    curl_filters[1, 2, 1, 0, 0] = +1

    print('curl intensity (1/6)')
    print(curl_filters.size / np.count_nonzero(curl_filters))

    # this is grad filter with swapped axes
    div_filters = np.zeros((3, 1, 2, 2, 2), dtype=np.float32)
    div_filters[0, 0, 0, 0, 0] = +1
    div_filters[0, 0, 1, 0, 0] = -1
    div_filters[1, 0, 0, 0, 0] = +1
    div_filters[1, 0, 0, 1, 0] = -1
    div_filters[2, 0, 0, 0, 0] = +1
    div_filters[2, 0, 0, 0, 1] = -1



    # transfer operators between levels could be implemented in the same manner
    # strides = 2 and 3^n multilinear stencil
    # restriction is just conv with stride equals 2
    R0_filter = np.ones((1, 1, 1, 1, 1), dtype=np.float32)

    I0_1d = np.array([1/4, 1/2, 1/4])
    I0_2d = I0_1d[..., None] * I0_1d
    I0_3d = I0_2d[..., None] * I0_1d

    # interpolation is transposed conv with stride 2
    # print(I0_3d)
    # print(I0_3d.sum())


    def grad_func():
        """Full mode inserts a junk padding at the start of each axis
        padding at the end of each axis may or may not be junk
        """
        import theano
        from theano import tensor as T

        X = T.ftensor5()
        from theano.tensor.nnet import conv3d
        q = np.swapaxes(grad_filters, 0, 1)
        grad_ops = conv3d(X, q, border_mode='full', filter_flip=False)
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)


    def curl_func():
        """Full mode inserts a junk padding at the start of each axis
        padding at the end of each axis may or may not be junk
        """
        import theano
        from theano import tensor as T

        X = T.ftensor5()
        from theano.tensor.nnet import conv3d
        q = np.swapaxes(curl_filters, 0, 1)
        grad_ops = conv3d(X, q, border_mode='full', filter_flip=False)
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)


    def div_func():
        """Full mode inserts a junk padding at the start of each axis
        padding at the end of each axis may or may not be junk
        """
        import theano
        from theano import tensor as T

        X = T.ftensor5()
        from theano.tensor.nnet import conv3d
        q = np.swapaxes(div_filters, 0, 1)
        grad_ops = conv3d(X, q, border_mode='full', filter_flip=False)
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)


    def grad_dir_func(d):
        import theano
        from theano import tensor as T

        X = T.ftensor5()
        from theano.tensor.nnet import conv3d
        shape = [1] * 5
        shape[d + 2] = 2
        f = np.array([-1, +1], np.float32).reshape(shape)

        grad_ops = conv3d(X, f, border_mode='valid', filter_flip=False)
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)


    def curl_dir_func(d):
        """Need to pass in two 1-form components here; cant really express this as a conv;
        has to be two convs plus addition... sucks bandwidth-wise"""
        import theano
        from theano import tensor as T

        X = T.ftensor5()
        Y = T.ftensor5()
        from theano.tensor.nnet import conv3d
        shape = [1] * 5
        shape[d + 2] = 2
        f = np.array([-1, +1], np.float32).reshape(shape)

        grad_ops = conv3d(X, f, border_mode='valid', filter_flip=False)
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)


    def pad(x):
        y = np.zeros([s+1 for s in x.shape])
        y[:, :, :-1, :-1, :-1] = x
        return y


    f = np.arange(1*3*4*5).reshape(1, 1, 3, 4, 5).astype(np.float32)
    f = np.random.rand(1, 1, 3, 4, 5).astype(np.float32)

    # print(grad_dir_func(1)(f))
    # f = pad(f)

    g = grad_func()(f)[:, :, 1:, 1:, 1:]
    g = np.random.normal(size=g.shape).astype(np.float32)
    c = curl_func()(g)[:, :, 1:, 1:, 1:]
    d = div_func()(c)[:, :, 1:, 1:, 1:]
    print(g.shape)
    # for q in g[0]:
    #     print(q)
    #     print()

    # print(np.gradient(f[0, 0]))
    # quit()

    print(c.shape)
    # for q in c[0]:
    #     print(q)
    #     print()
    # print(c.shape)
    # print (np.gradient(f[0,0]))
    print(d)


def theano_2d():
    """prototyping key ideas is easier in 2d

    for bcs, they are always assumed zero in this approach
    actively modelled dual variables have 1-1 with primal
    """

    import numpy as np
    import theano
    from theano import tensor as T
    from theano.tensor.nnet import conv2d
    from theano.tensor.nnet.abstract_conv import conv2d_grad_wrt_inputs as conv_t

    grad_filters = np.zeros((1, 2, 2, 2), dtype=np.float32)
    grad_filters[0, 0, 0, 0] = -1
    grad_filters[0, 0, 1, 0] = +1
    grad_filters[0, 1, 0, 0] = -1
    grad_filters[0, 1, 0, 1] = +1

    print('grad intensity (1/2)')
    print(grad_filters.size / np.count_nonzero(grad_filters))
    dual_div_filters = np.swapaxes(grad_filters, 0, 1)

    curl_filters = np.zeros((2, 1, 2, 2), dtype=np.float32)

    curl_filters[0, 0, 0, 0] = +1
    curl_filters[0, 0, 0, 1] = -1
    curl_filters[1, 0, 0, 0] = -1
    curl_filters[1, 0, 1, 0] = +1

    print('curl intensity (1/2)')
    print(curl_filters.size / np.count_nonzero(curl_filters))


    def grad_func():
        """Full mode inserts a junk padding at the start of each axis
        padding at the end of each axis may or may not be junk
        """

        X = T.ftensor4()
        q = np.swapaxes(grad_filters, 0, 1)
        grad_ops = conv2d(X, q, border_mode='full', filter_flip=False)
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)

    def curl_func():
        """Full mode inserts a junk padding at the start of each axis
        padding at the end of each axis may or may not be junk
        """
        X = T.ftensor4()
        q = np.swapaxes(curl_filters, 0, 1)
        grad_ops = conv2d(X, q, border_mode='full', filter_flip=False)
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)

    def downsample_0():
        """downsample a 0-form"""
        X = T.ftensor4()
        f1d = np.array([1/4, 1/2, 1/4]).reshape(1, 1 ,3)
        filters = f1d[..., :, None] * f1d[..., None, :]
        q = np.swapaxes(filters, 0, 1)
        grad_ops = conv2d(
            X, q,
            subsample=(2, 2),
            border_mode='full',
            filter_flip=False
        )
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)

    def upsample_0():
        """upsample a 0-form; or interpolate"""
        X = T.ftensor4()
        f1d = np.array([1/4, 1/2, 1/4]).reshape(1, 1, 3) * 2
        filters = f1d[..., :, None] * f1d[..., None, :]
        q = np.swapaxes(filters, 0, 1)
        grad_ops = conv_t(
            X, q,
            input_shape=(None, None, 3, 3),
            subsample=(2, 2),
            border_mode='full',
            filter_flip=False
        )
        return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)


    f = np.arange(1*3*4).reshape(1, 1, 3, 4).astype(np.float32)
    f = np.random.rand(1, 1, 3, 4).astype(np.float32)

    # print(grad_dir_func(1)(f))
    # f = pad(f)

    fu = upsample_0()(f)
    print(fu.shape)

    # g = grad_func()(f)[:, :, 1:, 1:]
    # # g = np.random.normal(size=g.shape).astype(np.float32)
    # c = curl_func()(g)[:, :, 1:, 1:]
    # # d = div_func()(c)[:, :, 1:, 1:]
    # print(g)
    # print(g.shape)
    # print(c)

# theano_2d()

def tensorflow_2d():
    """Look into tfe eager evaluation functionality to make this cleaner"""
    import numpy as np
    import tensorflow as tf

    grad_filters = np.zeros((1, 2, 2, 2), dtype=np.float32)
    grad_filters[0, 0, 0, 0] = -1
    grad_filters[0, 0, 1, 0] = +1
    grad_filters[0, 1, 0, 0] = -1
    grad_filters[0, 1, 0, 1] = +1

    print('grad intensity (1/2)')
    print(grad_filters.size / np.count_nonzero(grad_filters))
    dual_div_filters = np.swapaxes(grad_filters, 0, 1)

    curl_filters = np.zeros((2, 1, 2, 2), dtype=np.float32)

    curl_filters[0, 0, 0, 0] = +1
    curl_filters[0, 0, 0, 1] = -1
    curl_filters[1, 0, 0, 0] = -1
    curl_filters[1, 0, 1, 0] = +1

    print('curl intensity (1/2)')
    print(curl_filters.size / np.count_nonzero(curl_filters))


    def grad_func():
        """Full mode inserts a junk padding at the start of each axis
        padding at the end of each axis may or may not be junk
        """
        q = np.einsum('abxy->xyab', grad_filters)
        grad_tensor = tf.constant(value=grad_filters)
        # q = np.swapaxes(grad_filters, 0, 1)
        def inner(inp):
            inp = tf.Variable(name='grad_input')
            print(inp.shape)
            return tf.nn.convolution(
                inp,
                q,
                'SAME'
            )
        return inner

    # def curl_func():
    #     """Full mode inserts a junk padding at the start of each axis
    #     padding at the end of each axis may or may not be junk
    #     """
    #     X = T.ftensor4()
    #     q = np.swapaxes(curl_filters, 0, 1)
    #     grad_ops = conv2d(X, q, border_mode='full', filter_flip=False)
    #     return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)
    #
    # def downsample_0():
    #     """downsample a 0-form"""
    #     X = T.ftensor4()
    #     f1d = np.array([1/4, 1/2, 1/4]).reshape(1, 1 ,3)
    #     filters = f1d[..., :, None] * f1d[..., None, :]
    #     q = np.swapaxes(filters, 0, 1)
    #     grad_ops = conv2d(
    #         X, q,
    #         subsample=(2, 2),
    #         border_mode='full',
    #         filter_flip=False
    #     )
    #     return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)
    #
    # def upsample_0():
    #     """upsample a 0-form; or interpolate"""
    #     X = T.ftensor4()
    #     f1d = np.array([1/4, 1/2, 1/4]).reshape(1, 1, 3) * 2
    #     filters = f1d[..., :, None] * f1d[..., None, :]
    #     q = np.swapaxes(filters, 0, 1)
    #     grad_ops = conv_t(
    #         X, q,
    #         input_shape=(None, None, 3, 3),
    #         subsample=(2, 2),
    #         border_mode='full',
    #         filter_flip=False
    #     )
    #     return theano.function(inputs=[X], outputs=grad_ops, allow_input_downcast=False)

    with tf.Session() as sess:
        f = np.arange(1*3*4).reshape(1, 1, 3, 4).astype(np.float32)
        f = np.random.rand(3, 4).astype(np.float32)

        grad = grad_func()(f)

        result = sess.run([grad], feed_dict={'grad_input': f[None, ..., None]})
        print()

tensorflow_2d()
