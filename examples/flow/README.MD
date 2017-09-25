Various flow problems are solved in this module.

The intent here is to showcase working with complexes,
and their boundary structure and boundary conditions in particular

The general idea is that we write all flow equations in their first order form,
which highlights the necessary and possible boundary conditions.

We may then solve these equations in a least squares sense.
Premultiplying the first order system with itself leads to a second order symmetric diagonal dominant system,
With a laplacian structure on the diagonals, with their typical condition numbers.
These normal equations are easily solved with Krylov subspace solvers or Jacobi iteration.
