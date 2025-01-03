"""implementation of keenan cranes spin transforms paper,
using geometric algebra

https://www.cs.cmu.edu/~kmcrane/Projects/SpinTransformations/paper.pdf
https://github.com/syedalamabbas/SpinXForm/tree/master/SpinXForm
https://github.com/nitronoid/flo/blob/master/test/host_tests/src/test_mean_curvature.cpp#L25

Some bugs still; some quite evident shear components present atm
I dont think the eigen solver is converging
is cg/minres broken, without custom transpose implementation?
note that spinxform implementation above seem to use plain cg with plain inner prod of vectors
i suppose for inner product scalar component, reverse does not matter? or something?

wrt 2d generalization; perhaps curvature should enter as a 1-vector?
without dotting it with a normal?
will happen in geo-prod with ring of 1-edges anyway no?
this construction should interact in the same way in 2d and 3d

what does cnformal in 1d mean anyway? they opt for distance-preserving,
but thats more rigid than conformal.

"""
import numpy as np

from numga.algebra.algebra import Algebra
from numga.backend.numpy.context import NumpyContext as Context
from numga.examples.ga_sparse import *

from pycomplex import synthetic

context = Context(Algebra.from_str('x+y+z+'))


def as_scalar(v):
	return context.multivector.scalar(np.array(v)[..., None])
def as_vector(v):
	return context.multivector.vector(v)

def as_ga_sparse(C, V):
	"""Encode [R, n] adjacency matrix as a sparse matrix [R, C]"""
	I = np.arange(C.size) // C.shape[1]
	return MatrixContainer(
		(len(C), C.max()+1),
		I.flatten(),
		C.flatten(),
		V.flatten()
	)

as_diag = MatrixContainer.diag


def linear_solve(A, x0, b):
	"""wrap scipy minres solver to use GA-typed matrix and vectors"""
	from scipy.sparse.linalg import minres
	r = minres(
		A.as_operator(),
		b.values.flatten(),
		x0.values.flatten()
	)
	return x0.context.multivector(A.row_subspace, r[0].reshape(x0.shape + (-1,)))


def eig_solve(A, v):
	"""Solve for the smallest eigenvector of A using inverse power iteration,
	given an initial eigenvector guess v"""
	def linear_solve_(A, x0, b):
		from scipy.sparse.linalg import cg
		x, info = cg(A, b, x0, maxiter=100)
		assert info==0
		return x

	# flatten into plain LA-types
	def flatten(v):
		return v.values.flatten()
	def reverse(f):
		return v.copy(f.reshape(v.values.shape)).reverse().values.flatten()
	def cast(f):
		return v.copy(f.reshape(v.values.shape))
	vf = flatten(v)
	Af = A.as_operator()
	for i in range(10):
		vf = linear_solve_(Af, vf.copy(), vf.copy())
		Avf = Af(vf)
		# v, Av = cast(vf), cast(Avf)
		lamba = reverse(vf).dot(Avf) / reverse(vf).dot(vf)    # fixme: rayleigh requires full reversal!
		# lamba = np.linalg.norm(Avf) / np.linalg.norm(vf)

		print('eigval', lamba)
		print('residual', np.linalg.norm(Avf - vf * lamba))
		vf = vf / np.linalg.norm(vf)
	return v.copy(vf.reshape(v.values.shape))   # cast back to typed GA


def spin_transform_deform(mesh, rho):
	assert mesh.topology.is_closed
	assert mesh.topology.is_oriented
	assert mesh.topology.is_manifold
	assert mesh.topology.is_connected

	# some boilerplate to convert pycomplex mesh datastructures to GA-sparse matrix operators
	I20 = mesh.topology.incidence[2, 0]   # [F, 3] face-vertex incidence
	I21 = mesh.topology.incidence[2, 1]   # [F, 3] face-edge incidence
	I10 = mesh.topology.incidence[1, 0]   # [E, 2] edge-vertex incidence
	O10 = mesh.topology._orientation[0]   # [E, 2] edge-vertex relative orientations
	O21 = mesh.topology._orientation[1]   # [F, 3] face-edge relative orientations

	T10 = as_ga_sparse(I10, as_scalar(O10))                    # edge-vertex oriented boundary operator
	T21 = as_ga_sparse(I21, as_scalar(O21))                    # face-edge oriented boundary operator
	assert np.all((T21 * T10).values.values == 0)

	A10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) / 2))  # averages vertices over edges
	A20 = as_ga_sparse(I20, as_scalar(np.ones_like(I20) / 3))  # averages vertices over faces

	M2 = as_diag(as_scalar(mesh.primal_metric[2]))          # triangle area matrix
	M2i = as_diag(as_scalar(1 / mesh.primal_metric[2]))     # inverse triangle area matrix

	H1 = as_diag(as_scalar(mesh.compute_edge_ratio))


	vertices = as_vector(mesh.vertices)                     # cast [Vx3] float array to [V] 1vec array
	edges = T10 * vertices                                  # edge vectors from boundary operator

	L = ~T10 * H1 * T10                                     # [V x V] scalar cotan laplacian
	# mean curvature
	# H = L * vertices
	vn = as_vector(mesh.vertex_normals())
	tn = as_vector(mesh.triangle_normals()).normalized()


	# construct [F x V] 1-vector dirac matrix
	# this one is a little tricky;
	# relies on the fact that I20 and I21 refer to opposing vertices and edges in each entry
	# check that I20 and I21 refer to opposing edge-vertex pairs in each entry
	assert not np.any(I20[..., None] == I10[I21])
	D = M2i * (0.5) * as_ga_sparse(I20, edges[I21] * O21)
	# [F x V], pseudoscalar rho matrix
	R = as_diag(as_scalar(rho).dual()) * A20
	R = as_diag(tn * rho) * A20
	A = D - R           # [F x V], odd-grade
	Q = ~A * M2 * A     # [V x V], even-grade, or quaternionic, symmetric-positive-definite matrix
	assert Q.subspace.equals.even_grade()

	# now do an eigen solve
	# start with a unit quat for each vertex
	q = context.multivector.even_grade() * np.ones(mesh.topology.n_vertices)
	q = eig_solve(Q.product(q.subspace), q)
	q = q / q.norm().mean(axis=0)   # remove average scaling

	# obtain transformed edges by sandwiching with the rotor field
	transformed_edges = (A10 * q) >> edges

	# solve for new vertex positions that match transformed edges in least-square sense
	# we seek to approximate `transformed_edges = T10 * transformed_vertices`
	b = ~T10 * H1 * transformed_edges
	transformed_vertices = linear_solve(L.product(vertices.subspace), vertices, b)
	return mesh.copy(vertices=transformed_vertices.values)


def conformal_smooth(mesh):
	assert mesh.topology.is_closed
	assert mesh.topology.is_oriented
	assert mesh.topology.is_manifold
	assert mesh.topology.is_connected

	# some boilerplate to convert pycomplex mesh datastructures to GA-sparse matrix operators
	I20 = mesh.topology.incidence[2, 0]   # [F, 3] face-vertex incidence
	I21 = mesh.topology.incidence[2, 1]   # [F, 3] face-edge incidence
	I10 = mesh.topology.incidence[1, 0]   # [E, 2] edge-vertex incidence
	O10 = mesh.topology._orientation[0]   # [E, 2] edge-vertex relative orientations
	O21 = mesh.topology._orientation[1]   # [F, 3] face-edge relative orientations

	T10 = as_ga_sparse(I10, as_scalar(O10))                    # edge-vertex oriented boundary operator
	T21 = as_ga_sparse(I21, as_scalar(O21))                    # face-edge oriented boundary operator
	assert np.all((T21 * T10).values.values == 0)

	A10 = as_ga_sparse(I10, as_scalar(np.ones_like(I10) / 2))  # averages vertices over edges
	A20 = as_ga_sparse(I20, as_scalar(np.ones_like(I20) / 3))  # averages vertices over faces

	M2 = as_diag(as_scalar(mesh.primal_metric[2]))          # triangle area matrix
	M2i = as_diag(as_scalar(1 / mesh.primal_metric[2]))     # inverse triangle area matrix
	H1 = as_diag(as_scalar(mesh.compute_edge_ratio))

	vertices = as_vector(mesh.vertices)                     # cast [Vx3] float array to [V] 1vec array
	edges = T10 * vertices                                  # edge vectors from boundary operator

	L = ~T10 * H1 * T10     # cotan scalar laplacian on vertices
	# mean curvature
	H = L * vertices
	n = as_vector(mesh.vertex_normals())
	h = H.inner(n)      # signed mean curvature
	I = context.multivector.scalar().dual()

	# check that I20 and I21 refer to opposing edge-vertex pairs in each entry
	assert not np.any(I20[..., None] == I10[I21])
	# [F x V] 1-vector dirac matrix
	# this one is a little tricky;
	# relies on the fact that I20 and I21 refer to opposing vertices and edges in each entry
	D = M2i * (0.5) * as_ga_sparse(I20, edges[I21] * O21)
	# [F x V], pseudoscalar rho matrix
	R = as_diag(as_scalar(rho).dual()) * A20
	A = (D - R)           # [F x V], odd-grade
	Q = ~A * M2 * A     # [V x V], even-grade, or quaternionic, symmetric-positive-definite matrix
	assert Q.subspace.equals.even_grade()

	# now do an eigen solve
	# start with a unit quat for each vertex
	q = context.multivector.even_grade() * np.ones(mesh.topology.n_vertices)

	q = eig_solve(Q.product(q.subspace), q)
	q = q / q.norm().mean(axis=0)   # remove average scaling

	# obtain transformed edges by sandwiching with the rotor field
	transformed_edges = (A10 * q) >> edges

	# solve for new vertex positions that match transformed edges in least-square sense
	# we seek to approximate `transformed_edges = T10 * transformed_vertices`
	b = ~T10 * H1 * transformed_edges
	transformed_vertices = linear_solve(L.product(vertices.subspace), vertices, b)
	return mesh.copy(vertices=transformed_vertices.values)


if False:
	mesh = synthetic.icosphere(2).as_euclidian()
	rho = mesh.topology.chain(2)
	rho[0] = 1e0
elif True:
	cube = synthetic.n_cube(3, centering=True).boundary.as_23()
	for i in range(3):
		cube=cube.subdivide_cubical()
	mesh = cube.subdivide_simplicial().as_3()
	rho = np.sign(mesh.triangle_centroids()[:, 0]) * 1e-0 + 1e-0 * 0
	rho = -np.ones_like(rho) * 2
elif True:
	mesh = synthetic.n_simplex(3, equilateral=True).boundary.as_2().as_3()
	rho = [1, 1, -1, -1]


if False:
	conformal_smooth(mesh)
	quit()
from pycomplex.math.linalg import rotation_from_plane, orthonormalize
R = np.eye(3)
# R = rotation_from_plane([1, 0, 0], [0, 0, 1])
# R = orthonormalize(np.random.normal(size=(3,3)))

deformed = spin_transform_deform(mesh.transform(R), rho)

deformed.save_STL('deformed.stl')

if True:
	import matplotlib.pyplot as plt
	fig, ax = plt.subplots()
	# mesh.plot_3d(ax=ax, plot_dual=False)
	mesh.plot_primal_2_form(rho, ax=ax)

	fig, ax = plt.subplots()
	deformed.plot_3d(ax=ax, plot_dual=False)

	fig, ax = plt.subplots()
	deformed.plot_primal_2_form(rho, ax=ax)

	from pycomplex.math.linalg import rotation_from_plane
	R = rotation_from_plane([1, 0,0], [0, 0, 1])
	deformed = deformed.transform(R)
	fig, ax = plt.subplots()
	deformed.plot_3d(ax=ax, plot_dual=False)

	plt.show()
