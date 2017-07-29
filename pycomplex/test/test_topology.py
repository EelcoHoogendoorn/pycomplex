from pycomplex import topology, synthetic


def test_manifold_butterfly():
    butterfly_faces = [
        [0, 1, 2],
        [2, 3, 4],
    ]
    butterfly = topology.Topology2Triangular(butterfly_faces)

    print(butterfly.check_manifold())
    print(butterfly.is_manifold)
    # vertex 2 should have two connected components
    print(butterfly.regions_per_vertex)


def test_manifold_touching():
    # fan with loose triangle touching in the middle
    faces = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0],
        [2, 4, 5],
    ]
    butterfly = topology.Topology2Triangular(faces)

    print(butterfly.check_manifold())
    print(butterfly.is_manifold)
    # vertex 2 should have two connected components
    print(butterfly.regions_per_vertex)


def test_manifold_sphere():
    sphere = synthetic.icosphere(refinement=0)
    print(sphere.topology.is_manifold)

test_manifold_sphere()