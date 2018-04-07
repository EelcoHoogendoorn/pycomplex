
import pytest

from pycomplex import topology, synthetic


def test_manifold_butterfly():
    """Test that a butterfly topology is not manifold"""
    butterfly_faces = [
        [0, 1, 2],
        [2, 3, 4],
    ]
    from pycomplex.topology.simplicial import TopologyTriangular
    topology = TopologyTriangular.from_simplices(butterfly_faces)

    assert not topology.is_manifold
    # vertex 2 should have two connected components
    print(topology.regions_per_vertex)
    assert topology.regions_per_vertex[2] == 2


def test_manifold_touching():
    """Test that a triangle touching a manifold is not manifold"""
    # fan with loose triangle touching in the middle
    faces = [
        [0, 1, 2],
        [1, 2, 3],
        [2, 3, 0],
        [2, 4, 5],
    ]
    from pycomplex.topology.simplicial import TopologyTriangular
    topology = TopologyTriangular.from_simplices(faces)

    assert not topology.is_manifold
    # vertex 2 should have two connected components
    print(topology.regions_per_vertex)
    assert topology.regions_per_vertex[2] == 2


def test_manifold_sphere():
    """Test that a sphere is manifold"""
    assert synthetic.icosphere(refinement=0).topology.is_manifold


def test_manifold_hexacosichoron():
    """Test that a hexacosichoron is manifold"""
    assert synthetic.hexacosichoron().topology.is_manifold


def test_manifold_nsphere():
    """Test that an n-sphere is manifold"""
    for n_dim in [2, 3, 4, 5]:
        complex = synthetic.n_cube_dual(n_dim)
        assert complex.topology.is_manifold


def test_manifold_ncube():
    """Test that an n-cube is manifold"""
    for n_dim in [2, 3, 4, 5]:
        complex = synthetic.n_cube(n_dim)
        assert complex.topology.is_manifold
