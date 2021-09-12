import numpy as np


def stability(mesh, displacement, cog):
    """Perform static stability analysis for a floating object

    Rotation is performed in the x-z plane, around the y-axis
    Gravity is directed along the negative z axis

    Parameters
    ----------
    mesh : ComplexTriangularEuclidian3
        geometry to be analyzed.
        it should represent the displaced volume of water; so should not include modelled internal volumes
    displacement : float
        total displacement; should be some fraction of the volume of the mesh
    cog : array_like, [3], float
        center of gravity, relative to coordinate system of input `mesh`

    Returns
    -------
    angles : ndarray, [steps], float
        Rotation angles relative to neutral
    depth : ndarray, [steps], float
        Displacement of boat along the z axis for each angle
    cob : ndarray, [steps, 3], float
        Center of Bouyancy relative to cog
    """

    # want to rotate around cog
    mesh = mesh.translate(-np.array(cog))

    def solve_depth(mesh, volume: float) -> float:
        def target(d: float) -> float:
            return mesh.translate((0, 0, -d)).clip((0, 0, 0), (0, 0, 1.0)).as_3().volume() - volume
        import scipy.optimize
        return scipy.optimize.bisect(target, *mesh.box[:, -1])

    def stability(mesh, volume: float, r: np.array) -> (float, float):
        mesh = mesh.transform(r)
        depth = solve_depth(mesh, volume)
        cob = mesh.translate((0, 0, -depth)).clip((0, 0, 0), (0, 0, 1.0)).as_3().center_of_mass()
        return depth, cob

    def rotation(a: float) -> np.array:
        c, s = np.cos(a), np.sin(a)
        return [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]

    def sweep(mesh, volume: float, steps: int) -> (np.array, np.array):
        angles = np.linspace(-np.pi, np.pi, steps)
        result = [stability(mesh, volume, rotation(a)) for a in angles]
        depth, cob = zip(*result)
        return angles, np.array(depth), np.array(cob)

    angles, depth, cob = sweep(mesh, displacement, 100)
    return angles, depth, cob


if __name__ == '__main__':

    if True:
        from pycomplex import synthetic
        boat = synthetic.n_cube_grid((1, 1, 1), centering=True).boundary.as_23().subdivide_simplicial().as_3()
        cog = [0, 0, -0.3]
        displacement = 0.2
    else:
        path = r'/Users/eelco/Downloads/hull3.stl'
        from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
        boat = ComplexTriangularEuclidian3.load_STL(path)
        cog = [0, 0, 1.67]  # 2cm duplex
        cog = [0, 0, 1.93]  # 1cm duplex
        displacement = 100

    angles, depth, cob = stability(boat, displacement, cog)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(angles / np.pi * 180, cob[:, 0])
    plt.figure()
    plt.plot(angles / np.pi * 180, depth)
    plt.show()
