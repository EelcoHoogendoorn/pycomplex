
import numpy as np

from examples.diffusion.explicit import Diffusor

class MultiDiffusor(object):
    """Solve diffusion problems with multigrid logic and explicit integration"""

    def __init__(self, hierarchy):
        self.hierarchy = hierarchy
        self.diffusors = [Diffusor(l) for l in self.hierarchy]

        self.eigenvalues = np.array([d.largest_eigenvalue for d in self.diffusors]).flatten()

    def integrate_explicit(self, field, dt):
        """Integrate diffusion equation over a timestep dt

        Parameters
        ----------
        field : ndarray, [n_vertices], float
            primal 0-form
        dt : float
            timestep

        Returns
        -------
        field : ndarray, [n_vertices], float
            diffused primal 0-form

        """
        distance = self.largest_eigenvalue * dt
        steps = int(np.ceil(distance))
        fraction = distance / steps
        for i in range(steps):
            field = self.explicit_step(field, fraction)
        return field



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    kind = 'sphere'

    if kind == 'sphere':
        from pycomplex import synthetic
        complex = synthetic.icosahedron()
        hierarchy = [complex]
        for l in range(7):
            complex = complex.subdivide_loop()
            hierarchy.append(complex)



    if True:
        field = np.random.rand(complex.topology.n_elements[0])
    else:
        print(complex.box)
        field = complex.topology.chain(0, dtype=np.float)
        idx = 0
        idx = np.argmin(np.linalg.norm(complex.vertices - [0, 0], axis=1))
        field[idx] = 1

    diffusor = MultiDiffusor(hierarchy)
    print(diffusor.eigenvalues)

    field = diffusor.integrate_explicit_sigma(field, 1.5)
    print(field.min(), field.max())

    if kind == 'sphere':
        complex = complex.as_euclidian()
        complex.plot_primal_0_form(field)
    if kind == 'regular':
        tris = complex.subdivide_simplicial()
        field = tris.topology.transfer_operators[0] * field
        tris.as_2().plot_primal_0_form(field)
    if kind == 'letter':
        complex.plot_primal_0_form(field, plot_contour=False)

    plt.show()