"""Gray-Scott Reaction-Diffusion example, on a variety of complexes

This should be trivial to apply to complexes of any dimension, although only 2d complexes are implemented at the moment

"""

import numpy as np

from pycomplex.math import linalg


class ReactionDiffusion(object):
    """
    Various interesting parameter combinations, governing the rate of synthesis
    of u and rate of breakdown of v, respectively.
    Different rate parameters yield very different results, but the majority of
    parameters combination do not yield interesting results at all.
    So if you feel like trying something yourself, start by permuting some existing settings
    """
    params = dict(
        divide_and_conquer  = (0.035, 0.099),   # dividing blobs
        aniogenesis         = (0.040, 0.099),   # a vascular-like structure
        fingerprints        = (0.032, 0.091),   # a fingerprint-like pattern
        holes               = (0.042, 0.101),
        labyrinth           = (0.045, 0.108),   # somehow this configuration does not like closed loops
        chemotaxis          = (0.051, 0.115),   # growing roots?

        # lower parameter values tend to destabilize the patterns
        unstable_blobs      = (0.024, 0.084),
        unstable_labyrinth  = (0.024, 0.079),
        unstable_holes      = (0.022, 0.072),

        # even lower parameters lead to wave-like phenomena
        swimming_medusae    = (0.011, 0.061),
        traveling_waves     = (0.019, 0.069),
        standing_waves      = (0.015, 0.055),
        trippy_chaos        = (0.025, 0.075),
    )

    def __init__(self, complex, key='fingerprints'):
        self.complex = complex

        from examples.diffusion.explicit import Diffusor
        self.diffusor = Diffusor(complex)

        size = self.complex.topology.n_elements[0]

        # this is important for initialization! right initial conditions matter a lot
        self.state = np.zeros((2, size), np.float32)
        self.state[0] = 1
        # add seeds
        self.state[1, np.random.randint(size, size=10)] = 1

        self.coefficients = self.params[key]

        # setting this much higher will have numerical error affect the physical behavior.
        # also, doesnt add much since even on regular grid, on of the diffusions needs a double step
        self.dt = 1
        # Diffusion constants for u and v. Probably best not to touch these. ratio is important
        # higher values result in features that span more elements in the mesh
        ru = 1 / 8
        rv = 0.5 / 8
        self.mu = ru, rv

    def gray_scott_derivatives(self, u, v):
        """the gray-scott equation; calculate the time derivatives, given a state (u,v)"""
        f, g = self.coefficients
        reaction = u * v * v                # reaction rate of u into v; note that the production of v is autocatalytic
        source   = f * (1 - u)              # replenishment of u is proportional to its deviation from one
        sink     = g * v                    # decay of v is proportional to its concentration
        udt = - reaction + source           # time derivative of u
        vdt = + reaction - sink             # time derivative of v
        return udt, vdt                     # return both rates of change

    def integrate(self, derivative):
        """
        forward euler integration of the equations, given their state and time derivative at this point
        the state after a small timestep dt is taken to be the current state plus the time derivative times dt
        this approximation to the differential equations works well as long as dt is 'sufficiently small'
        """
        for s,d in zip(self.state, derivative):
            s += d * self.dt
        for s, mu in zip(self.state, self.mu):
            s[...] = self.diffusor.integrate_explicit(s, self.dt * mu)

    def simulate(self, iterations):
        """Generator function to do the time integration"""
        for i in range(iterations):
            print(i)
            # make 20 timesteps per frame; we dont need to show every one of them,
            # since the change from the one to the next is barely perceptible
            for r in range(20):
                # update the chemical concentrations
                self.integrate(self.gray_scott_derivatives(*self.state))


if __name__ == '__main__':
    from examples.util import save_animation
    import matplotlib.pyplot as plt

    kind = 'sphere'
    if kind == 'sphere':
        from pycomplex import synthetic
        surface = synthetic.icosphere(refinement=6).copy(radius=50)
    if kind == 'letter':
        from examples.subdivision import letter_a
        surface = letter_a.create_letter(5).subdivide_simplicial().as_3()
        surface = surface.copy(vertices=surface.vertices * 40)
        surface = surface.copy(vertices=np.dot(surface.vertices, linalg.power(linalg.orthonormalize(np.random.randn(3, 3)).T, 0.2)))

    if kind == 'regular':
        from pycomplex import synthetic
        if True:
            surface = synthetic.n_cube_grid((256, 256))
        else:
            surface = synthetic.n_cube(2)
            surface = surface.copy(vertices=surface.vertices * 128)
            for i in range(1):
                surface = surface.subdivide_cubical()
        surface = surface.as_22().as_regular()
        tris = surface.subdivide_simplicial()


    assert surface.topology.is_oriented
    print(surface.topology.n_elements)
    if False:
        surface.plot(plot_dual=False, plot_vertices=False)

    path = r'c:\development\examples\reaction_diffusion_7'
    rd = ReactionDiffusion(surface, key='swimming_medusae')

    for i in save_animation(path, frames=200, overwrite=True):

        rd.simulate(1)
        form = rd.state[1]

        # plot the resulting pattern

        if kind == 'sphere':
            surface.as_euclidian().plot_primal_0_form(form, plot_contour=False, shading='gouraud')
        if kind == 'regular':
            form = tris.topology.transfer_operators[0] * form
            tris.as_2().plot_primal_0_form(form, plot_contour=False, shading='gouraud')
        if kind == 'letter':
            surface.plot_primal_0_form(form, plot_contour=False, shading='gouraud')

        plt.axis('off')
