"""Gray-Scott Reaction-Diffusion example


"""

import numpy as np


class ReactionDiffusion(object):

    # Diffusion constants for u and v. Probably best not to touch these
    ru = 1
    rv = 0.5

    """
    Various interesting parameter combinations, governing the rate of synthesis
    of u and rate of breakdown of v, respectively.
    Different rate parameters yield very different results, but the majority of
    parameters combination do not yield interesting results at all.
    So if you feel like trying something yourself, start by permuting some existing settings
    """
    params = dict(
        divide_and_conquer  = (0.035, 0.099),   #dividing blobs
        aniogenesis         = (0.040, 0.099),   #a vascular-like structure
        fingerprints        = (0.032, 0.091),   #a fingerprint-like pattern
        holes               = (0.042, 0.101),
        labyrinth           = (0.045, 0.108),   #somehow this configuration does not like closed loops
        chemotaxis          = (0.051, 0.115),   #growing roots?

        #lower parameter values tend to destabilize the patterns
        unstable_blobs      = (0.024, 0.084),
        unstable_labyrinth  = (0.024, 0.079),
        unstable_holes      = (0.022, 0.072),

        #even lower parameters lead to wave-like phenomena
        swimming_medusae    = (0.011, 0.061),
        traveling_waves     = (0.019, 0.069),
        standing_waves      = (0.015, 0.055),
        trippy_chaos        = (0.025, 0.075),
    )

    def __init__(self, complex, key='fingerprints'):
        self.complex = complex

        self.dt = 1 # timestep is limited by diffusion, which is constrained by eigenvalue

        # this is important for initialization! right initial conditions matter a lot
        self.state   = np.zeros((2, self.size), np.float)
        self.state[0] = 1
        # add seeds
        self.state[1,np.random.randint(self.size, size=10)] = 1

        self.coefficients = self.params[key]

        self.laplacian = self.vertex_laplacian()

    @property
    def size(self):
        return self.complex.topology.n_elements[0]

    def vertex_laplacian(self):
        """Laplacian mapping from primal 0 form to primal 0 form"""
        import scipy.sparse
        complex = self.complex
        T01 = complex.topology.matrices[0]
        grad = T01.T
        div = T01

        D1P1 = scipy.sparse.diags(complex.D1P1)
        D2P0 = scipy.sparse.diags(complex.D2P0)
        P0D2 = scipy.sparse.diags(complex.P0D2)

        # construct our laplacian
        laplacian = div * D1P1 * grad

        largest_eigenvalue = scipy.sparse.linalg.eigsh(laplacian, M=D2P0, k=1, which='LM', tol=1e-6,
                                                       return_eigenvectors=False)

        self.largest_eigenvalue = largest_eigenvalue
        print('eig', largest_eigenvalue)
        return P0D2 * laplacian

    def diffuse(self, state, mu):
        return self.laplacian * (state * (-mu / self.largest_eigenvalue))

    def gray_scott_derivatives(self, u, v):
        """the gray-scott equation; calculate the time derivatives, given a state (u,v)"""
        f, g = self.coefficients
        reaction = u * v * v                        # reaction rate of u into v; note that the production of v is autocatalytic
        source   = f * (1 - u)                      # replenishment of u is proportional to its deviation from one
        sink     = g * v                            # decay of v is proportional to its concentration
        udt = self.diffuse(u, self.ru) - reaction + source  # time derivative of u
        vdt = self.diffuse(v, self.rv) + reaction - sink    # time derivative of v
        return udt, vdt                             # return both rates of change

    def integrate(self, derivative):
        """
        forward euler integration of the equations, giveen their state and time derivative at this point
        the state after a small timestep dt is taken to be the current state plus the time derivative times dt
        this approximation to the differential equations works well as long as dt is 'sufficiently small'
        """
        for s,d in zip(self.state, derivative):
            s += d * self.dt

    def simulate(self, iterations):
        """Generator function to do the time integration"""
        for i in range(iterations):
            # make 20 timesteps per frame; we dont need to show every one of them,
            # since the change from the one to the next is barely perceptible
            for r in range(20):
                # update the chemical concentrations
                self.integrate(self.gray_scott_derivatives(*self.state))


if __name__ == '__main__':
    kind = 'letter'
    if kind == 'sphere':
        from pycomplex import synthetic
        surface = synthetic.icosphere(refinement=5)
        surface.metric()
    if kind == 'letter':
        from examples.subdivision import letter_a
        surface = letter_a.create_letter(4).to_simplicial().as_3()
        surface.vertices *= 10
        surface.metric()
    if kind == 'regular':
        from pycomplex import synthetic
        surface = synthetic.n_cube_grid((128, 128)).as_22().as_regular()
        surface.metric()

    assert surface.topology.is_oriented
    print(surface.topology.n_elements)
    if False:
        surface.plot(plot_dual=False, plot_vertices=False)

    rd = ReactionDiffusion(surface)
    print('starting sim')
    rd.simulate(50)
    print('done with sim')

    form = rd.state[0]

    # plot the resulting pattern

    if kind == 'sphere':
        surface = surface.as_euclidian()
    if kind == 'regular':
        surface = surface.to_simplicial().as_2()
        form = surface.topology.transfer_operators[0] * form

    surface.plot_primal_0_form(form, plot_contour=False)
