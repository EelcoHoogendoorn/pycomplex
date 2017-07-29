"""Gray-Scott Reaction-Diffusion example


"""

import numpy as np


class ReactionDiffusion(object):

    # Diffusion constants for u and v. Probably best not to touch these
##    ru = 0.2*1.5
##    rv = 0.1*1.5
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

    def __init__(self, complex, key='swimming_medusae'):
        self.complex = complex

        self.dt = 0.015

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
        """Laplacian mapping from"""
        import scipy.sparse
        T01 = self.complex.topology.matrices[0]
        grad = T01.T
        div = T01

        D1P1 = scipy.sparse.diags(self.complex.D1P1)
        P0D2 = scipy.sparse.diags(self.complex.P0D2)
        # construct our laplacian
        laplacian = div * D1P1 * grad
        # solve for some eigenvectors
        # w, v = scipy.sparse.linalg.eigsh(laplacian, M=sparse_diag(mass), which='SA', k=20)

        s, V = scipy.sparse.linalg.eigsh(laplacian, mass=P0D2, k=1, which='LR', tol=1e-5)
        self.largest_eigenvalue = s
        return P0D2 * laplacian

    def diffuse(self, state, mu):
##        return self.complex.diffuse(state) * (mu / -self.complex.largest * 3)
        return self.laplacian * (state * (-mu * 2))

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
        """
        generator function to do the time integration, to be used inside the animation
        rather than computing all the image frames before starting the animation,
        the frames are computed 'on demand' by this function, returning/yielding
        the image frames one by one
        """
        #repeat 'frames' times
        for i in range(iterations):
            #make 20 timesteps per frame; we dont need to show every one of them,
            #since the change from the one to the next is barely perceptible
            for r in range(20):
                #update the chemical concentrations
                self.integrate(self.gray_scott_derivatives(*self.state))

            #every 20 iterations, yield output
            #the v field is what we yield to be plotted
            #we might as well plot u, as it visualizes the dynamics just as well
##            yield v


if __name__ == '__main__':
    from pycomplex import synthetic
    sphere = synthetic.icosphere(refinement=5)
    sphere.metric()

    rd = ReactionDiffusion(sphere)
    print('starting sim')
    rd.simulate(1)
    print('done with sim')

    # plot a spherical harmonic
    sphere.as_euclidian().plot_primal_0_form(rd.state[1])
