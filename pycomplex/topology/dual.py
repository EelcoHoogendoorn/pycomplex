import numpy as np
import numpy_indexed as npi
from cached_property import cached_property

from pycomplex.topology.topology import BaseTopology2


class Dual(object):

    def __init__(self, primal):
        self.primal = primal
        self.boundary = primal.boundary
        self.dual = primal.dual()


    def __getitem__(self, item):
        """alias for matrix?"""
        return self.dual[item]

    def form(self, n):
        """allocate a dual n-form. This is a block-vector"""
        bn = self.boundary.n_elements
        i = self.primal.n_elements[n]
        i = i - p
        d = 0
        # FIXME
        return