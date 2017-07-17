
import pycosat

import numpy as np
from cached_property import cached_property

from pycomplex.topology import ManifoldException
from pycomplex.topology.base import BaseTopology


class Topology1(BaseTopology):

    @cached_property
    def n_vertices(self):
        return self.n_elements[0]

    @cached_property
    def n_edges(self):
        return self.n_elements[1]


class BaseTopology2(BaseTopology):

    @cached_property
    def n_vertices(self):
        return self.n_elements[0]

    @cached_property
    def n_edges(self):
        return self.n_elements[1]

    @cached_property
    def n_faces(self):
        return self.n_elements[2]

    def check_manifold(self):
        """
        Raises
        ------
        ManifoldException
            If the topology does not describe a valid manifold
        """
        # all edges must be well-formed, having a single start and end
        if np.any(np.abs(self.T01).sum(axis=0) != 2) or np.any(self.T01.sum(axis=0) != 0):
            raise ManifoldException('Every 1-simplex must be defined in terms of a single start and end 0-simplex')

        # better written in terms of D01 I suppose; dual of the above check, allowing for open boundaries
        if np.any(np.abs(self.T12).sum(axis=1) > 2):
            raise ManifoldException('Every n-1-simplex must be adjacent to at most 2 n-simplices')

        # do this for all subsequent matrices
        if (self.T01 * self.T12).nnz != 0:
            raise ManifoldException('Every n-simplex must be defined in terms of a closed boundary of n-1-simplices')

        if not np.all(self.regions_per_vertex == 1):
            raise ManifoldException('The neighborhood of every 0-simplex should be isomorphic to an n-disc')


