
class Hierarchy(object):
    """Subdivision hierarchy; list of recursive topologies and their transfer operators"""

    def __init__(self, levels, transfers):
        """

        Parameters
        ----------
        levels : list of n_levels Topologies
        transfers : list of n_levels-1 lists of transfer operators for all primal elements

        """
        self.levels = levels
        self.transfers = transfers

    @staticmethod
    def generate(topology, n_levels):
        """

        Parameters
        ----------
        topology : Topology
        n_levels : int

        Returns
        -------
        Hierarchy
        """
        levels = [topology]
        transfer = []
        for i in range(n_levels):
            l, t = topology[-1].subdivide_cubical()
            topology.append(l)
        return Hierarchy(levels, transfer)

    def expand(self, chain, k):
        """Expand a k-chain from the root to tip level

        Parameters
        ----------
        chain : k-chain on the root level

        """
        if not self.levels[0].n_elements[k] == len(chain):
            raise ValueError

        for t in self.transfers:
            chain = t[k] * chain
        return chain