"""Linear system specialized to the stencil case

Remains to be seen how much of this can be unified into a more abstract base class;
dont worry about it yet, can unify things when we know what they look like
"""


class Block(object):


class System(object):

    def __init__(self, complex, block):
        self.complex = complex
        self.block = block

    def canonical(self, complex):
        """Formulate full cochain complex"""