import numpy as np
import numpy_indexed as npi
import scipy.sparse

from pycomplex.math import linalg
from pycomplex.complex.simplicial import ComplexTriangularEuclidian3

from examples.subdivision.letter_a import create_letter

letter = create_letter(3).as_23().subdivide_simplicial().as_3()

