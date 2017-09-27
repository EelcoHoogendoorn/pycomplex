"""Try and implement harmonics on letter mesh.
Nice test of condition number concerns"""

import numpy as np
import numpy_indexed as npi
import scipy.sparse

from pycomplex.math import linalg
from pycomplex.complex.simplicial.euclidian import ComplexTriangularEuclidian3
import matplotlib.pyplot as plt

from examples.subdivision.letter_a import create_letter

letter = create_letter(3).as_23().subdivide_simplicial().as_3()

letter.plot(plot_dual=False)
plt.show()