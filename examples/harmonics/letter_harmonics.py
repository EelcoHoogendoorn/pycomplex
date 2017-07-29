import numpy as np
import numpy_indexed as npi
import scipy.sparse

from pycomplex.math import linalg
from pycomplex.complex.simplicial import ComplexTriangularEuclidian3

from examples.subdivision.letter_a import create_letter

letter = create_letter(3).as_23().to_simplicial().as_3()



seed = letter.topology.chain(0, fill=0, dtype=np.float)
idx = np.argmin(np.linalg.norm(letter.vertices - [2, 2, -3], axis=1))
print(idx)
seed[idx] = 10
geo = letter.geodesic(seed)

# letter.plot_3d(plot_dual=False, backface_culling=True, plot_vertices=False)
letter.plot_primal_0_form(geo)
