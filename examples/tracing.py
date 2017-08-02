"""Once picking on n-simplices works we should be able to create a raytracer that visualizes the hexacosichoron

Moving forward of a ray can be described as a rotation matrix.
Just generate a grid of such rotations descibing a perspective projection,
and return colors based on proximity to primal/dual edges or somesuch
"""

from pycomplex import synthetic

space = synthetic.hexacosichoron()


