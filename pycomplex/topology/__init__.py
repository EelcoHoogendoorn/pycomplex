"""Discrete topology module

"""
import numpy as np

# dtypes enforced for indices referring to elements;
# 16 bits is too few for many applications, but 32 should suffice for almost all
# these types are used globally throughout the package; changing them here should change them everywhere
index_dtype = np.int32
sign_dtype = np.int8


class ManifoldException(Exception):
    pass
