import os
from setuptools import find_packages, setup

setup(
    name='pycomplex',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy>=1.9',
        'numpy-indexed',
        'matplotlib',
        'pycosat',
        'cached-property',
        'fastcache',
        'funcsigs'
    ],
    license='LGPL',
    platforms='any',
    zip_safe=False,
)
