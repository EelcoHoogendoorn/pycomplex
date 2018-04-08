import os
from setuptools import find_packages, setup

setup(
    name='pycomplex',
    version=os.environ['PKG_VERSION'],
    packages=find_packages(),
    license='LGPL',
    # long_description=open('README.md').read(),
    platforms='any',
    zip_safe=False,
)
