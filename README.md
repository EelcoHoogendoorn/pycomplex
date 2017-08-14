pycomplex
=========

Summary
-------
The name of this library derives from the notion of an (abstract) simplicial complex, and is not related to complex numbers.
The scope of this library is probably best captured with the term DEC (discrete exterior calculus),
although the terms discrete differential geometry, discrete topology, subdivision surfaces and multigrid also apply.

Applications
------------
This library provides useful functionality in a wide range of computational and graphical modelling contexts
- Computational modelling, such as fluid dynamics and electromagnetics
- Surface modelling, subdivision curves, and other geometric manipulations

The examples folder contains a comprehensive set of brief but illustrative use cases.
This includes simple problems illustrating concepts in computational geometry or discrete exterior calculus,
to the implementation of some landmarks papers in these fields.

Aside from serving my own projects and curiosity, it is my aim that the coding this in this package should be sufficiently clean and generic
that it may serve the dissemination of ideas between novices and people working out new ideas alike.

Features
--------
- All functionality is vectorized and efficiently implemented, so useful for real-world problem sizes
- (almost all) functionality is agnostic to the number of dimensions used
- Complete separation of topology and geometry
- Handling of simplicial and cubical complexes
- Handling of different geometries (spherical, Euclidian, and perhaps others)
- Hierarchical subdivision logic, for use in subdivision surfaces and multigrid solvers

Possible novelties
------------------
- Construction of boundary dual and relation to formulation of boundary conditions
- Efficient picking of primal n-simplices using a power diagram constructed over dual-0-forms
- Fundamental-domain interpolation of dual 0-forms; much simpler conceptually than barycentric, and can be efficiently implemented

See also
--------
The discrete exterior calculus components of this package provide a lot of overlapping functionality with <a href="https://github.com/hirani/pydec">pydec</a>,
although the vectorized implementation provided here should scale a lot better

Many of the algorithms implemented here are generalized variants of those initially implemented in <a href="https://github.com/EelcoHoogendoorn/Escheresque">Escheresque</a>,
which will become the first real use-cases of this package.

Todo
----
- Add 3d electrodynamics example
- Multigrid support
    - Add hierarchy class, to manage all transfers across levels
    - Black box multigrid (at least for 0-forms)
    - Multigrid eigen solver
- Implement picking on euclidian simplicial complexes (requires dealing with non-acute simplices)
- Implement hodge-optimized meshes (http://www.geometry.caltech.edu/pubs/MMdGD11.pdf)
- Implement geometric elasticity (http://imechanica.org/files/Geometry_Elasticity.pdf)
- Clean up boundary condition handling in examples
- Replace ugly casting system with some more automagic dynamic dispatch

Speculative todo
----------------
- Cylindrical coordinates and corresponding metric calculations
- Hybrid triangle-quad meshes
- Some basic support for FEM