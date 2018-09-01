[![Build Status](https://travis-ci.org/EelcoHoogendoorn/pycomplex.svg?branch=master)](https://travis-ci.org/EelcoHoogendoorn/pycomplex)

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
- Computational modelling, such as fluid dynamics, electromagnetics or elasticity
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
- Handling of different geometries (spherical, Euclidian, and potentially others)
- Handling of boundary topology, geometry and boundary conditions in a unified manner
- Hierarchical subdivision logic, for use in subdivision surfaces and multigrid solvers

Possible novelties
------------------
- Construction of boundary dual and relation to formulation of boundary conditions
- Picking of primal/dual power diagram elements using only a single closest-point query
- Fundamental-domain interpolation of dual 0-forms; much simpler conceptually than barycentric, and can be efficiently implemented
- Semi-structured multigrid transfer operator


See also
--------
The discrete exterior calculus components of this package provide a lot of overlapping functionality with <a href="https://github.com/hirani/pydec">pydec</a>,
although the vectorized implementation provided here should scale a lot better

Many of the algorithms implemented here are generalized variants of those initially implemented in <a href="https://github.com/EelcoHoogendoorn/Escheresque">Escheresque</a>,
which will become the first real use-cases of this package.

Todo
----
- Stencil-based cochain complexes on regular nd-grids; possibly with efficient gpu kernels
- Add 3d electrodynamics example; preferably also by means of implementing a complex with spacetime-metric
- Multigrid support
    - Black box multigrid on simplicial and regular meshes, for all k-forms
- Replace ugly casting system with some more automagic dynamic dispatch

Speculative todo
----------------
- Cylindrical coordinates and corresponding metric calculations
- Hybrid triangle-quad meshes
- Try to implement something along the lines of these references
    https://pdfs.semanticscholar.org/f3b2/532c517e1e6efaff90e7fe69d2b9e8ff75bc.pdf
    https://arxiv.org/pdf/0804.0279.pdf

Development
-----------
Pycomplex currently targets python 3 primarily. The source is mostly python 2 compliant and would be easy to make so if the need arose, but compatibility is currently not actively maintained for convenience reasons.
Pycomplex has been successfully (but not systematically) tested on all major platforms

Testing
-------
To invoke the tests from the command line, run pytest --show_plot=False
Many tests have a visual component that will be suppressed by using this flag


