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
- Surface modelling, subdivision curves, surface smoothing

Features
--------
- All functionality is vectorized and efficiently implemented, so useful for real-world problem sizes
- Complete separation of topology and geometry
- Handling of different geometries (spherical, euclidian, and perhaps others)
- Handling of simplicial and cubical complexes
- Hierarchical subdivision logic, for use in subdivision surfaces and multigrid solvers

See also
--------
The discrete exterior calculus components of this package provide a lot of overlapping functionality with pydec,
although the vectorized implementation provided here should scale a lot better

Todo
----
- Build up interpolation and restriction multigrid transfer operators over different topologies
- Fix cubical meshes for n > 3; Fully n-dim cube subdivision; Correct relative orientation calculations for higher dims
- Fix simplicial meshes for n > 3
- Black box multigrid
- Minres with constraints solver

Speculative todo
----------------
- Cylindrical coordinates and corresponding metric calculations
- Hybrid triangle-quad meshes
- Some basic support for FEM