************
Tatva (तत्त्व)
************

Lego-like building blocks for FEM
=================================

``tatva`` is a Sanskrit word meaning *principle* or *elements of reality*.
True to its name, ``tatva`` provides fundamental Lego-like building blocks
(elements) which can be used to construct complex finite element method (FEM)
simulations. ``tatva`` is a pure Python library for FEM simulations and is
built on top of JAX and Equinox, making it easy to use FEM in a differentiable
way.


Features
========

* Energy-based formulation of FEM operators with automatic differentiation via JAX.
* Capability to handle coupled-PDE systems with multi-field variables, KKT conditions, and constraints.
* Element library covering line, surface, and volume primitives (Line2, Tri3, Quad4, Tet4, Hex8) with consistent JAX-compatible APIs.
* Mesh and Operator abstractions that map, integrate, differentiate, and interpolate fields on arbitrary meshes.
* Automatic handling of stacked multi-field variables through the tatva.compound utilities while preserving sparsity patterns.


.. toctree::
   :maxdepth: 2
   :caption: Content

   about
   getting_started
   api/modules