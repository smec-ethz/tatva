<div align="center">

<img src="assets/logo-small.png" alt="drawing" width="400"/>

<h3 align="center">Tatva (तत्त्व) : Lego-like building blocks for differentiable FEM</h3>

`tatva` (is a Sanskrit word which means principle or elements of reality). True to its name, `tatva` provide fundamental Lego-like building blocks (elements) which can be used to construct complex finite element method (FEM) simulations. `tatva` is purely written in Python library for FEM simulations and is built on top of JAX ecosystem, making it easy to use FEM in a differentiable way.

</div>

<div align="center">

[![Documentation](https://github.com/smec-ethz/tatva-docs/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/smec-ethz/tatva-docs/actions/workflows/pages/pages-build-deployment)
[![Tests](https://github.com/smec-ethz/tatva/actions/workflows/run_tests.yml/badge.svg)](https://github.com/smec-ethz/tatva/actions/workflows/run_tests.yml)
![PyPI](https://img.shields.io/pypi/v/tatva)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/smec-ethz/tatva)

</div>


<div align="center">

[**Documentation**](https://smec-ethz.github.io/tatva-docs/)
| [**Usage & Examples**](https://smec-ethz.github.io/tatva-docs/examples/linear_elasticity/)
| [**Contributing**](CONTRIBUTING.md)

</div>


## License

`tatva` is distributed under the GNU Lesser General Public License v3.0 or later. See `COPYING` and `COPYING.LESSER` for the complete terms. © 2025 ETH Zurich (SMEC).

## Features

- Energy-based formulation of FEM operators with automatic differentiation via JAX.
- Capability to handle coupled-PDE systems with multi-field variables, KKT conditions, and constraints. 
- Element library covering line, surface, and volume primitives (Line2, Tri3, Quad4, Tet4, Hex8) with consistent JAX-compatible APIs.
- Mesh and Operator abstractions that map, integrate, differentiate, and interpolate fields on arbitrary meshes.
- Automatic handling of stacked multi-field variables through the `tatva.compound` utilities while preserving sparsity patterns.
- MPI parallelism support.

## Installation

Install the current release from PyPI:

```bash
pip install tatva
```

For development work, clone the repository and install it in editable mode (use your preferred virtual environment tool such as `uv` or `venv`):

```bash
git clone https://github.com/smec-ethz/tatva.git
cd tatva
pip install -e .
```

## Documentation

Available at [**smec-ethz.github.io/tatva-docs**](https://smec-ethz.github.io/tatva-docs/). The documentation includes API references, tutorials, and examples to help you get started with `tatva`.

## Usage

Create a mesh, pick an element type, and let `Operator` perform the heavy lifting with JAX arrays:

```python
import jax.numpy as jnp
from tatva.element import Tri3
from tatva.mesh import Mesh
from tatva.operator import Operator

coords = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
elements = jnp.array([[0, 1, 2], [0, 2, 3]])

mesh = Mesh(coords, elements)

op = Operator(mesh, Tri3())
nodal_values = jnp.arange(coords.shape[0], dtype=jnp.float64)

# Integrate a nodal field over the mesh
total = op.integrate(nodal_values)

# Evaluate gradients at all quadrature points
gradients = op.grad(nodal_values)
```

Examples for various applications will be added very soon. They showcase patterns such as
mapping custom kernels, working with compound fields, and sparse assembly helpers.

## Dense vs Sparse vs Matrix-free

A unique aspect of `tatva` is that it can handle construct dense matrices, sparse matrices, and matrix-free operators. `tatva` uses matrix-coloring algorithm and sparse differentiation to construct a sparse matrix. We use our own coloring library ![tatva-coloring](https://github.com/smec-ethz/tatva-coloring) to color a matrix based on sparsity pattern, one can use other coloring libraries such as ![pysparsematrixcolorings](https://github.com/gdalle/pysparsematrixcolorings) for more advanced coloring algorithms. This significantly reduces the memory consumption. For large problems, we can also use matrix-free operators which do not require storing the matrix in memory. Since we have a energy functional, we can make use of `jax.jvp` ti compute the matrix-vector product without explicitly forming the matrix. This is particularly useful for large problems where storing the matrix is not feasible.


## Paper

To know more about `tatva` and how it works please check: ([arXiv link](https://arxiv.org/abs/2602.12365v1))


## 👉 How to contribute

We welcome your help to improve tatva. To prevent wasted effort, you must discuss your idea with the team before you write any code. Please read our full [Contributing Guide](CONTRIBUTING.md) for complete details.

Follow this exact workflow to contribute:

1. **Discuss First:** Open an Issue to explain the problem and propose your numerical approach.
2. **Wait for Approval:** Do not start coding until a maintainer approves your Issue.
3. **Fork the Project:** Create a personal copy of the repository to test your ideas.
4. **Create a Branch:** (`git checkout -b feature/your-feature-name`)
5. **Commit your Changes:** (`git commit -m 'Add your feature'`)
6. **Push to the Branch:** (`git push origin feature/your-feature-name`)
7. **Open a Pull Request:** Submit your PR and link it to the approved Issue.

Don't forget to give the project a star. Thank you!
