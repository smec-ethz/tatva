
<img src="assets/logo-small.png" alt="drawing" width="400"/>

Functional programming and differentiable framework for finite element method (FEM) simulations. `femsolver` is a Python library for finite element method (FEM) simulations. It is built on top of JAX and Equinox, making it easy to use FEM in a differentiable way.

## License

Copyright © 2025 ETH Zurich (Mohit Pundir)
`femsolver` is free software: you can redistribute it and/or modify it under the terms of the GNU Lesser General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
`femsolver` is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along with `femsolver`.  If not, see https://www.gnu.org/licenses/.


## Features

- Functional programming interface for FEM simulations
- Differentiable operations using JAX
- Support for linear, nonlinear, and mixed FEM simulations


## Installation

Clone the repository and install the package with pip:
```bash
pip install path/to/femsolver
```

You can also use pip to install directly from the GitLab repository. Make sure
you have access to the repository and have set up SSH keys for authentication.
```bash
pip install git+ssh://git@gitlab.ethz.ch/compmechmat/research/mohit-pundir/femsolver.git
```

> [!note]
> We strongly recommend to always use a virtual environment. We further
> recommend using [uv](https://docs.astral.sh/uv/).



## Usage


A complete guide on how to use `femsolver` is available in the [course notes](https://gitlab.ethz.ch/compmechmat/teaching/stcm/course-notes).

Some of the examples are available in the `examples/notebooks` directory.

## Roadmap

The roadmap for `femsolver` is to be updated as we progress with the development. Currently, the roadmap is as follows:

- [ ] Add support for Hermite elements for beam analysis (**Mohit**)
- [ ] Add support for shell elements for plate analysis (**Flavio**)

Currently, the roadmap for different applications/examples/use-cases of `femsolver` is as follows:

- [x] Add example for linear elasticity (**Mohit**)
- [x] Add example for nonlinear elasticity (**Mohit**)
- [x] Add example for Dirichlet BCs as constraints (**Mohit**)
- [x] Add example for matrix-free solvers (with Dirichlet BCs) (**Mohit**)
- [x] Add example for contact problems with penalty method (**Mohit**)
- [x] Add example for contact problems with Lagrange multipliers (**Flavio**)
- [ ] Add example for contact problems with augmented Lagrangian method (**Flavio**)
- [ ] Add example for contact problems with Nitsche method (**Flavio**)
- [x] Add example for cohesive fracture problems (**Mohit**)
- [x] Add example for cohesive fracture problems under quasi-static loading (**Mohit**)
- [x] Add example for cohesive fracture problems in dynamics (**Mohit**)
- [x] Add example for thermal-mechanical coupled problems (**Mohit**)
- [ ] Add example for phase-field fracture coupled problems (**Mohit**)


## Dense vs Sparse 

A unique aspect of `femsolver` is that it can handle both dense and sparse matrices. This is achieved by using the library `sparsejac` that allows automatic differentiation of a functional based on a sparsity pattern. This significantly reduces the memory consumption. For more details on how the automatic differentiation can be done using sparsity pattern, please check the link below:

- ![Paper](https://arxiv.org/html/2501.17737v1)</br>
- ![Github: sparsejac](https://github.com/mfschubert/sparsejac)</br>
- ![Github: Sparsediffax, python interface for the paper](https://github.com/gdalle/sparsediffax)</br>

## Profiling

### Time usage profiling

Below we provide the computational time for the assembly of the sparse stiffness matrix for linear elasticity problem. The code is available in `benchmarking/profiling_time.py`.


![Assembly Time](benchmarking/assembly_time_cpu.png)

The time above doesnot account for the compilation time  of the functions. In JAX, the first time a function is called, it is compiled and repeated calls are faster. This compilation time is not included in the time above. The time above is for a single core of a CPU. 



### Memory usage profiling

We use pprof to profile the memory usage. Please follow the instruction on JAX's documentation on profiling  ![Link to documentation](https://docs.jax.dev/en/latest/device_memory_profiling.html). Using `go` and `pprof`.

For 20000 degrees of freedom and a sparse linear elastic framework (`benchmarking/profiling_memory_usage.py`), a total of `15 MB` memory is used on `CPU`. The distribution of memory usage is as follows:

![](benchmarking/profile001.svg)

