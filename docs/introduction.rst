SPINS-B
=======

SPINS-B is the open source version of `SPINS <http://techfinder.stanford.edu/technologies/S18-012_inverse-design-software-for>`_,
a framework for gradient-based (adjoint) photonic optimization developed over
the past decade at Jelena Vuckovic's `Nanoscale and Quantum Photonics Lab <http://nqp.stanford.edu>`_
at Stanford University. For commercial use, the full version can be licensed
through the `Stanford Office of Technology and Licensing <http://techfinder.stanford.edu/technologies/S18-012_inverse-design-software-for>`_ (see FAQ).

The overall architecture is explained in our paper `Nanophotonic Inverse Design with SPINS: Software Architecture and Practical Considerations <https://arxiv.org/abs/1910.04829>`_. 


Features
--------
- Gradient-based (adjoint) optimization of photonic devices
- 2D and 3D device optimization using finite-difference frequency-domain (FDFD)
- Support for custom objective functions, sources, and optimization methods
- Automatically save design methodology and all hyperparameters used in optimization for reproducibility

Overview
--------
Traditional nanophotonic design typically relies on parameter sweeps, which are
expensive both in terms of computation power and time, and restrictive in their
parameter space. Likewise, completely blackbox optimization algorithms, such
as particle swarm and genetic algorithms, are also highly inefficient. In both
these cases, the computational costs limit the degrees of the freedom of the
design to be quite small. In contrast, by
leveraging gradient-based optimization methods, our nanophotonic inverse design
algorithms can efficiently optimize structures with tens of thousands of degrees
of freedom. This enables the algorithms to explore a much larger space of
structures and therefore design devices with higher efficiencies, smaller
footprint, and novel functionalities.


Requirements
------------
- Python 3.5+
- Some version of BLAS (e.g. OpenBLAS, ATLAS, Intel MKL)
- `Maxwell solver <http://github.com/stanfordnqp/maxwell-b>`_ for 3D simulations

Recommendations
---------------
- We recommend using `virtual environments <https://docs.python.org/3.6/tutorial/venv.html>`_
  to isolate installation from the rest of the system.
- If using OpenBLAS, we recommend setting the number of OpenBLAS threads
  (:code:`OPENBLAS_NUM_THREADS` flag) to 1 as SPINS-B leverages parallelism itself.

Installation
------------
Simply clone the SPINS-B repository and run :code:`pip`:

.. code:: bash

   $ pip3 install ./spins-b

Getting Started
---------------
See the grating coupler optimization example and the wavelength demultiplexer
example in the :code:`examples` folder. The grating coupler example covers
setting up, running, and resuming a 2D optimization. The wavelength
demultiplexer example covers setting up and running a 3D optimization as well
as various ways of processing the optimization logs.

More documentation is forthcoming.

General Concepts
----------------
- **Optimization plan**: The optimization plan defines all the photonic
  optimization problem (i.e. simulation region and desired objective) as well
  as the sequence of optimization steps to achieve that objective. You define
  an optimization plan which is then executed by SPINS-B. Doing so enables
  you to have an exact record of all the parameters used to design a device
  as well as the ability to resume optimization if the optimization fails
  midway.
- **Simulation space**: The simulation space defines the simulation region
  as well as the design region (see below).
- **Design area and design region**: The design region is the region of the
  permittivity distribution that is allowed to vary during the optimization.
  The design region is defined as the difference between two permittivity
  distributions: Where the difference is non-zero corresponds to the design
  region. Since most photonic devices are fabricated using top-down lithography,
  SPINS-B by default (this can be changed) assumes that the permittivity
  distribution along the z-axis is the same, and hence we speak of a
  *design area*.
- **Parametrization**: The parametrization defines how to describe the
  permittivity of the design area. The simplest parametrization is to simply
  describe the value of each pixel on the Yee grid.
- **Monitors**: Monitors are used to log data during the optimization process.
  *Simple monitors* simply record the value of a function whereas
  *field monitors* post-processes vector field data and can select out a
  particular plane to save data.
- **Transformation**: Optimization in SPINS-B actually consists of a sequence
  of optimization problems. Each optimization is described by a *transformation*
  (because they transform the parametrization from one to another).

FAQ
---

What's different between SPINS-B and SPINS?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SPINS is a fully-featured optimization design suite available for commercial
use. It is a superset of SPINS-B and includes the ability to design devices
without writing any code with user-friendly interfaces and to apply precise
fabrication constraints (minimum gap and curvature constraints). All devices
shown in our published work rely on capabilities found in the fully-featured
SPINS.

How are structures simulated?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SPINS-B uses the finite difference frequency domain (FDFD) simulation method.
This choice was made because in many photonic device designs, we are concerned
with device operation in a small bandwidth at particular frequencies. The
FDFD method is often faster than the more widely used finite difference time
domain (FDTD) method in these cases.

SPINS-B can use both a CPU-based solver or the GPU-accelerated Maxwell FDFD
solver. For 2D simulations, we recommend using a direct matrix CPU-based
solver ("local_direct") because it is faster. 3D simulations require too much
memory and an iterative solver must be used. We recommend the GPU-accelerated
MaxwellFDFD solver ("maxwell_cg") in this case.


Publications
------------
Any publications resulting from the use of this software should acknowledge
SPINS-B and cite the following papers:

For general device optimization:

- Su et al. Nanophotonic Inverse Design with SPINS: Software Architecture and Practical Considerations. *arXiv:1910.04829* (2019).

For grating coupler optimization:

- Su et al. Fully-automated optimization of grating couplers. *Opt. Express* (2018).
- Sapra et al. Inverse design and demonstration of broadband grating couplers.
  *IEEE J. Sel. Quant. Elec.* (2019).