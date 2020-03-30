Installation
============

Requirements
------------
- Python 3.6+
- Installation of a supported electromagnetic solver.


Installation
------------
Simply download and install SPINS with `pip`, though we recommend using
`virtual environments <https://docs.python.org/3.6/tutorial/venv.html>`_ to
isolate your installation from your system.

:command:`$ git clone http://github.com/stanfordnqp/spins-b`

:command:`$ pip install ./spins-b`

Solvers
-------
SPINS is a framework for encoding and running optimization for electromagnetics.
Although SPINS comes packaged with a finite-difference frequency-domain solver
(FDFD) solver, you have a choice to install other simulators in the backend.
The packaged FDFD solver is only efficient for small problems, so it is
recommended to install another simulator for large 2D problems or 3D problems.
You must follow the installation procedure for the apprioriate simulator before
using Goos. Listed below are the currently supported simulators.

FDFD Local Matrix Solver
~~~~~~~~~~~~~~~~~~~~~~~~
This is a CPU-based solver that comes packaged with SPINS. This solver simply
sets up the appropriate matrix equation and runs a direct matrix solve using
BLAS to solve. It is therefore fast and efficient for small 2D problems but is
slow for large 2D or 3D problems.

There are no additional required installation steps to use this solver but we
recommend installing UMFPACK and using either ATLAS, OpenBLAS, or Intel MKL.
We find that UMFPACK runs orders of magnitude faster than SuperLU, but UMFPACK
will limit your simulation size to 4 GB.

On Ubuntu, simply install both SWIG and libsuitesparse:

:command:`$ sudo apt install libsuitesparse-dev swig`


Then install the Python package for UMFPACK:

:command:`$ pip install scikit-umfpack`

Maxwell-B
~~~~~~~~~
`Maxwell-B <https://github.com/stanfordnqp/maxwell-b>`_ is a multi-GPU
finite-difference frequency-domain (FDFD) solver. As a frequency domain solver,
it is efficient for problems where you care only about a few wavelengths. The
Maxwell-B solver must be installed on a machine with at least one NVIDIA GPU.
See the `README` file under the Maxwell source folder for details.

MEEP
~~~~
`MEEP <https://github.com/NanoComp/meep>`_ is an open-source finite-difference
time-domain (FDTD) solver. As a time-domain solver, it can simulate the
frequency response across a broad range of frequencies with a single
simulation. MEEP can be parallelized across multiple CPUs using MPI.

As of this writing, install MEEP using the `nightly build <https://meep.readthedocs.io/en/latest/Installation/#nightly-builds>`_
as the main release contains a bug that was only recently fixed.

Custom
~~~~~~
If the above choices are not to your liking, you may choose to define your own
solver.
