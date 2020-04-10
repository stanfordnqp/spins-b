Navigating SPINS Codebase
=========================

SPINS is a large nanophotonic optimization library with several different
structures. Here we provide a brief overview of the sublibraries in SPINS:

- `invdes`: This is the initially-released library that handles the inverse
  design aspect of SPINS.
- `goos`: This is the next generation inverse design library meant to replace
  the older `invdes` library. This library is still under development, though
  you can run many optimizations already with `goos`.
- `goos_sim`: This module contains simulators for `goos`.
- `fdfd_tools` and `fdfd_solvers`: This sublibraries handle setup and running
  finite-difference frequency-domain (FDFD) simulations.
- `gridlock`: This library handles drawing on the Yee grid. The Yee grid is
  a set of offset grids that are used in FDTD and FDFD simulations.

There are a few other small sub-libraries under the `spins` package that perform
tasks specific for the `invdes` library, which we will ignore here.
