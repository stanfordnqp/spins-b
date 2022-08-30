.. image:: https://travis-ci.com/stanfordnqp/spins-b.svg?branch=master
    :target: https://travis-ci.com/stanfordnqp/spins-b
    
.. image:: https://codecov.io/gh/stanfordnqp/spins-b/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/stanfordnqp/spins-b
    
THIS FORK: adding black-box optimization in Spins-B
===================================================
Spins-B is probably the most well known code for photonics inverse design
in the community. Therefore we  chose to do our work on gradient-free optimization inside it. The present fork is aimed at gathering branches which include gradient-free optimization.

Spins-b (as forked in the present repository)
is based on optimization as follows:
1- random initialization
2- optimization by BFGS: more precisely this is L-BFGS-B
3- discretization enforced by sigmoid transformation.
In the case of grating, there is an additional optimization
step 4- by SLSQP, using a parametrization. Here the optimization is continuous
(the parametrization is continuous) but the design is discrete (there are only two permittivities).

We add one more step termed NG, at the end of 3:
- Lengler's method (https://dl.acm.org/doi/10.1145/3321707.3321733)
- equipped with a smoothing operator 

The smoothing operator is detailed here:
(https://github.com/facebookresearch/nevergrad/blob/8403d6c9659f40fec2a3cf7f474b3d8610f0f2e4/nevergrad/optimization/optimizerlib.py#L388).

We are *very* grateful to Spins-B for providing us with this great code, central for our experiments.

For example https://github.com/teytaud/spins-b/tree/patch-1 contains code for running Lengler+smoothing directly in Spins-B for the Bend90 case.

Our results
===========
1+2+3+NG is better than 1+2+3 because the discretization by our discrete optimization methods works better than enforcing discretization through sigmoids.
NG alone (as opposed to 1+2+3+NG) outperforms numerically 1+2+3 in some cases, in particular for large budgets. However, without the initial BFGS step, NG sometimes provides designs which are not smooth enough: the initial point provided by L-BFGS-B as included in Spins-B is essential for ensuring a good smoothness, which is better for buildability.

Discussing with us
==================
Our code uses Nevergrad (https://github.com/facebookresearch/nevergrad)
We are intensive Nevergrad users and we are happy to chat in https://www.facebook.com/groups/nevergradusers/


SPINS-B
=======

SPINS-B is the open source version of `SPINS <https://stanford.resoluteinnovation.com/technologies/S18-012_spins-inverse-design-software-for>`_,
a framework for gradient-based (adjoint) photonic optimization developed over
the past decade at Jelena Vuckovic's `Nanoscale and Quantum Photonics Lab <http://nqp.stanford.edu>`_
at Stanford University. The full version can be licensed
through the `Stanford Office of Technology and Licensing <http://techfinder.stanford.edu/technologies/S18-012_inverse-design-software-for>`_ (see FAQ).

The overall architecture is explained in our paper `Nanophotonic Inverse Design with SPINS: Software Architecture and Practical Considerations <https://arxiv.org/abs/1910.04829>`_. 

Documentation
-------------
`Documentation <http://spins-b.readthedocs.io>`_ is continually updated over time.

Features
--------
- Gradient-based (adjoint) optimization of photonic devices
- 2D and 3D device optimization using finite-difference frequency-domain (FDFD)
- Support for custom objective functions, sources, and optimization methods
- Automatically save design methodology and all hyperparameters used in optimization for reproducibility

Upcoming Features
-----------------
We are protoyping the next version of SPINS-B. This version of SPINS-B will support these new features:

- Co-optimization of multiple device regions simulataneously
- Integration with FDTD and other electromagnetic solvers
- Easier to use and extend

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
