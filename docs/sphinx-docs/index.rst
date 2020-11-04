.. Farad documentation master file, created by
   sphinx-quickstart on Wed Nov  4 13:12:54 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Farad's documentation!
=================================

The forward and reverse automatic differentiation (Farad) package provides the ability
to automatically calculate derivatives to machine precision. This is extremely useful
for applications ranging from scientific computing with differential equations to deep
learning algorithms. This technique is especially useful when the overhead of symbolic 
derivatives is infeasible or the accuracy numerical differentiation is insufficient.

Farad uses dual numbers connected in a computational graph to sequentially compute
derivatives in the forward automatic differentiation mode. This package is also capable
of performing reverse automatic differentiation, with several examples of several use-cases
of the library for solving problems requiring accurate derivatives.

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
