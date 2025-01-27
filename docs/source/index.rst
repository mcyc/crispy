CRISPy Documentation
====================

    **CRISPy** (Computational Ridge Identification with SCMS for Python) is a Python library for identifying density
    ridges in multidimensional space.


Overview
---------------------

**CRISPy** uses the Subspace Constrained Mean Shift (SCMS) algorithm to identify density ridges
(e.g., filaments) in multidimensional data, designed with a focus on scientific and astrophysical usage in
2D and 3D. CRISPy's main features include:

- Efficient implementation of the SCMS algorithm for Python.
- Tools for gridding, pruning, and refining ridge results.
- Flexibility to tune parameters for domain-specific applications.
- Seamless integration with popular formats like `.fits` for input and output.

Citation
---------------------
When publishing with **CRISPy**, please cite the following for the software:

    1. Chen, M. C.-Y., et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?"
       ApJ (`2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...891...84C/abstract>`_).

and the following for the statistical framework:

    2. Chen, Y.-C., et al. "Generalized Mode and Ridge Estimation."
       (`2014 <https://arxiv.org/abs/1406.1803>`_).

Navigation
---------------------
To get started quickly, please see the :doc:`Install <installation>` and :doc:`Quick Start <quick_start>` pages.
There is also a navigation bar at the top for general exploration, including,
:doc:`Tutorials <tutorials/index>`, :doc:`Guides <guides>`, and :doc:`API Reference <api/index>`.

.. toctree::
   :hidden:
   :maxdepth: 2

   Install <installation>
   Starting <quick_start>
   guides
   tutorials/index
   API <api/index>