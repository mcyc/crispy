CRISPy Documentation
====================

    **CRISPy** (Computational Ridge Identification with SCMS for Python) is a Python library for identify density
    ridges in multi-dimensional space.


Overview
---------------------

CRISPy identify density ridges using the Subspace Constrained Mean Shift (SCMS) algorithm based on the
statistical framework generalized by `Yen-Chi Chen <http://faculty.washington.edu/yenchic/>`_.
This library adapts and extends the method for Python
applications, with focus on scientific and astrophysical usage.

CRISPy library's key features include:

- Efficient implementation of the SCMS algorithm for Python.
- Tools for gridding, pruning, and refining ridge results.
- Flexibility to tune parameters for domain-specific applications.
- Seamless integration with popular formats like `.fits` for input and output.

Citation
---------------------
When publishing with **CRISPy**, please cite the following for the software:

    1. Chen, M. C.-Y., et al. "Velocity-Coherent Filaments in NGC 1333: Evidence for Accretion Flow?"
       ApJ (`2020 <https://ui.adsabs.harvard.edu/abs/2020ApJ...891...84C/abstract>`_).

and the statistical framework:

    2. Chen, Y.-C., et al. "Cosmic Web Reconstruction through Density Ridges: Method and Algorithm."
       MNRAS (`2015 <https://arxiv.org/abs/1501.05303>`_).

    3. Chen, Y.-C., et al. "Generalized Mode and Ridge Estimation."
       (`2014 <https://arxiv.org/abs/1406.1803>`_).

For further reference on the initial development of SCMS, please see:

    4. Ozertem, U., and Erdogmus, D. "Locally Defined Principal Curves and Surfaces."
       JMLR (`2011 <https://jmlr.org>`_).

Navigation
---------------------
To get started quickly, please see the :doc:`Install <installation>` and :doc:`Quick Start <quick_start>` pages.
Use the navigation bar at the top to explore other pages, including,
:doc:`Tutorials <tutorials/index>`, :doc:`Guides <guides>`, and :doc:`API Reference <api/index>`.

.. toctree::
   :hidden:
   :maxdepth: 2

   Install <installation>
   Starting <quick_start>
   guides
   tutorials/index
   API <api/index>
