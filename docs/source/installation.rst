Installation
=================
.. note::

   **Recommended Install**
    CRISPy is currently undergoing rapid developments,
    install from the source (see Option 3) to stay up-to-date with the latest version.

System Requirements
--------------------
CRISPy is compatible with the following environments:
- Python 3.8 or later.
- Dependencies include `NumPy`, `Astropy`

Instructions
-------------

1. **Install from PyPI**:

   The easiest way to install CRISPy is through PyPI. Run the following command in your terminal:

   .. code-block:: bash

       pip install astro-crispy

2. **Install from Source**:
   to install the latest developing version, clone the CRISPy GitHub repository:

   .. code-block:: bash

        git clone https://github.com/mcyc/crispy.git
        cd crispy
        pip install -e .


   To use a specific version using a tag, for example, v1.4.2, run the following after
   the initial install:

   .. code-block:: bash

        git checkout v1.4.2
        git pull

   .. note::
       If you encounter issues with pre-existing versions of dependencies or want to ensure
       that the pinned versions of CRISPy's dependencies are installed, use the following command:

       .. code-block:: bash

           pip install --upgrade --force-reinstall --no-cache-dir -e .

       This command ensures that all dependencies are freshly installed, replacing any older or conflicting versions.

3. **Dependencies**:
   CRISPy will automatically install its dependencies during installation. If any issues occur, install them manually:

   .. code-block:: bash

       pip install numpy astropy spectral-cube pyspeckit


Verification
~~~~~~~~~~~~~
To verify the installation, open a Python interpreter and import CRISPy:

.. code-block:: python

    import crispy
    print(crispy.__version__)

If no errors occur, CRISPy is installed correctly.
