API Reference
=============

This section contains the complete API documentation for PetaKit5D.

.. toctree::
   :maxdepth: 2

   image_processing
   microscope_data_processing
   utils

Overview
--------

PetaKit5D provides three main modules:

* **image_processing**: General image processing functions (52 functions)
* **microscope_data_processing**: Microscopy-specific utilities (31 functions)
* **utils**: Utility functions (15 functions)

All functions are available through top-level imports for convenience::

    from petakit5d import read_tiff, filter_gauss_3d, decon_psf2otf
