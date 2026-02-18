Welcome to PetaKit5D Documentation
====================================

PetaKit5D is a Python library for petabyte-scale 5D microscopy image processing, 
providing comprehensive tools for image filtering, deconvolution, stitching, and analysis.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index
   migration

Features
--------

* Complete image processing suite (52 functions)
* Microscopy data processing (31 functions)  
* Deconvolution utilities
* Tile stitching and feather blending
* MIP generation
* TIFF and Zarr I/O support
* Production-ready with 99.8% test coverage

Installation
------------

Install PetaKit5D using pip::

    pip install petakit5d

Or with optional dependencies::

    pip install petakit5d[zarr]  # With Zarr support
    pip install petakit5d[all]   # With all optional dependencies

Quick Start
-----------

.. code-block:: python

    from petakit5d import read_tiff, filter_gauss_3d, write_tiff

    # Load data
    image = read_tiff('data.tif')

    # Apply Gaussian filter
    filtered = filter_gauss_3d(image, sigma=2.0)

    # Save result
    write_tiff('output.tif', filtered)

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
