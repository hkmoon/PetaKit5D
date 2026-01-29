Installation
============

Basic Installation
------------------

Install PetaKit5D using pip::

    pip install petakit5d

With Optional Dependencies
--------------------------

For Zarr support::

    pip install petakit5d[zarr]

For development::

    pip install petakit5d[dev]

For documentation building::

    pip install petakit5d[docs]

For everything::

    pip install petakit5d[all]

From Source
-----------

Clone the repository and install in development mode::

    git clone https://github.com/hkmoon/PetaKit5D.git
    cd PetaKit5D
    pip install -e .[dev]

Requirements
------------

* Python >= 3.8
* numpy >= 1.20.0
* scipy >= 1.7.0
* tifffile >= 2021.0.0
* scikit-image >= 0.18.0

Optional dependencies:

* zarr >= 2.10.0 (for Zarr support)
* numcodecs >= 0.9.0 (for Zarr compression)

Troubleshooting
---------------

If you encounter issues with installation:

1. Make sure your Python version is 3.8 or higher::

    python --version

2. Update pip to the latest version::

    pip install --upgrade pip

3. If zarr installation fails, install it separately::

    pip install zarr numcodecs
    pip install petakit5d
