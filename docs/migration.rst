Migration from MATLAB
======================

This guide helps MATLAB users transition to the Python version of PetaKit5D.

Key Differences
---------------

**Indexing**

MATLAB uses 1-based indexing, Python uses 0-based::

    % MATLAB
    image(1, 1, 1)  % First element
    
    # Python
    image[0, 0, 0]  # First element

**Array Order**

MATLAB arrays are column-major (Fortran order), NumPy arrays are row-major (C order)::

    % MATLAB: (z, y, x) or (height, width, depth)
    size(image)  % Returns [nz, ny, nx]
    
    # Python: same convention maintained
    image.shape  # Returns (nz, ny, nx)

**Function Names**

Most functions maintain similar names, converted to snake_case::

    % MATLAB
    filterGauss3D(image, sigma)
    
    # Python
    filter_gauss_3d(image, sigma)

Function Mapping
----------------

File I/O
~~~~~~~~

::

    % MATLAB
    readtiff(filename)
    writetiff(filename, data)
    
    # Python
    read_tiff(filename)
    write_tiff(filename, data)

Filtering
~~~~~~~~~

::

    % MATLAB
    filterGauss3D(image, sigma)
    bilateralFilter(image, sigma_spatial, sigma_intensity)
    
    # Python
    filter_gauss_3d(image, sigma)
    bilateral_filter(image, sigma_spatial, sigma_intensity)

Deconvolution
~~~~~~~~~~~~~

::

    % MATLAB
    psf_gen_new(raw_psf, dz_data, dz_psf)
    XR_rotate_PSF(psf, angle)
    
    # Python
    psf_gen(raw_psf, dz_data, dz_psf)
    rotate_psf(psf, angle)

Stitching
~~~~~~~~~

::

    % MATLAB
    normxcorr3_max_shift(template, image, max_shifts)
    featherBlending3D(tile1, tile2, bbox)
    
    # Python
    normxcorr3_max_shift(template, image, max_shifts)
    feather_blending_3d(tile1, tile2, bbox)

Common Patterns
---------------

**Workflow Orchestration**

MATLAB uses wrapper functions, Python uses classes or functions::

    % MATLAB
    XR_decon_data_wrapper(args)
    
    # Python - create your own workflow
    def deconvolution_workflow(data, psf):
        psf_proc = psf_gen(psf, dz_data=0.5, dz_psf=0.2)
        otf = decon_psf2otf(psf_proc, data.shape)
        # ... continue workflow
        return result

**Parallel Processing**

MATLAB uses parfor, Python uses various options::

    % MATLAB
    parfor i = 1:n
        result{i} = process(data{i});
    end
    
    # Python - use multiprocessing or joblib
    from joblib import Parallel, delayed
    results = Parallel(n_jobs=-1)(
        delayed(process)(data[i]) for i in range(n)
    )

**GPU Processing**

MATLAB uses gpuArray, Python uses CuPy::

    % MATLAB
    gpu_data = gpuArray(data);
    result = process(gpu_data);
    cpu_result = gather(result);
    
    # Python - use CuPy
    import cupy as cp
    gpu_data = cp.asarray(data)
    result = process(gpu_data)
    cpu_result = cp.asnumpy(result)

Import Structure
----------------

Python organizes imports more explicitly::

    % MATLAB - functions automatically available
    result = filterGauss3D(image, 2.0);
    
    # Python - explicit imports
    from petakit5d import filter_gauss_3d
    result = filter_gauss_3d(image, 2.0)

Best Practices
--------------

1. **Use top-level imports** for convenience::

    from petakit5d import read_tiff, filter_gauss_3d, write_tiff

2. **Check array shapes** - Python shows (z, y, x) or (depth, height, width)

3. **Use context managers** for files when available

4. **Leverage NumPy/SciPy** - many operations have efficient built-ins

5. **Use type hints** for better code documentation::

    def process_volume(volume: np.ndarray, sigma: float) -> np.ndarray:
        return filter_gauss_3d(volume, sigma)

Getting Help
------------

* Check function docstrings: ``help(filter_gauss_3d)``
* Read the API documentation
* Look at example notebooks in the ``examples/`` directory
* Report issues on GitHub
