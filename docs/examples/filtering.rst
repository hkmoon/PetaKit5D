Image Filtering
===============

This tutorial covers various image filtering operations.

For complete code examples, see: ``examples/02_image_filtering.ipynb``

Gaussian Filtering
------------------

3D Gaussian smoothing::

    from petakit5d import filter_gauss_3d
    
    # Apply Gaussian filter
    smoothed = filter_gauss_3d(image, sigma=2.0)

Bilateral Filtering
-------------------

Edge-preserving smoothing::

    from petakit5d import bilateral_filter
    
    smoothed = bilateral_filter(image, 
                                sigma_spatial=5,
                                sigma_intensity=0.1)

Laplacian of Gaussian
---------------------

Edge detection with LoG::

    from petakit5d import filter_log_nd
    
    edges = filter_log_nd(image, sigma=2.0)
