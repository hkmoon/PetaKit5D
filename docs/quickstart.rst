Quick Start Guide
=================

This guide will help you get started with PetaKit5D.

Basic I/O Operations
--------------------

Reading and writing TIFF files::

    from petakit5d import read_tiff, write_tiff

    # Read a TIFF file
    image = read_tiff('input.tif')

    # Write a TIFF file
    write_tiff(image, 'output.tif')

Working with Zarr::

    from petakit5d import read_zarr, write_zarr, create_zarr

    # Create a Zarr file
    create_zarr('data.zarr', data_size=(100, 512, 512), 
                chunk_size=(10, 256, 256))

    # Write blocks
    write_zarr_block('data.zarr', data_block, bbox_zarryu=[0, 10, 0, 256, 0, 256])

Image Filtering
---------------

Gaussian filtering::

    from petakit5d import filter_gauss_3d

    # Apply 3D Gaussian filter
    filtered = filter_gauss_3d(image, sigma=2.0)

Bilateral filtering::

    from petakit5d import bilateral_filter

    # Edge-preserving smoothing
    smoothed = bilateral_filter(image, sigma_s=5, sigma_r=0.1)

Deconvolution Workflow
----------------------

PSF preprocessing::

    from petakit5d import psf_gen, decon_psf2otf

    # Preprocess raw PSF
    psf = psf_gen(raw_psf, dz_data=0.5, dz_psf=0.2, method='masked')

    # Convert to OTF
    otf = decon_psf2otf(psf, image_size=(100, 512, 512))

Tile Stitching
--------------

Computing tile offsets::

    from petakit5d import normxcorr3_max_shift

    # Find offset between two tiles
    offset = normxcorr3_max_shift(tile1, tile2, max_shifts=[50, 50, 50])

Blending tiles::

    from petakit5d import feather_blending_3d

    # Seamless blending
    merged = feather_blending_3d(tile1, tile2, overlap_bbox, distance_weight)

Volume Processing
-----------------

Cropping::

    from petakit5d import crop_3d

    # Crop a 3D volume
    cropped = crop_3d(volume, bbox=[10, 90, 50, 450, 50, 450])

Resampling::

    from petakit5d import resample_stack_3d

    # Downsample by factor of 2
    downsampled = resample_stack_3d(volume, 
                                    resample_factor=[2, 2, 2],
                                    method='cubic')

MIP Generation
--------------

Creating maximum intensity projections::

    from petakit5d import save_mip_frame

    # Save MIPs along all axes
    save_mip_frame(volume, 'output_mip.tif', axis_list=[0, 1, 2])

Next Steps
----------

* Check the :doc:`api/index` for complete API reference
* Explore :doc:`examples/index` for detailed tutorials
* Read the :doc:`migration` guide if coming from MATLAB
