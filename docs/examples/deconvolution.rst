Deconvolution Workflow
======================

Complete deconvolution pipeline for microscopy images.

See: ``examples/03_deconvolution_workflow.ipynb``

PSF Preprocessing
-----------------

Prepare PSF for deconvolution::

    from petakit5d import psf_gen, rotate_psf, decon_psf2otf
    
    # Preprocess raw PSF
    psf = psf_gen(raw_psf, 
                  dz_data=0.5,
                  dz_psf=0.2,
                  method='masked')
    
    # Rotate PSF if needed
    rotated_psf = rotate_psf(psf, angle=32.5, reverse=False)
    
    # Convert to OTF
    otf = decon_psf2otf(rotated_psf, image_size)

Complete Pipeline
-----------------

Full deconvolution workflow::

    # Load data
    image = read_tiff('data.tif')
    raw_psf = read_tiff('psf.tif')
    
    # Preprocess PSF
    psf = psf_gen(raw_psf, dz_data=0.5, dz_psf=0.2)
    otf = decon_psf2otf(psf, image.shape)
    
    # Apply deconvolution (external library needed)
    # deconvolved = richardson_lucy(image, otf)
    
    # Save result
    write_tiff('deconvolved.tif', deconvolved)
