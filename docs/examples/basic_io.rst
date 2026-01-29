Basic I/O Operations
====================

This tutorial covers reading and writing microscopy data files.

For complete code examples, see the Jupyter notebook: ``examples/01_basic_io.ipynb``

Reading TIFF Files
------------------

Reading single TIFF files::

    from petakit5d import read_tiff
    
    image = read_tiff('data.tif')
    print(f"Shape: {image.shape}, dtype: {image.dtype}")

Writing TIFF Files
------------------

Writing processed data::

    from petakit5d import write_tiff
    
    write_tiff('output.tif', processed_image)

Working with Zarr
-----------------

Creating Zarr files for large datasets::

    from petakit5d import create_zarr, write_zarr_block
    
    # Create Zarr file
    create_zarr('large_data.zarr', 
                data_size=(1000, 2048, 2048),
                chunk_size=(10, 512, 512),
                dtype='uint16')
    
    # Write blocks
    for i in range(0, 1000, 10):
        block = process_chunk(i)
        write_zarr_block('large_data.zarr', block, 
                        bbox_zarryu=[i, i+10, 0, 2048, 0, 2048])
