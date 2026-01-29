Tile Stitching
===============

Complete workflow for stitching image tiles.

See: ``examples/04_stitching_example.ipynb``

Computing Tile Offsets
----------------------

Find overlap between tiles::

    from petakit5d import normxcorr3_max_shift
    
    # Find offset
    offset = normxcorr3_max_shift(tile1, tile2, 
                                  max_shifts=[50, 50, 50])

Feather Blending
----------------

Seamless blending of overlapping tiles::

    from petakit5d import feather_blending_3d, distance_weight_single_axis
    
    # Compute distance weights
    weights = distance_weight_single_axis(range_size, 
                                         buffer_size=10,
                                         dfactor=0.99)
    
    # Blend tiles
    merged = feather_blending_3d(tile1, tile2, 
                                overlap_bbox,
                                distance_weight)

Complete Pipeline
-----------------

Full stitching workflow::

    # Load tiles
    tiles = [read_tiff(f'tile_{i}.tif') for i in range(n_tiles)]
    
    # Compute all pairwise offsets
    offsets = {}
    for i, j in tile_pairs:
        offset = normxcorr3_max_shift(tiles[i], tiles[j], max_shifts)
        offsets[(i,j)] = offset
    
    # Global optimization (external library)
    # positions = optimize_positions(offsets)
    
    # Blend tiles
    result = stitch_tiles(tiles, positions, feather_blending_3d)
    
    # Save
    write_tiff('stitched.tif', result)
