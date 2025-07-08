'''
Convert data (e.g. RGB) from L2 files into GeoTiff, which is useful for georeferencing
'''

import os
from glob import glob
from itertools import chain
import xarray as xr


PATTERNS = ['PRS_L2_*.nc']


def convert_tiff(data, savename):
    if data.rio.crs:
        data.rio.to_raster(savename)
    else:
        raise ValueError('Please add crs to the input data.')
    '''
    # --- backup ---
    else:
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        print('Missing proj info !!! We assign it as 4326 ...')
        #data.rio.write_crs('epsg:4326').rio.to_raster(savename)
        data.rio.to_raster(savename, recalc=True)

    # Load original (band, y, x)
    with rasterio.open(savename) as src:
        data = src.read()  # shape: (band, y, x)
    
    # Transpose to (y, x, band)
    data = np.transpose(data, (1, 2, 0))
    
    # Save new file
    # Flip both axes
    data = np.flip(data, axis=(0, 1))  # flip y and x axes

    transform = from_origin(0, data.shape[0], 1, -1)
    
    new_meta = {
        'driver': 'GTiff',
        'height': data.shape[0],
        'width': data.shape[1],
        'count': data.shape[2],
        'dtype': data.dtype,
        'transform': transform,
        'crs': None  # No projection
    }
    
    with rasterio.open(savename.replace('.tif', '_test.tif'), 'w', **new_meta) as dst:
        for i in range(data.shape[2]):
            dst.write(data[:, :, i], i + 1)
    '''


def main():
    # the root dir of L2 data
    data_dir = '../hypergas/resources/test_data/ch4_cases/'

    # the variable name to be kept
    varname = 'rgb'

    l2_filelist = list(chain(
        *[glob(os.path.join(data_dir, '**', pattern), recursive=True) for pattern in PATTERNS]))
    l2_filelist = list(sorted(l2_filelist))

    for filename in l2_filelist:
        print(f'Converting {filename}')
        ds = xr.open_dataset(filename, decode_coords='all')
        data = ds[varname]
        savename = filename.replace('.nc', '_rgb.tif')
        print(f'Exported to {savename}')
        convert_tiff(data, savename)


if __name__ == '__main__':
    main()
