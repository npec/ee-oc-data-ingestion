# Level-2 Swath Resampling

The script herein prepare Level-2 OC swath imagery for upload into Earth Engine.

## Data: 
- Sample files downloaded from NASA's [OB.DAAC](https://oceancolor.gsfc.nasa.gov/) and JAXA's [G-Portal](https://gportal.jaxa.jp/gpr/) are included in the `sample_data` folder.

## User steps:
See `swathresmaple.ipynb` for details
1. Obtain the products for your region of interest from the [OB.DAAC](https://oceancolor.gsfc.nasa.gov/) and/or [G-Portal](https://gportal.jaxa.jp/gpr/)
2. Define in `swathresmaple.ipynb` the input and output path, projection of the target map, and Google Cloud and Earth Engine parameters in case of using command-line tools.
3. Run the resampling cell. It outputs the result into a GeoTIFF file. Specify the output GeoTIFF file prior to that.
4. [Optional] - Upload the TIFF file to GCloud, then build the manifest file.
5. [Optional] - Use the manifest file to ingest the file.


## Resampled level-2 swath in Google Earth Engine
1. [MODIS/Aqua](https://code.earthengine.google.com/xxxx) 
2. [SGLI/GCOM-C](https://code.earthengine.google.com/xxxx)

The code was used in (Maure, Simon, Terauchi, 2022) https://doi.org/10.3390/rsxxxx



