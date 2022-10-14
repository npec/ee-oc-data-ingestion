# Level-2 Swath Remapping

The script herein preprocess Level-2 OC swath imagery for upload and ingestion into Google Earth Engine (GEE).
The script can be adapted to any swath imagery other than OC data. 

## Data: 
- Sample files were downloaded from NASA's [OB.DAAC](https://oceancolor.gsfc.nasa.gov/) and JAXA's [G-Portal](https://gportal.jaxa.jp/gpr/).
- [MODIS/Aqua](https://oceandata.sci.gsfc.nasa.gov/ob/getfile/A2022125035500.L2_LAC_OC.nc)
- [SGLI/GCOM-C](https://gportal.jaxa.jp/download/standard/GCOM-C/GCOM-C.SGLI/L2.OCEAN.IWPR/3/2022/05/03/GC1SG1_202205030152F05810_L2SG_IWPRQ_3000.h5)

Obtain the files from the above links and once downloaded save them in `sample_data` folder.

## User steps:
There 3 Python notebooks included
1. [SWATH-2-GEOTIFF-2-GEE.ipynb](https://github.com/npec/ee-oc-data-ingestion/blob/main/SWATH-2-GEOTIFF-2-GEE.ipynb) - does the swath remapping and translation of the GeoTIFF image. It also includes the steps for data ingestion. Further, remapping with JAXA's DCT is also included.
2. [SWATH_REMAP_MODISA.ipynb](https://github.com/npec/ee-oc-data-ingestion/blob/main/SWATH_REMAP_MODISA.ipynb) - reproduces the figures for MODIS/Aqua remapping. The histograms in this notebook are also reproduced in GEE and can be found at [this link]().
3. [SWATH_REMAP_SGLI.ipynb](https://github.com/npec/ee-oc-data-ingestion/blob/main/SWATH_REMAP_SGLI.ipynb) - reproduces the figures for SGLI/GCOM-C remapping. Similar to MODIS/Aqua, the histograms in this notebook are also reproduced in GEE and can be found at [this link]().

See the above notebooks for details, briefly:
1. Obtain the products for your region of interest from the [OB.DAAC](https://oceancolor.gsfc.nasa.gov/), [G-Portal](https://gportal.jaxa.jp/gpr/), or any other data provider. 
2. Define in [`SWATH-2-GEOTIFF-2-GEE.ipynb`](https://github.com/npec/ee-oc-data-ingestion/blob/main/SWATH-2-GEOTIFF-2-GEE.ipynb) the input and output parameters including the I/O path, projection of the target area, and the Google Cloud bucket and Earth Engine parameters in case of using command-line tools.
3. Run the remapping cells and watch the output results in the `result` folder. The DCT remapping step can be commented out.
4. [Optional] - Upload the GeoTIFF file to GCloud, then build the manifest file.
5. [Optional] - Use the manifest file to ingest the file.

## Remapped Level-2 swath in Google Earth Engine
1. [MODIS/Aqua](https://ermaure.users.earthengine.app/view/swath-reprojection-modisa) 
2. [SGLI/GCOM-C](https://ermaure.users.earthengine.app/view/swath-reprojection-sgli)
3. [SGLI/GCOM-C PY vs. DCT](https://ermaure.users.earthengine.app/view/py-vs-dct)

The code was used in [Maúre, Simon, Terauchi, 2022 Remote Sensing. 2022; 14(19):4906](https://doi.org/10.3390/rs14194906)



