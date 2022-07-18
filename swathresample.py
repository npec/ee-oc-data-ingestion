import subprocess
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
import pyproj
from netCDF4 import Dataset
from osgeo import osr, gdal
from pyresample import SwathDefinition, create_area_def
from pyresample.kd_tree import resample_nearest

from . import L2File, FileError


class SwathResample(L2File):

    def __init__(self, file, trg_tif, srs='laea', area_id='custom'):
        super().__init__(file)
        self.uri = None
        self.srs = srs
        self.area_id = area_id
        self.dataset = None
        self.target_geo = None
        self.data = None
        self.scale_factor = None
        self.add_offset = None
        self.source_geo = None
        self.tmp_tif = Path(trg_tif).parent.joinpath(
            Path(trg_tif).name.replace('.tif', '_tmp.tif'))
        self.trg_tif = trg_tif
        self.datum = "WGS84"
        # ---------------
        # geo-projections
        # ---------------
        self.source_geo = SwathDefinition(
            lons=self.lon, lats=self.lat)
        self.target_geo = self.get_area_def()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is not None:
            exc_info = ''.join(traceback.format_exception(
                exc_type, exc_value, exc_traceback))
            raise FileError(exc_info)
        if exc_type is None:
            self.dataset = None
            self.close()

    # ------------------
    # Level-2 Operations
    # ------------------
    def resample(self, i, data, roi):
        start_time = time.perf_counter()
        """
        Mapping of the L2 data.
        target_geo: projection
        remapped data onto a regular grid
        """

        def elapsed():
            sec = time.perf_counter() - start_time
            mn = sec // 60
            sec = sec % 60
            info = f'{int(mn):2} min {int(sec):02} sec' \
                if mn > 0 else f'{sec:.3f} sec'
            if i == 0:
                print(f"resampling... "
                      f"\n\t{i:2}: {self.key:>9} | Elapsed {info}")
            else:
                print(f"\t{i:2}: {self.key:>9} | Elapsed {info}")

        fill_value = data.fill_value
        out = resample_nearest(source_geo_def=self.source_geo,
                               data=data,
                               target_geo_def=self.target_geo,
                               fill_value=None,
                               radius_of_influence=roi)
        if self.key not in ('l2_flags', 'QA_flag'):
            mask = out.mask.copy()
            out[mask] = fill_value
            out.mask = mask
            np.ma.set_fill_value(out, fill_value)
        elapsed()
        return out

    def scale(self, key, data):
        """https://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2002/msg00034.html
        """
        if 'flag' in key:
            self.add_offset = 0
            self.scale_factor = 1
            return data
        if key in ('chlor_a', 'CHLA'):
            min_val = 0.001
            max_val = 100
            max_int = (1 << 31) - 1
            self.add_offset = np.float32(min_val)
            self.scale_factor = np.float32((max_val - min_val) / max_int)
            return ((data - self.add_offset) / self.scale_factor).astype(np.int32)
        if 'Rrs' in key:
            self.add_offset = 0.050
            self.scale_factor = 2e-06
            return ((data - self.add_offset) / self.scale_factor).astype(np.int32)
        attrs = self.get_attrs(key=key).pop(key)
        self.add_offset = 0
        self.scale_factor = 1
        if 'add_offset' in attrs.keys():
            self.add_offset = attrs.pop('add_offset')
        if 'scale_factor' in attrs.keys():
            self.scale_factor = attrs.pop('scale_factor')
        return ((data - self.add_offset) / self.scale_factor).astype(np.int32)

    # ------------------
    # GeoTIFF Operations
    # ------------------

    def open(self, bands: int):
        """
        Creates GeoTIFF file
        Parameters
        ----------
            bands: int
                number of bands to be contained in the output tif
        Returns
        -------
            geotif filename and SRS
        """

        # ---------------------
        # osr output projection
        # ---------------------
        srs = osr.SpatialReference()
        srs.ImportFromProj4(self.target_geo.proj_dict)
        srs.SetProjCS(self.srs)
        srs.SetWellKnownGeogCS(self.datum)
        wkt = srs.ExportToWkt()

        # -----------------------
        # begin to create geotiff
        # -----------------------
        driver = gdal.GetDriverByName('GTiff')

        # --------------------------
        # Select GDAL GeoTIFF driver
        # --------------------------
        dtype = gdal.gdalconst.GDT_Int32

        # -----------------------
        # Define output data type
        # -----------------------
        height, width = self.target_geo.shape

        # --------------------------------
        # Define rows/cols from array size
        # --------------------------------
        self.dataset = driver.Create(str(self.tmp_tif),
                                     xsize=width,
                                     ysize=height,
                                     bands=bands,
                                     eType=dtype)

        # ---------------------------------
        # Specify parameters of the GeoTIFF
        # ---------------------------------
        extent = self.target_geo.area_extent
        # ----------------
        # Set Geotransform
        # ----------------
        transform = [extent[0], self.target_geo.pixel_size_x, 0,
                     extent[-1], 0, -self.target_geo.pixel_size_y]
        self.dataset.SetGeoTransform(transform)

        # --------------
        # Set projection
        # --------------
        self.dataset.SetProjection(wkt)
        return

    def tif_attrs(self, key: str = None):
        """
        Generates attributes from hdf5 files: SGLI, used in nc file writing
        :return: updated attributes
        """
        attrs = self.get_attrs(key=key)
        if self.file.suffix == '.nc':
            if key:
                attrs = {f'{key}_{at}: {val}'
                         for at, val in attrs.pop(key).items()}
                if key == 'l2_flags':
                    return attrs
                return attrs.update({f'{key}_valid_min': f'{self.data.min()}',
                                     f'{key}_valid_max': f'{self.data.max()}',
                                     f'{key}_scale_factor': f'{self.scale_factor}',
                                     f'{key}_add_offset': f'{self.add_offset}'})
            return attrs

        if key:
            attrs = {f'{key}_{at}: {val}'
                     for at, val in attrs.pop(key).items()}
            if key == 'QA_flag':
                return attrs
            attrs.update({f'{key}_valid_min': f'{self.data.min()}',
                          f'{key}_valid_max': f'{self.data.max()}',
                          f'{key}_scale_factor': f'{self.scale_factor}',
                          f'{key}_add_offset': f'{self.add_offset}'})
            return attrs

    def append(self, band: int, key: str):
        attrs = self.tif_attrs(key=key)
        band = self.dataset.GetRasterBand(band)  # Get band #
        band.SetDescription(key)
        band.SetMetadata(attrs)

        if key not in ('l2_flags', 'QA_flag'):
            mask = self.data.mask.copy()
            self.data[mask] = self.data.fill_value
            band.SetNoDataValue(int(self.data.fill_value))  # Set fill value
        band.WriteArray(self.data.astype(np.int32))  # Write data array to band #
        band.FlushCache()  # Export data
        band = None

    def translate(self):
        """
        https://trac.osgeo.org/gdal/wiki/CloudOptimizedGeoTIFF
        https://gdal.org/python/osgeo.gdal-module.html
        The gdal_translate utility can be used to convert raster
        data between different formats, potentially performing some
        operations like subsettings, resampling, and rescaling pixels
        in the process.
        """

        # --------------
        # WARP/TRANSLATE
        # --------------
        cmd = 'gdal_translate ' \
              '-of GTiff ' \
              '-r nearest ' \
              '-ot Int32 ' \
              '-co TILED=YES ' \
              '-co COPY_SRC_OVERVIEWS=YES ' \
              '-co compress=lzw ' \
              f'{self.tmp_tif} {self.trg_tif}'
        dash = "=" * len(cmd)
        print(f'{dash}\n{cmd}\n{dash}')
        # --------------------
        # Get rid of input tif
        # --------------------
        subprocess.check_call(f'gdaladdo -r nearest {self.tmp_tif}')
        status = subprocess.check_call(cmd, shell=True)
        return status

    def get_area_def(self):
        """Generates the grid projection for mapping L2 data based on input data resolution.
        If lonlat griding scheme is used, the grid resolution will be exact at the centre

        Returns
        -------
        AreaDefinition: AreaDefinition
           area definition with pyproj information embedded
        """
        # --------------
        # subarea limits (box)
        # --------------
        lon_0, lat_0 = self.get_swath_centre()
        bbox = self.get_bounds()
        lower_left_x, upper_right_x = bbox[0], bbox[2]
        lower_left_y, upper_right_y = bbox[1], bbox[3]

        # -----------
        # pyproj proj
        # -----------
        proj_def = dict(datum=self.datum, lat_0=lat_0, lon_0=lon_0, proj=self.srs)
        if self.srs not in ('lonlat', 'longlat'):
            proj = pyproj.Proj(proj_def)
            (lower_left_x, lower_left_y) = proj.transform(lower_left_x, lower_left_y)
            (upper_right_x, upper_right_y) = proj.transform(upper_right_x, upper_right_y)

        # -----------
        # area extent
        # -----------
        area_extent = (lower_left_x, lower_left_y,
                       upper_right_x, upper_right_y)

        # ---------------
        # area definition
        # ---------------
        # Avoid resolution becoming smaller than original
        resolution = self.spatial_resolution() + 1
        return create_area_def(projection=proj_def,
                               area_extent=area_extent,
                               resolution=resolution,
                               area_id=self.area_id)

    def close(self, obj=None):
        if isinstance(obj, Dataset) or \
                isinstance(obj, h5py.File):
            super().close(obj=obj)
            self.dataset = None
            return
        super().close()
        self.dataset = None
