import subprocess
import time
import traceback
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset
from osgeo import osr, gdal
from pyproj import Transformer, Geod
from pyresample import SwathDefinition, create_area_def
from pyresample.kd_tree import resample_nearest

from . import L2File, FileError


class SwathResample(L2File):

    def __init__(self, file, trg_tif,
                 srs='laea',
                 area_id='nowpap_region',
                 datum="WGS84"):
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
        self.datum = datum
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
    def resample(self, i, data, roi, key):
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
                      f"\n\t{i:2}: {key:>9} | Elapsed {info}")
            else:
                print(f"\t{i:2}: {key:>9} | Elapsed {info}")

        fill_value = data.fill_value
        out = resample_nearest(source_geo_def=self.source_geo,
                               data=data,
                               target_geo_def=self.target_geo,
                               fill_value=None,
                               radius_of_influence=roi)
        if key not in ('l2_flags', 'QA_flag'):
            mask = out.mask.copy()
            out[mask] = fill_value
            out.mask = mask
            np.ma.set_fill_value(out, fill_value)
        if key in ('l2_flags', 'QA_flag'):
            out[out.mask] = 0
        elapsed()
        return out

    def scale(self, key, data):
        """https://www.unidata.ucar.edu/mailing_lists/archives/netcdfgroup/2002/msg00034.html
        """
        if 'flag' in key:
            return data

        attrs = self.get_attrs(key=key).pop(key)
        if ('add_offset' in attrs.keys()) and \
                ('scale_factor' in attrs.keys()):
            self.add_offset = attrs.pop('add_offset')
            self.scale_factor = attrs.pop('scale_factor')
            return ((data - self.add_offset) / self.scale_factor).astype(np.int32)

        if key in ('chlor_a', 'CHLA', 'chl_ocx'):
            # min_val = 0
            # max_val = 100
            # max_int = (1 << 31) - 1
            self.add_offset = 0.001
            self.scale_factor = 1e-06
            return ((data - self.add_offset) / self.scale_factor).astype(np.int32)

        if ('Rrs' in key) or ('NWLR' in key):
            self.add_offset = 0.050
            self.scale_factor = 2e-06
            return ((data - self.add_offset) / self.scale_factor).astype(np.int32)

        if ('Offset' in attrs.keys()) and \
                ('Slope' in attrs.keys()):
            self.add_offset = attrs.pop('Offset')
            self.scale_factor = attrs.pop('Slope')
            return ((data - self.add_offset) / self.scale_factor).astype(np.int32)

    # ------------------
    # GeoTIFF Operations
    # ------------------

    @staticmethod
    def get_band(file):
        ds = gdal.Open(str(file), gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        meta = band.GetMetadata()
        sds = band.ReadAsArray()
        novalue = band.GetNoDataValue()

        sds = np.ma.masked_where(sds == novalue, sds)
        if 'valid_min' in meta.keys():
            mnv = float(meta['valid_min'])
            sds = np.ma.masked_where(sds < mnv, sds)
        if 'valid_max' in meta.keys():
            mxv = float(meta['valid_max'])
            sds = np.ma.masked_where(sds > mxv, sds)
        np.ma.set_fill_value(sds, novalue)
        if 'CHLA' in file.name:
            return sds.astype(np.float32)
        return sds

    def copy_tif(self, bands: int, src):
        """
            Creates GeoTIFF file from existing one
            Parameters
            ----------
                src:
                bands: int
                    number of bands to be contained in the output tif
            Returns
            -------
                geotif filename and SRS
        """

        # --------------
        # src file input
        # --------------
        ds = gdal.Open(str(src), gdal.GA_ReadOnly)

        # -----------------------
        # begin to create geotiff
        # -----------------------
        driver = gdal.GetDriverByName('GTiff')

        # --------------------------
        # Select GDAL GeoTIFF driver
        # --------------------------
        dtype = gdal.gdalconst.GDT_Int32

        # --------------------------------
        # Define rows/cols from array size
        # --------------------------------
        self.dataset = driver.Create(str(self.tmp_tif),
                                     xsize=ds.RasterXSize,
                                     ysize=ds.RasterYSize,
                                     bands=bands,
                                     eType=dtype)

        # ---------------------------------
        # Specify parameters of the GeoTIFF
        # ---------------------------------

        # ----------------
        # Set Geotransform
        # ----------------
        self.dataset.SetGeoTransform(ds.GetGeoTransform())

        # --------------
        # Set projection
        # --------------
        self.dataset.SetProjection(ds.GetProjection())
        self.dataset.SetMetadata(self.glob_attrs)
        return

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
        srs.ImportFromProj4(self.target_geo.proj4_string)
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
        self.dataset.SetMetadata(self.glob_attrs)
        return

    def tif_attrs(self, key: str = None):
        """
        Generates attributes from hdf5 files: SGLI, used in nc file writing
        :return: updated attributes
        """
        attrs = self.get_attrs(key=key)
        if self.file.suffix == '.nc':
            if key:
                attrs = {f'{key}_{at}': val
                         for at, val in attrs.pop(key).items()
                         if f'{val}' != 'NA'}
                if key == 'l2_flags':
                    return attrs
                attrs[f'{key}_valid_min'] = self.data.min()
                attrs[f'{key}_valid_max'] = self.data.max()
                attrs[f'{key}_scale_factor'] = self.scale_factor
                attrs[f'{key}_add_offset'] = self.add_offset
                return attrs
            return attrs

        if key:
            exclude = ('Slope', 'Offset', 'Rrs_Slope', 'Rrs_Offset')
            # ,'Minimum_valid_DN', 'Maximum_valid_DN', 'Error_DN'
            attrs = {f'{key.replace("NWLR", "Rrs")}_{at}': val
                     for at, val in attrs.pop(key).items()
                     if not ((f'{val}' == 'NA') or (at in exclude))}
            if key == 'QA_flag':
                return attrs
            attrs[f'{key.replace("NWLR", "Rrs")}_valid_min'] = self.data.min()
            attrs[f'{key.replace("NWLR", "Rrs")}_valid_max'] = self.data.max()
            attrs[f'{key.replace("NWLR", "Rrs")}_scale_factor'] = self.scale_factor
            attrs[f'{key.replace("NWLR", "Rrs")}_add_offset'] = self.add_offset
            return attrs
        return attrs

    def append(self, band: int, key: str):
        attrs = self.tif_attrs(key=key)
        band = self.dataset.GetRasterBand(band)  # Get band
        band.SetDescription(key)
        band.SetMetadata(attrs)

        if key not in ('l2_flags', 'QA_flag'):
            mask = self.data.mask.copy()
            self.data[mask] = self.data.fill_value
            band.SetNoDataValue(int(self.data.fill_value))  # Set fill value
        band.WriteArray(self.data.astype(np.int32))  # Write data array to band #
        band.FlushCache()  # Export data
        band = None
        return band

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
        if self.dataset is not None:
            self.dataset = None
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
        left, right = bbox[0], bbox[2]
        bottom, top = bbox[1], bbox[3]
        print(f'Bounds: {bbox}')
        # area_extent = left, bottom, right, top
        """
        Transform boundary densifying the edges to account for nonlinear 
        transformations along these edges and extracting the outermost bounds.
        https://pyproj4.github.io/pyproj/stable/api/transformer.html
        """
        # -----------
        # pyproj proj
        # -----------
        # Avoid resolution becoming smaller than original
        lonlat = self.srs in ('lonlat', 'latlon', 'latlong', 'longlat')
        resolution = self.spatial_resolution()
        proj_def = dict(datum=self.datum, lat_0=lat_0, lon_0=lon_0, proj=self.srs)
        if not lonlat:
            resolution += 1
        transproj = Transformer.from_crs(
            "EPSG:4326",
            proj_def,
            always_xy=True
        )
        # -----------
        # area extent
        # -----------
        area_extent = transproj.transform_bounds(
            left, bottom, right, top
        )
        print(f'Extent: {area_extent}')
        print(f'Proj: {proj_def}\nAreaID: {self.area_id}\n'
              f'ResolutionUsed: {resolution}')
        if not lonlat:
            # ---------------
            # area definition
            # ---------------
            return create_area_def(projection=proj_def,
                                   area_extent=area_extent,
                                   resolution=resolution,
                                   area_id=self.area_id)

        # need shape if lon/lat proj is used, otherwise create_area_dim fails with metric resolution
        g = Geod(ellps=self.datum)
        dy = (resolution * 360.) / (2. * np.pi * g.b)
        dx = (resolution * 360.) / (2. * np.pi * g.a * np.cos(np.deg2rad(lat_0)))
        shape = (np.arange(area_extent[3] - dy / 2, area_extent[1], -dy).size,
                 np.arange(area_extent[0] + dx / 2, area_extent[2], dx).size)

        # ---------------
        # area definition
        # ---------------
        return create_area_def(projection=proj_def,
                               area_extent=area_extent,
                               resolution=resolution,
                               shape=shape,
                               area_id=self.area_id)

    def close(self, obj=None):
        if isinstance(obj, Dataset) or \
                isinstance(obj, h5py.File):
            super().close(obj=obj)
            self.dataset = None
            return
        super().close()
        self.dataset = None
