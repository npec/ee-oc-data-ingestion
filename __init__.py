import re
from pathlib import Path

import h5py
import numpy as np
from dateutil.parser import parse
from netCDF4 import Dataset


def geo_interp(array: np.array, interval: int):
    """
        Bilinear  interpolation of SGLI geo-location corners to a spatial grid
        Author: K. Ogata
        License: MIT

        Parameters
        ----------
        array: np.array
            either lon or lat
        interval: int
            resampling interval in pixels

        Return
        ------
        out_geo: np.array
            2-D array with dims == to geophysical variables

    """
    sds = np.concatenate((array, array[-1].reshape(1, -1)), axis=0)
    sds = np.concatenate((sds, sds[:, -1].reshape(-1, 1)), axis=1)

    ratio_0 = np.tile(
        np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32),
        (sds.shape[0] * interval, sds.shape[1] - 1))

    ratio_1 = np.tile(
        np.linspace(0, (interval - 1) / interval, interval, dtype=np.float32).reshape(-1, 1),
        (sds.shape[0] - 1, (sds.shape[1] - 1) * interval))

    sds = np.repeat(sds, interval, axis=0)
    sds = np.repeat(sds, interval, axis=1)
    interp = (1. - ratio_0) * sds[:, :-interval] + ratio_0 * sds[:, interval:]
    return (1. - ratio_1) * interp[:-interval, :] + ratio_1 * interp[interval:, :]


def haversine(sx, ex, sy, ey):
    """
    https://www.movable-type.co.uk/scripts/latlong.html
    """
    rad = np.pi / 180
    dx, dy = (ex - sx) * rad, (ey - sy) * rad
    sx *= rad
    sy *= rad
    ey *= rad

    bx = np.cos(ey) * np.cos(dx)
    by = np.cos(ey) * np.sin(dx)
    cy = np.arctan2(np.sin(sy) + np.sin(ey),
                    np.sqrt((np.cos(sy) + bx) * (np.cos(sy) + bx) + by * by))
    cx = sx + np.arctan2(by, np.cos(sy) + bx)
    return cx / rad, cy / rad


class FileError(Exception):
    """A custom exception used to report errors"""

    def __init__(self, message: str):
        super().__init__(message)


class IO:
    def __init__(self, file):
        self.input_files = None
        if type(file) in (tuple, list):
            self.input_files = file
            iwpr, = [f for f in file if 'IWPR' in Path(f).name]
            self.file = Path(iwpr)
        else:
            self.file = Path(file)
        self.key = None
        self.fill_int = -2147483648
        self.fill_float = -32767
        self.nwlr_obj = None
        self.nwlr_path = None
        if self.file.suffix == '.h5':
            self.obj = h5py.File(self.file, mode='r')
            self.glob_attrs = self.fmt_attr(path='Global_attributes')
            # In case input_file is not None
            if self.input_files is not None:
                file, = [f for f in self.input_files if 'NWLR' in Path(f).name]
                self.nwlr_obj = h5py.File(file, mode='r')
                self.glob_attrs.update(self.fmt_attr(path='Global_attributes', nwlr=True))
                self.nwlr_path = self.nwlr_obj['Image_data']
            self.path = self.obj['Image_data']
            self.lon = self.get_geo(key='Longitude')
            self.lat = self.get_geo(key='Latitude')
        if self.file.suffix == '.nc':
            self.obj = Dataset(self.file, mode='r')
            self.path = self.obj.groups['geophysical_data']
            self.lon = self.get_geo(key='longitude')
            self.lat = self.get_geo(key='latitude')
            self.glob_attrs = {
                at: self.obj.getncattr(at) for at in self.obj.ncattrs()}

    # -----------
    # Get methods
    # -----------
    def get_keys(self):
        """
            Gets the key (variable) names from level-2 data which are
            found in `geophysical_data` group in the NASA's netCDF4 file
            or in `Image_data` group in JAXA's HDF5 file.

        Returns
        -------
            list
                A list of geophysical variables such as Rrs, chlor_a, etc
        """

        if self.file.suffix == '.nc':
            return list(self.path.variables.keys())

        if self.file.suffix == '.h5':
            keys = [key for key in self.path.keys()
                    if key != 'Line_tai93']
            if self.nwlr_path is not None:
                keys += [key for key in self.nwlr_path.keys()
                         if key != 'Line_tai93']
            return keys
        raise FileError('Unexpected file format')

    def get_attrs(self, key: str = None):
        if self.file.suffix == '.nc':
            return self.nc_attrs(key=key)
        if self.file.suffix == '.h5':
            return self.h5_attrs(key=key)
        raise FileError('Unexpected file format')

    def get_geo(self, key: str):
        if self.file.suffix == '.nc':
            return self.obj.groups['navigation_data'][key][:]

        data = self.get_sds(key=key)
        interval = self.fmt_attr(
            path=f'Geometry_data/{key}'
        ).get('Resampling_interval')

        attrs = self.fmt_attr(path=f'Image_data')
        nol = attrs.get('Number_of_lines')
        nop = attrs.get('Number_of_pixels')
        img_slice = (slice(0, nol), slice(0, nop))

        if key == 'Longitude':
            stride = False
            if np.abs(np.nanmin(data) - np.nanmax(data)) > 180.:
                stride = True
                data[data < 0] = 360. + data[data < 0]
            data = geo_interp(array=data, interval=interval)[img_slice]
            if stride:
                data[data > 180.] = data[data > 180.] - 360.
            return data
        # Get Latitude
        return geo_interp(array=data, interval=interval)[img_slice]

    # ----------
    # Attributes
    # ----------
    def nc_attrs(self, key: str = None):
        attrs = self.glob_attrs.copy()
        if key:
            key_attrs = {at: self.path[key].getncattr(at)
                         for at in self.path[key].ncattrs()}
            units = [key for key in key_attrs.keys()
                     if 'unit' in key.lower()]
            if len(units) > 0:
                if units[0] == 'Unit':
                    unit = key_attrs.pop(units[0])
                    key_attrs['units'] = unit
            if len(units) == 0:
                key_attrs['units'] = 'NA'
            attrs.update({key: key_attrs})
        return attrs

    def h5_attrs(self, key: str = None):
        attrs = self.glob_attrs.copy()
        if key is None:
            return attrs
        if key in self.obj.keys():
            return self.fmt_attr(path=key)
        path = f'Image_data/{key}'
        if key in ('Longitude', 'Latitude'):
            path = f'Geometry_data/{key}'
        if (self.nwlr_obj is not None) and ('NWLR' in key):
            key_attrs = self.fmt_attr(path=path, nwlr=True)
        else:
            key_attrs = self.fmt_attr(path=path)
        if 'Unit' in key_attrs.keys():
            unit = key_attrs.pop('Unit')
            key_attrs['units'] = unit
        attrs.update({key: key_attrs})
        return attrs

    def fmt_attr(self, path: str, nwlr=False):
        if nwlr:
            obj = self.nwlr_obj
        else:
            obj = self.obj
        result = {}
        for key, val in obj[path].attrs.items():
            if key in ('Dim0', 'Dim1'):
                continue
            try:
                val = val[0]
            except IndexError:
                pass
            if type(val) in (bytes, np.bytes_):
                val = val.decode()
            result.update({key: val})
        desc = result['Data_description'] \
            if 'Data_description' in result.keys() else None
        if desc and ('Remote Sensing Reflectance(Rrs)' in desc):
            result['units'] = result.pop('Rrs_unit')
        return result

    def get_dimensions(self):
        if self.file.suffix == '.nc':
            height = self.obj.dimensions['number_of_lines'].size
            width = self.obj.dimensions['pixels_per_line'].size
        elif self.file.suffix == '.h5':
            attrs = self.fmt_attr(path='Image_data')
            height = attrs.pop('Number_of_lines')
            width = attrs.pop('Number_of_pixels')
        else:
            raise FileError('Unexpected file format')
        return height, width

    # ----
    # Data
    # ----
    def get_data(self, key: str):
        """
          get the data from file
          return masked array with geophysical data

          Parameters
          ----------
          key: str
             pointer of the data to be read in the file

          Returns
          -------
          data: np.ma.array
            key geophysical data masked_array

        """
        if self.file.suffix == '.nc':
            return self.path[key][:]
        if self.file.suffix == '.h5':
            return self.get_sds(key=key)
        raise FileError('Unexpected file format')

    def get_dn(self, key: str):
        """
          get the digital number

          Parameters
          ----------
          key: str
             variable name

          Returns
          -------
          array: np.array
        """
        if key in ('Longitude', 'Latitude'):
            return self.obj[f'Geometry_data/{key}'][:]
        if ('NWLR' in key) and (self.nwlr_obj is not None):
            return self.nwlr_path[key][:]
        return self.path[key][:]

    def get_sds(self, key: str):
        """
            https://shikisai.jaxa.jp/faq/docs/GCOM-C_Products_Users_Guide_entrylevel__attach4_jp_191007.pdf#page=46
        """
        attrs = self.h5_attrs(key=key).pop(key)
        sdn = self.get_dn(key=key)

        if key == 'QA_flag':
            return sdn.astype(np.int16)

        mask = False
        if 'Error_DN' in attrs.keys():
            mask = mask | np.where(np.equal(sdn, attrs.pop('Error_DN')), True, False)
        if 'Land_DN' in attrs.keys():
            mask = mask | np.where(np.equal(sdn, attrs.pop('Land_DN')), True, False)
        if 'Cloud_error_DN' in attrs.keys():
            mask = mask | np.where(np.equal(sdn, attrs.pop('Cloud_error_DN')), True, False)
        if 'Retrieval_error_DN' in attrs.keys():
            mask = mask | np.where(np.equal(sdn, attrs.pop('Retrieval_error_DN')), True, False)
        if ('Minimum_valid_DN' in attrs.keys()) and ('Maximum_valid_DN' in attrs.keys()):
            mask = mask | np.where((sdn <= attrs.pop('Minimum_valid_DN')) |
                                   (sdn >= attrs.pop('Maximum_valid_DN')), True, False)

        # Convert DN to PV
        slope, offset = 1, 0
        if 'NWLR' in key:
            if ('Rrs_slope' in attrs.keys()) and \
                    ('Rrs_slope' in attrs.keys()):
                slope = attrs.pop('Rrs_slope')
                offset = attrs.pop('Rrs_offset')
        else:
            if ('Slope' in attrs.keys()) and \
                    ('Offset' in attrs.keys()):
                slope = attrs.pop('Slope')
                offset = attrs.pop('Offset')

        sds = np.ma.masked_where(
            mask, sdn * slope + offset
        ).astype(np.float32)
        np.ma.set_fill_value(sds, self.fill_float)
        return sds

    # ----------
    # Close File
    # ----------
    def close(self, obj=None):
        if obj is None:
            self.obj.close()
            if self.nwlr_obj is not None:
                self.nwlr_obj.close()
            [setattr(self, key, None)
             for key in self.__dict__.keys()]
            return
        obj.close()


class L2File(IO):

    # -----------
    # Get methods
    # -----------
    def get_swath_centre(self):

        if self.file.name.endswith('.nc'):
            scan = self.obj.groups['scan_line_attributes']
            clon = np.median(scan['clon'][:])
            clat = np.median(scan['clat'][:])
            return clon, clat

        if self.file.name.endswith('.h5'):
            # Descending sensor
            attrs = self.fmt_attr(path='Image_data')
            pixels = attrs.pop('Number_of_pixels')
            lines = attrs.pop('Number_of_lines')
            xb, yb = self.get_bounds(bbox=False)
            sx = np.median(np.linspace(xb[1], xb[0], pixels))
            ex = np.median(np.linspace(xb[3], xb[2], pixels))
            sy = np.median(np.linspace(yb[1], yb[0], lines))
            ey = np.median(np.linspace(yb[3], yb[2], lines))
            return haversine(sx=sx, ex=ex, sy=sy, ey=ey)

        raise FileError('Unexpected file format')

    def get_bounds(self, bbox=True):
        """
        Gets the file geospatial boundaries

        Returns
        -------
        tuple
            (x0, x1, y0, y1): geospatial limits of the image
        """
        if self.file.name.endswith('.nc'):
            x0 = self.glob_attrs.get('westernmost_longitude')
            x1 = self.glob_attrs.get('easternmost_longitude')
            y0 = self.glob_attrs.get('southernmost_latitude')
            y1 = self.glob_attrs.get('northernmost_latitude')
            return x0, y0, x1, y1

        if self.file.name.endswith('.h5'):
            # Descending sensor
            attrs = self.fmt_attr(path='Geometry_data')
            x00 = attrs.pop('Upper_right_longitude')
            x01 = attrs.pop('Upper_left_longitude')
            x10 = attrs.pop('Lower_right_longitude')
            x11 = attrs.pop('Lower_left_longitude')

            y00 = attrs.pop('Upper_right_latitude')
            y01 = attrs.pop('Upper_left_latitude')
            y10 = attrs.pop('Lower_right_latitude')
            y11 = attrs.pop('Lower_left_latitude')

            xa = x00, x01, x10, x11
            ya = y00, y01, y10, y11
            if bbox is True:
                return np.min(xa), np.min(ya), np.max(xa), np.max(ya)
            return xa, ya
        raise FileError('Unexpected file format')

    def get_flag_names(self, key: str = 'l2_flags'):

        if self.file.suffix == '.nc':
            flag_meanings = self.get_attrs(key=key).pop(key)
            return flag_meanings.pop('flag_meanings').split()

        if self.file.suffix == '.h5':
            flag_meanings = self.get_attrs(key='QA_flag').pop('QA_flag')

            def flag_meaning(flg: str, j: int):
                start = len(f'Bit-{j}) ')
                end = flg.index(': ')
                return flg[start:end]

            return [flag_meaning(flg=flag, j=i)
                    for i, flag in enumerate(flag_meanings.pop('Data_description').split('\n'))
                    if len(flag) > 0]

        raise FileError('Unexpected file format')

    def get_mission_name(self):
        attrs = self.get_attrs()
        platform = instrument = ''

        if 'platform' in attrs.keys():
            platform = attrs.pop('platform')
        if 'Satellite' in attrs.keys():
            platform = re.search(r'\((.*?)\)', attrs.pop('Satellite'))
            platform = platform.group(1) if platform else ''
        if 'instrument' in attrs.keys():
            instrument = attrs.pop('instrument')
        if 'Sensor' in attrs.keys():
            instrument = re.search(r'\((.*?)\)', attrs.pop('Sensor'))
            instrument = instrument.group(1) if instrument else ''

        mission = '/'.join([instrument, platform])
        name = self.file.name.split('.')[0].split('_')[0]
        if mission == '/':
            mission = ''.join(re.findall('[a-z]', name, re.IGNORECASE))
        return mission

    # -----------
    # Date Parser
    # -----------
    def parse_date(self):

        if self.file.name.endswith('.nc'):
            attrs = self.nc_attrs()
        elif self.file.name.endswith('.h5'):
            attrs = self.h5_attrs()
        else:
            raise FileError('Unexpected file format')

        if 'time_coverage_start' in attrs.keys():
            start_date = parse(attrs.pop('time_coverage_start'))
            end_date = parse(attrs.pop('time_coverage_end'))
            return [start_date, end_date]
        if 'period_coverage_start' in attrs.keys():
            start_date = parse(attrs.pop('period_coverage_start'))
            end_date = parse(attrs.pop('period_coverage_end'))
            return [start_date, end_date]
        if 'Scene_start_time' in attrs.keys():
            start_date = parse(attrs.pop('Scene_start_time'))
            end_date = parse(attrs.pop('Scene_end_time'))
            return [start_date, end_date]

    # ------------------
    # Spatial Resolution
    # ------------------
    def spatial_resolution(self):
        if self.file.suffix == '.h5':
            attrs = self.h5_attrs(key='Image_data')
            return attrs.pop('Grid_interval')

        if self.file.suffix == '.nc':
            attrs = self.nc_attrs()
        else:
            raise FileError('Unexpected file format')
        spr = None

        if 'spatialResolution' in attrs.keys():
            spr = attrs.pop('spatialResolution')
        if 'spatial_resolution' in attrs.keys():
            spr = attrs.pop('spatial_resolution')

        unit = ''.join(re.findall('[a-z]', spr, re.IGNORECASE))
        spr = float(spr.strip(unit))
        if unit.lower() == 'km':
            spr *= 1000
        return spr


__all__ = [
    'FileError',
    'IO',
    'L2File',
    'geo_interp',
]
