#!/usr/bin/env python
"""
Renderer of weather radar data for MistNet CNN

"""

import datetime
import os
import pathlib
import json
import argparse
import time
import multiprocessing as mp

import numpy as np
import h5py
from scipy import interpolate, io


class RadarRenderer:
    # MistNet Model Parameters
    range_max = 150000  # in meters
    pixel_dimensions = 600  # width and height of interpolated image in pixels
    padding = 4  # number of pixels to pad interpolated image with on all sides (for the convolutions)
    max_elevation_spread = 6  # maximum difference in degrees between lowest and highest scan elevation

    # Radar metadata
    odim_radar_db = None
    correct_products = {
        'Germany': ['URHOHV'],
        'Netherlands': [],
        'United States': []
    }
    standard_target_elevations = {
        'Germany': [0.5, 1.5, 2.5, 3.5, 4.5],
        'Netherlands': [0.3, 1.2, 2.0, 2.7, 4.5],
        'United States': [0.5, 1.5, 2.5, 3.5, 4.5]
    }
    standard_target_sp_products = {
        'Germany': ['DBZH', 'VRADH', 'WRADH'],
        'Netherlands': ['DBZH', 'VRADH', 'WRADH'],
        'United States': ['DBZH', 'VRADH', 'WRADH']
    }
    standard_target_dp_products = {
        'Germany': ['URHOHV', 'ZDR'],
        'Netherlands': ['RHOHV', 'ZDR'],
        'United States': []
    }

    def __init__(self, pvolfile, **kwargs):
        self.pvolfile = pathlib.Path(pvolfile).resolve()
        self.output_type = kwargs.get('output_type', 'npz')
        self.output_file = kwargs.get('output_file', pathlib.Path(self.pvolfile.parent.as_posix() + '/' +
                                                                  self.pvolfile.stem + '.' + self.output_type))
        if not isinstance(self.output_file, pathlib.Path):
            self.output_file = pathlib.Path(self.output_file).resolve()

        self.radar_format = kwargs.get('radar_format', None)
        self.elevations = set()
        self.target_elevations = kwargs.get('target_elevations', None)
        self.skipped_scans = kwargs.get('skipped_scans', [])
        self.target_sp_products = kwargs.get('target_sp_products', None)
        self.target_dp_products = kwargs.get('target_dp_products', None)
        self.skip_render = kwargs.get('skip_render', False)
        self.skip_save = kwargs.get('skip_save', False)

        if self.pvolfile.suffix == '.h5' or self.radar_format == 'ODIM':
            # We assume the file is ODIM formatted

            # Load ODIM database on first instantiation of class with ODIM formatted data
            if self.odim_radar_db is None:
                with open(os.environ['ODIM_DATABASE']) as db:
                    self.odim_radar_db = json.load(db)

            self._f = self.load_odim_file(self.pvolfile)
            self.radar = self.load_radar_volume(self._f)
            self._f.close()

            if self.target_elevations is None:
                self.target_elevations = self.standard_target_elevations[self.radar['country']].copy()

            if self.target_sp_products is None:
                self.target_sp_products = self.standard_target_sp_products[self.radar['country']].copy()

            if self.target_dp_products is None:
                self.target_dp_products = self.standard_target_dp_products[self.radar['country']].copy()

            self.selected_data = self.select_datasets_odim()

            if not self.skip_render:
                self.render = self.render_to_mistnet()
                self.sp_data, self.dp_data = self.stack_interpolated_elevations(self.render)

                if not self.skip_save:
                    self.save_file(self.sp_data, self.dp_data)

    def load_odim_file(self, pvolfile):
        """
        Loads an ODIM formatted weather radar file.

        :param pvolfile: path to ODIM formatted file
        :return: ODIM formatted HDF5 file object
        """
        return self.load_hdf5_file(pvolfile)

    def load_hdf5_file(self, pvolfile):
        """
        Loads an HDF5 weather radar file
        :param pvolfile: path to HDF5 file
        :return: HDF5 file object
        """
        return h5py.File(pvolfile, 'r')

    def load_radar_volume(self, f):
        """
        Loads the radar volume into a dictionary of elevations, corresponding scans, radar products and attributes. Also
        determines the country of the radar, useful to determine processing requirements for the radar data.

        :param f: ODIM formatted HDF5 file object
        :return: dictionary of radar file metadata, organized by elevations
        """
        radar = dict()

        def format_attributes(attributes):
            """
            Converts attributes with bytestrings to normal strings and unpacks data stored in numpy arrays.

            @NOTE: Probably needs significant changes to deal with NEXRAD radar data as more data is stored in
                the HDF5 attributes.

            :param attributes: dictionary of attributes
            :return: re-formatted dictionary of attributes
            """
            attrs = {k: v.decode('UTF-8') if isinstance(v, bytes) else v for k, v in attributes.items()}
            attrs = {k: v[0] if isinstance(v, np.ndarray) else v for k, v in attrs.items()}
            return attrs

        for dataset in f.keys():  # Datasets correspond with scans (mostly elevations)
            if dataset in ['what', 'where', 'how']:  # Base what, where, how groups
                attrs = format_attributes(f[dataset].attrs)
                radar.update(attrs)
            elif dataset in self.skipped_scans:
                continue
            else:
                scan_groups = {}

                for group in f[dataset].keys():
                    if group in ['what', 'where', 'how']:
                        attrs = format_attributes(f[dataset][group].attrs)
                        scan_groups.update(attrs)
                    else:
                        product = format_attributes(f[dataset][group]['what'].attrs)
                        product.update({'values': f[dataset][group]['data'][()]})
                        scan_groups.update({product['quantity']: product})

                elevation = round(scan_groups['elangle'], 2)

                if elevation not in radar.keys():
                    radar[elevation] = scan_groups
                elif isinstance(radar[elevation], list):
                    radar[elevation].append(scan_groups)
                else:
                    radar[elevation] = [radar[elevation]]
                    radar[elevation].append(scan_groups)

                self.elevations.add(elevation)

        f.close()

        # Extract country so we can determine some country-specific processing steps
        source = dict(pair.split(':') for pair in radar['source'].split(','))
        try:
            if 'NOD' in source.keys():
                country = [radar['country'].strip() for radar in self.odim_radar_db if
                           radar['odimcode'].strip() == source['NOD']]
                radar['country'] = country[0]
            else:
                country = [radar['country'].strip() for radar in self.odim_radar_db if
                           radar['wmocode'].strip() == source['WMO']]
                radar['country'] = country[0]
        except KeyError:
            # We are probably dealing with an ODIM conversion of a NEXRAD file, which lacks the NOD and WMO codes
            radar['country'] = 'United States'

        return radar

    def select_datasets_odim(self, combine_multiple_scans=True):
        """
        Picks elevations closest to the given target elevations, validates whether required data is present and picks
        the best scans based on unambiguous velocity and PRF in case more than 1 scan is present at a certain elevation.

        :param combine_multiple_scans: Boolean, True by default, which determines whether best scans should be selected
            based on unambiguous velocities and PRF. If set to False, it will simply return the index of the first sweep
            at this elevation
        :return: dictionary of elevations, their corresponding products and the sweep index to select for a product if
            multiple are present at one elevation level. Dict values are set to None if there is only one sweep per
            elevation and no 'best scan' exists.
        """
        # Check if there are enough elevations in the radar volume
        if len(self.elevations) < len(set(self.target_elevations)):
            raise RadarException('Number of available elevations ({}) is lower than the number of target elevations '
                                 '({}).'.format(len(self.elevations), len(self.target_elevations)))

        # Pick elevations closest to target elevations
        picked_elevs = [min(self.elevations, key=lambda x: abs(x - trg_elev)) for trg_elev in self.target_elevations]
        if len(set(picked_elevs)) < len(picked_elevs):
            picked_elevs = self.pick_elevations_iteratively(self.target_elevations)

        if max(picked_elevs) - min(picked_elevs) > self.max_elevation_spread:
            raise RadarException('Difference (in degrees) between lowest ({}) and highest picked elevation ({}) larger '
                                 'than maximum allowed value: {}.'
                                 .format(min(picked_elevs), max(picked_elevs), self.max_elevation_spread))

        selected_data = {}

        for elev in picked_elevs:
            if isinstance(self.radar[elev], list):
                for scan in self.radar[elev]:
                    self.check_product_availability(scan.keys(), picked_elevs, elev)

                # No exceptions were thrown, so we can assume all data is available in all scans at elevation elev
                if combine_multiple_scans:
                    high_unamb_velocity, lowest_prf = self.pick_best_scans(self.radar[elev])
                    selected_data[elev] = {'DBZH': lowest_prf, 'VRADH': high_unamb_velocity,
                                           'WRADH': high_unamb_velocity, 'RHOHV': lowest_prf}

                    if 'DBZV' in self.target_dp_products:
                        selected_data[elev].update({'DBZV': lowest_prf})
                    else:
                        selected_data[elev].update({'ZDR': lowest_prf})
                else:
                    selected_data[elev] = {product: 0 for product in self.target_sp_products}
                    selected_data[elev].update({product: 0 for product in self.target_dp_products})
            else:
                self.check_product_availability(self.radar[elev].keys(), picked_elevs, elev)
                selected_data[elev] = {product: None for product in self.target_sp_products}
                selected_data[elev].update({product: None for product in self.target_dp_products})

        return selected_data

    def pick_elevations_iteratively(self, target_elevations):
        """
        Picks elevations iteratively rather than through a list comprehension, in case the latter method resulted in two
        or more of the elevations are duplicates.

        :param target_elevations: List of target elevations
        :return: List of iteratively picked elevations closest to target elevations
        """
        picked_elevs = []
        available_elevations = self.elevations

        for trg_elev in target_elevations:
            if len(picked_elevs) == len(target_elevations):
                break

            picked_elev = min(available_elevations, key=lambda x: abs(x - trg_elev))
            picked_elevs.append(picked_elev)
            available_elevations.remove(picked_elev)

        return sorted(picked_elevs)

    def check_product_availability(self, products, picked_elevs, elev):
        """
        Checks if all required products in target_sp_products and target_dp_products exist within all scans in a
        volume. If that is not the case, and these products cannot be computed based on other existing products,
        it raises an exception.

        NOTE: This will also throw an exception if single scans do not contain all the necessary products, but
            combined they do. For now that should do.

        :param products: dictionary keys of products in a scan
        :param picked_elevs: list of picked elevations
        :paral elev: float of current elevation
        :return: True if all target products are present, raises an exception if not
        """
        check_sp_products = all(product in products for product in self.target_sp_products)
        if not check_sp_products:
            raise RadarException('Some of the target single-pol products ({}) are missing at the elevations ({}) '
                                 'closest to the target elevations ({}).'
                                 .format(self.target_sp_products, picked_elevs, self.target_elevations))

        check_dp_products = all(product in products for product in self.target_dp_products)
        if not check_dp_products:
            # Check if we can derive dual-pol products from existing products
            if 'ZDR' in self.target_dp_products and 'ZDR' not in products:
                if 'DBZH' not in products or 'DBZV' not in products:
                    raise RadarException('ZDR is missing and cannot be computed at target elevation: {}.'.format(elev))
                else:
                    self.target_dp_products.extend(['DBZH', 'DBZV'])

                    try:
                        self.target_dp_products.remove('ZDR')
                    except ValueError:
                        pass

                    self.target_dp_products = list(set(self.target_dp_products))

            if 'RHOHV' in self.target_dp_products and 'RHOHV' not in products:
                raise RadarException('RHOHV is missing and cannot be computed at target elevation: {}.'.format(elev))

            # # RHOHV and ZDR are present or can be derived, so we can also derive DPR.
            # if 'DPR' in self.target_dp_products:
            #     self.target_dp_products.remove('DPR')

            targets_dp = self.target_dp_products.copy()

            remove_values = ['ZDR', 'DBZH', 'DBZV', 'RHOHV']
            for value in remove_values:
                try:
                    targets_dp.remove(value)
                except ValueError:
                    pass

            check_dp_products = all(product in products for product in targets_dp)
            if not check_dp_products and len(targets_dp) > 0:
                raise RadarException('Some of the target-dual-pol products ({}) are missing at the elevations ({}) '
                                     'closest to the target elevations ({}).'
                                     .format(self.target_dp_products, picked_elevs, self.target_elevations))
            else:
                return

    def pick_best_scans(self, scans):
        """
        Picks the best scans from multiple scans at the same elevation for different product types. The logic is as
        follows: the scan with the highest unambiguous velocity will result in the most accurate data for VRADH and
        WRADH, whereas the scan with the lowest pulse repetition frequency (PRF) has the longest range and is therefore
        better suited for DBZH and DBZV.

        :param scans: List of N dictionaries corresponding to N number of scans at a certain elevation
        :return: index of the scan with the highest unambiguous velocity and the index of the scan with the lowest PRF
        """
        unamb_velocity = [self.calculate_unambiguous_velocities(scan) for scan in scans]
        highest_unamb_velocity = max(unamb_velocity)

        # Check if 2 or more scans have the same highest value of the unambiguous velocity
        if unamb_velocity.count(highest_unamb_velocity) > 1:
            raise RadarException(
                'Two or more scans at elevation {} share the same unambiguous velocity. Add one of the scans to '
                'skipped_scans, so only a single scan can be selected based on the highest unambiguous velocity'
                    .format(round(scans[0]['elangle'], 2))
            )

        huvi = unamb_velocity.index(highest_unamb_velocity)  # index of scan with highest unambiguous velocity

        lowest_prf = min(scans, key=lambda x: x['highprf'])  # find lowest value of high prf
        lowest_prf_scans = [scan for scan in scans if scan['highprf'] == lowest_prf['highprf']]

        lti = 0
        if len(lowest_prf_scans) > 1:
            """
            Apparently there are multiple scans with the same lowest value for the highprf, so we pick the one that is 
            closest in time to the scan with the highest unambiguous velocity.
            """
            enddates = [datetime.datetime.strptime(scan['enddate'], '%Y%m%d') for scan in scans]
            endtimes = [datetime.datetime.strptime(scan['endtime'], '%H%M%S').time() for scan in scans]
            end_dt = [datetime.datetime.combine(date, time) for date, time in zip(enddates, endtimes)]

            time_between_scans = [(index, abs(end_dt[huvi] - dt)) for index, dt in enumerate(end_dt)]
            time_between_scans.pop(huvi)  # Remove timedelta of 0:00:00
            least_time = min(time_between_scans, key=lambda x: x[1])
            lti = least_time[0]  # index

        return huvi, lti

    def calculate_unambiguous_velocities(self, scan):
        """
        Calculates unambiguous velocity interval following Holleman & Beekhuis (2003).

        Holleman, I., & Beekhuis, H. (2003). Analysis and correction of dual PRF velocity data.
            Journal of Atmospheric and Oceanic Technology, 20(4), 443-453.

        :param scan: dictionary of scan
        :return: unambiguous velocity interval in m/s
        """
        if 'wavelength' in self.radar.keys():
            wavelength = self.radar['wavelength'] / 100
        else:
            wavelength = scan['wavelength'] / 100

        unamb_vel_high = (wavelength * scan['highprf']) / 4
        unamb_vel_low = (wavelength * scan['lowprf']) / 4
        dualprf_unamb_vel = (unamb_vel_high * unamb_vel_low) / (unamb_vel_high - unamb_vel_low)
        return dualprf_unamb_vel

    def render_to_mistnet(self):
        """
        Render the radar volume, consisting of the scans at the target elevations, to the MistNet specifications.

        If ZDR and RHOHV are present, depolarization ratio is calculated using Kilambi et al. (2018).

        Kilambi, A., Fabry, F., & Meunier, V. (2018). A Simple and Effective Method for Separating Meteorological
            from Nonmeteorological Targets Using Dual-Polarization Data.
            Journal of Atmospheric and Oceanic Technology, 35(7), 1415-1424.

        :return: dictionary of interpolated (rendered) radar volume
        """
        interpolated_volume = {}

        for elevation, products in self.selected_data.items():
            parsed_products = {}

            for product, scan_index in products.items():
                if scan_index is None:
                    parsed_products[product] = self.parse_odim_data(elevation, product)
                else:
                    parsed_products[product] = self.parse_odim_data(elevation, product, index=scan_index)

            # Correct parsed products with reflectivity if necessary
            for product in self.correct_products[self.radar['country']]:
                uncorrected = parsed_products[product]
                parsed_products[product] = self.correct_with_reflectivity(parsed_products['DBZH'], uncorrected)

            # Calculate ZDR if DBZV is present
            if 'DBZV' in parsed_products.keys():
                parsed_products['ZDR'] = parsed_products['DBZH'] - parsed_products['DBZV']
                parsed_products.pop('DBZV')
                products.update({'ZDR': products['DBZH']})

            # Calculate depolarization ratio if ZDR and RHOHV are present
            if 'ZDR' in parsed_products.keys() and 'RHOHV' in parsed_products.keys():
                zdr_linear = np.power(10, parsed_products['ZDR'] / 10)
                dpr_linear = (zdr_linear + 1 - 2 * np.sqrt(zdr_linear) * parsed_products['RHOHV']) / \
                             (zdr_linear + 1 + 2 * np.sqrt(zdr_linear) * parsed_products['RHOHV'])

                with np.errstate(invalid='ignore'):
                    # There are NaNs in the input arrays, so we have to ignore those
                    dpr = 10 * np.log10(dpr_linear)

                parsed_products['DPR'] = dpr
                products.update({'DPR': products['DBZH']})

            interpolated_volume[elevation] = self.interpolate_elevation(elevation, parsed_products, scan_index=products)

        # With ZDR calculated, we can now remove DBZH and DBZV from the list of target products
        if 'DBZV' in self.target_dp_products:
            self.target_dp_products.remove('DBZH')
            self.target_dp_products.remove('DBZV')
            self.target_dp_products.append('ZDR')

        # With DPR calculated, we still need to add it to target_dp_products for proper stacking
        if 'DPR' in interpolated_volume[list(self.elevations)[0]].keys():
            self.target_dp_products.append('DPR')

        return interpolated_volume

    def parse_odim_data(self, elevation, product, index=None):
        """
        Parse raw data by correcting them with the calibration offset and gain values.

        :param elevation: float of elevation angle
        :param product: string of product name
        :param index: index of scan to use if there are multiple scans at one elevation
        :return: numpy array with the corrected data
        """
        e, p, i = elevation, product, index

        if index is None:
            nan = np.logical_or(self.radar[e][p]['values'] == self.radar[e][p]['nodata'],
                                self.radar[e][p]['values'] == self.radar[e][p]['undetect'])

            corrected_data = self.radar[e][p]['values'] * self.radar[e][p]['gain'] + self.radar[e][p]['offset']
        else:
            nan = np.logical_or(self.radar[e][i][p]['values'] == self.radar[e][i][p]['nodata'],
                                self.radar[e][i][p]['values'] == self.radar[e][i][p]['undetect'])

            corrected_data = self.radar[e][i][p]['values'] * self.radar[e][i][p]['gain'] + self.radar[e][i][p]['offset']

        corrected_data[nan] = np.nan

        return corrected_data

    def correct_with_reflectivity(self, reflectivity, product):
        """
        Correct product by setting cells of a product to NaN where the reflectivity is NaN.

        :param reflectivity: numpy array of the reflectivity (DBZH) values
        :param product: numpy array of the product values to be corrected
        :return: numpy array of the reflectivity-corrected values
        """
        reflectivity_nan = np.isnan(reflectivity)
        product[reflectivity_nan] = np.nan
        return product

    def interpolate_elevation(self, elevation, products, scan_index=None):
        """
        Interpolates the polar data to a regular grid for MistNet.

        :param elevation: float of the elevation angle
        :param products: dictionary of products
        :param scan_index: dictionary with scan indices to use if more scans per elevation exist
        :return: dictionary of interpolated products
        """
        x = np.linspace(-self.range_max, self.range_max, self.pixel_dimensions)
        y = -x
        X, Y = np.meshgrid(x, y)

        # Convert cartesian to polar coordinates
        def cart2pol(x, y):
            rho = np.sqrt(x ** 2 + y ** 2)
            phi = np.arctan2(y, x)
            return rho, phi

        R, PHI = cart2pol(X, Y)

        def pol2cmp(theta):
            bearing = np.degrees(np.pi / 2 - theta)
            bearing = bearing % 360
            return bearing

        PHI = pol2cmp(PHI)

        # Convert slant range to ground range
        def slant2ground(r, theta):
            earth_radius = 6371200  # from NARR GRIB file
            multiplier = 4 / 3
            r_e = earth_radius * multiplier  # earth effective radius

            theta = np.radians(theta)
            h = np.sqrt(np.power(r, 2) + np.power(r_e, 2) + (2 * r_e * np.multiply(r, np.sin(theta)))) - r_e
            s = r_e * np.arcsin(np.divide(np.multiply(r, np.cos(theta)), (r_e + h)))

            return s

        interpolated_products = {}

        for product_name, product in products.items():
            azimuths, ranges = self.get_azimuths_ranges_odim(elevation, product_name, scan_index)

            maxrange_mask = R > max(ranges)

            groundranges = slant2ground(ranges, elevation)

            aa, rr = np.meshgrid(groundranges, azimuths)

            i = interpolate.griddata((aa.ravel(), rr.ravel()), product.ravel(), (R, PHI), method='nearest')

            # Now remove nearest neighbour interpolation outside of maxrange
            i[maxrange_mask] = np.nan

            i = np.pad(i, self.padding, mode='constant', constant_values=np.nan)
            interpolated_products[product_name] = i

        return interpolated_products

    def get_azimuths_ranges_odim(self, elevation, product_name=None, scan_index=None):
        """
        Generates 1D arrays containing azimuths in degrees and range values in meters for every rangebin along a ray

        :param elevation: float of the elevation angle
        :param product_name: string of the product name (e.g. DBZH)
        :param scan_index: index of scan, applicable if there are multiple scans at one elevation
        :return: 1D array for azimuths in degrees and ranges in meters
        """
        if scan_index[product_name] is None:
            r = self.radar[elevation]['rstart'] + self.radar[elevation]['rscale'] * \
                np.arange(0, self.radar[elevation]['nbins'])
            az = np.arange(0, 360, 360 / self.radar[elevation]['nrays'])
        else:
            index = scan_index[product_name]
            r = self.radar[elevation][index]['rstart'] + self.radar[elevation][index]['rscale'] * \
                np.arange(0, self.radar[elevation][index]['nbins'])
            az = np.arange(0, 360, 360 / self.radar[elevation][index]['nrays'])

        return az, r

    def stack_interpolated_elevations(self, interpolated_volume):
        """
        Stacks the interpolated elevations to the format expected by the MistNet model, which - as of writing - is a
        15x608x608 numpy array containing for DBZH, VRADH and WRADH and a 10x608x608 numpy array for RHOHV and ZDR.

        :param interpolated_volume: dictionary of interpolated volume organized by elevation angle
        :return: numpy arrays of stacked single-pol and dual-pol products
        """
        # Determine size of final numpy array
        first_dataset = next(iter(interpolated_volume))
        nr_elevations = len(interpolated_volume)
        sp_products = [data for data in self.target_sp_products if data in interpolated_volume[first_dataset].keys()]
        dp_products = [data for data in self.target_dp_products if data in interpolated_volume[first_dataset].keys()]
        first_dim_sp = nr_elevations * len(sp_products)
        first_dim_dp = nr_elevations * len(dp_products)

        image_dims = self.pixel_dimensions + 2 * self.padding

        # Prepopulate single-pol and dual-pol arrays
        sp_data = np.zeros((first_dim_sp, image_dims, image_dims))
        dp_data = np.zeros((first_dim_dp, image_dims, image_dims))

        # Fill single-pol array
        i = 0
        for product in sp_products:
            for elevation in interpolated_volume:
                sp_data[i, :, :] = interpolated_volume[elevation][product]
                i += 1

        # Fill dual-pol array
        i = 0
        for product in dp_products:
            for elevation in interpolated_volume:
                dp_data[i, :, :] = interpolated_volume[elevation][product]
                i += 1

        return sp_data, dp_data

    def save_file(self, rdata, dp):
        """
        Save stacked single-pol and dual-pol products to the format expected by the MistNet model.

        :param rdata: numpy array of stacked single-pol data
        :param dp: numpy array of stacked dual-pol data
        """
        if self.output_file.suffix == '.mat':
            io.savemat(self.output_file, {'rdata': rdata, 'dp': dp})
        elif self.output_file.suffix == '.npz':
            np.savez_compressed(self.output_file, rdata=rdata, dp=dp)

        print('Processed: {}'.format(self.pvolfile.name))


class RadarException(Exception):
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='Path to input file or folder. If given a folder, all files within are '
                                      'processed.', type=str)
    parser.add_argument('output', help='Path to output file or folder. If given a folder, all processed files are '
                                       'stored there.', type=str)
    parser.add_argument('-e', '-elevations', nargs=5, help='Five numbers separated by a space indicating the target'
                                                           'elevations to render to MistNet. Works only when a single'
                                                           'file is to be processed.', type=float)
    parser.add_argument('-t', '-type', help='Output file types.', choices=['npz', 'mat'], type=str, default='npz')
    parser.add_argument('-c', '-cores', help='Cores to use for parallel processing. Defaults to number of available'
                                             'cores minus 2.', choices=range(1, mp.cpu_count()+1),
                        default=mp.cpu_count() - 2, type=int)
    args = parser.parse_args()

    input_path = pathlib.Path(args.input).resolve()
    output_path = pathlib.Path(args.output).resolve()

    if input_path.is_file() and not output_path.is_dir():
        RadarRenderer(input_path, output_file=output_path, target_elevations=args.e, output_type=args.t)

    elif input_path.is_file() and output_path.is_dir():
        output_file = pathlib.Path(output_path.as_posix() + '/' + input_path.stem + '.' + args.t)
        RadarRenderer(input_path, output_file=output_file, target_elevations=args.e)

    elif input_path.is_dir() and output_path.is_file():
        print('Output location should be a folder. Exiting now.')
    else:
        # Apparently both input and output are folders
        start = time.time()

        rendered_files = []
        raw_files = []

        for file in output_path.glob('*'):
            if file.suffix == '.' + args.t:
                rendered_files.append(file.stem)

        for file in input_path.glob('*'):
            if file.suffix == '.h5':
                raw_files.append(file)

        unprocessed_files = [file for file in raw_files if file.stem not in rendered_files]

        print('Files left to process: {}'.format(len(unprocessed_files)))

        def render_radar_file(file):
            output_file = pathlib.Path(output_path.as_posix() + '/' + file.stem + '.' + args.t)
            try:
                RadarRenderer(file, output_file=output_file, target_elevations=args.e)
            except (RadarException, OSError) as e:
                print('Problem encountered while processing {}: {}'.format(file.stem, e))


        with mp.Pool(processes=args.c) as pool:
            pool.map(render_radar_file, unprocessed_files)

        end = time.time()

        print('Processed {} files in {} seconds.'.format(len(unprocessed_files), end - start))
