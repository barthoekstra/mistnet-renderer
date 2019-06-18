#!/usr/bin/env python

"""
Renderer of radar data for MistNet CNN

Author: Bart Hoekstra
"""
import os
import pathlib
import json
from datetime import datetime

import h5py
import numpy as np
from scipy import interpolate, io


class RadarRenderer:

    # MistNet Model parameters
    range_max = 150000  # in meters
    pixel_dimensions = 600  # width and height of interpolated image in pixels
    padding = 4  # number of pixels to pad interpolated image with on all sides (for the convolutions)

    # Radar metadata
    odim_radar_db = None
    correct_products = {'Germany': ['RHOHV'], 'Netherlands': [], 'United States': []}

    def __init__(self, pvolfile, **kwargs):
        self.pvolfile = pathlib.Path(pvolfile).resolve()
        self.output_type = kwargs.get('output_type', 'npz')pycharmgitkrake

        self.output_file = kwargs.get('output_file', pathlib.Path(self.pvolfile.parent.as_posix() + '/' +
                                                                  self.pvolfile.stem + '.' + self.output_type))
        if not isinstance(self.output_file, pathlib.Path):
            self.output_file = pathlib.Path(self.output_file).resolve()

        self.radar_format = kwargs.get('radar_format', None)
        self.target_elevations = kwargs.get('target_elevations', [0.5, 1.5, 2.5, 3.5, 4.5])
        self.skipped_scans = kwargs.get('skipped_scans', [])
        self.target_sp_products = kwargs.get('target_sp_products', ['DBZH', 'VRADH', 'WRADH'])
        self.target_dp_products = kwargs.get('target_dp_products', ['RHOHV', 'ZDR'])

        if self.pvolfile.suffix == '.h5' or self.radar_format == 'ODIM':
            # We assume the file is ODIM formatted

            # Load ODIM database on first instantiation of class with ODIM formatted data
            if self.odim_radar_db is None:
                with open(os.environ['ODIM_DATABASE']) as db:
                    self.odim_radar_db = json.load(db)

            self._f = self.load_odim_file(self.pvolfile)
            self.meta = self.extract_odim_metadata(self._f)

            try:
                self.picked_datasets = self.select_datasets_odim()
            except ValueError as e:
                print('{}: {}'.format(pvolfile, e))
                return

            interpolated_datasets = {}

            for elevation, datasets in self.picked_datasets.items():
                selected_data_sp = {product: {'dataset': dataset, 'data': self.meta[dataset]['products'][product]}
                                    for product, dataset in datasets.items()
                                    if product in self.target_sp_products}
                selected_data_dp = {product: {'dataset': dataset, 'data': self.meta[dataset]['products'][product]}
                                    for product, dataset in datasets.items()
                                    if product in self.target_dp_products}

                selected_data = {}
                selected_data.update(selected_data_sp)
                selected_data.update(selected_data_dp)

                products = {}

                for product, data_location in selected_data.items():
                    data, _ = self.parse_odim_data(data_location['dataset'], data_location['data'])
                    products[product] = {'dataset': data_location['dataset'], 'data': data}

                products = self.correct_with_reflectivity(products, self.correct_products[self.meta['country']])

                # Calculate ZDR if DBZV is present and remove DBZV after doing so
                if 'DBZV' in products.keys():
                    products['ZDR'] = {'data': products['DBZH']['data'] - products['DBZV']['data'],
                                       'dataset': products['DBZH']['dataset']}
                    products.pop('DBZV')

                interpolated_data = self.interpolate_scan_products(products=products, elevation=elevation)

                interpolated_datasets[elevation] = interpolated_data

            if 'DBZH' in self.target_dp_products:
                self.target_dp_products.remove('DBZH')
                self.target_dp_products.remove('DBZV')
                self.target_dp_products.extend(['ZDR'])

            sp_data, dp_data = self.stack_interpolated_datasets(interpolated_datasets)

            self.save_file(rdata=sp_data, dp=dp_data)

        elif self.pvolfile.suffix == '' or self.radar_format == 'NEXRAD':
            self.load_nexrad_file(self.pvolfile)
        else:
            raise Exception('Could not infer radar data format from filename. Set type to ODIM or NEXRAD')

    def load_odim_file(self, pvolfile):
        return self.load_hdf5_file(pvolfile)

    def load_nexrad_file(self, pvolfile):
        raise NotImplementedError

    def load_hdf5_file(self, pvolfile):
        return h5py.File(self.pvolfile, 'r')

    def extract_odim_metadata(self, f):
        meta = {}

        # Match country with the ODIM radars database
        source = dict(pair.split(':') for pair in f['what'].attrs.get('source').decode('UTF-8').split(','))
        try:
            if 'NOD' in source.keys():
                country = [radar['country'].strip() for radar in self.odim_radar_db if
                           radar['odimcode'].strip() == source['NOD']]
            elif 'WMO' in source.keys():
                country = [radar['country'].strip() for radar in self.odim_radar_db if
                           radar['wmocode'].strip() == source['WMO']]
        except KeyError:
            # We are probably dealing with an ODIM conversion of a NEXRAD file, which lacks the NOD and WMO codes
            country = ['United States']

        meta['country'] = country[0]
        meta['wavelength'] = f['how'].attrs.get('wavelength')

        # Loop over datasets to extract products and corresponding elevations
        for dataset in f:
            if not f[dataset].name.startswith('/dataset'):
                continue

            if meta['wavelength'] is None:
                meta['wavelength'] = f[dataset]['how'].attrs.get('wavelength')

            meta[dataset] = {}

            meta[dataset]['elevation'] = f[dataset]['where'].attrs.get('elangle')
            meta[dataset]['elevation'] = f[dataset]['where'].attrs.get('elangle')
            meta[dataset]['range_start'] = f[dataset]['where'].attrs.get('rstart')
            meta[dataset]['range_scale'] = f[dataset]['where'].attrs.get('rscale')
            meta[dataset]['range_bins'] = f[dataset]['where'].attrs.get('nbins')
            meta[dataset]['azim_bins'] = f[dataset]['where'].attrs.get('nrays')
            meta[dataset]['prf_high'] = f[dataset]['how'].attrs.get('highprf')
            meta[dataset]['prf_low'] = f[dataset]['how'].attrs.get('lowprf')
            dt_start = '{}T{}'.format(f[dataset]['what'].attrs.get('startdate').decode('UTF-8'),
                                      f[dataset]['what'].attrs.get('starttime').decode('UTF-8'))
            dt_end = '{}T{}'.format(f[dataset]['what'].attrs.get('enddate').decode('UTF-8'),
                                    f[dataset]['what'].attrs.get('endtime').decode('UTF-8'))
            meta[dataset]['dt_start'] = datetime.strptime(dt_start, '%Y%m%dT%H%M%S')
            meta[dataset]['dt_end'] = datetime.strptime(dt_end, '%Y%m%dT%H%M%S')

            meta[dataset] = {k: v[0] if type(v) is np.ndarray else v for k, v in
                             meta[dataset].items()}  # unpack entirely

            meta[dataset]['products'] = {}

            for data in f[dataset]:

                if not data.startswith('data'):
                    continue

                quantity = f[dataset][data]['what'].attrs.get('quantity').decode('UTF-8')
                meta[dataset]['products'][quantity] = data

        return meta

    def select_datasets_odim(self, select_best_scans=True):
        """
        Scans closest to trg_elevs are picked. Throws an exception if these do not contain target_sp_products and
        target_dp_products or if target_dp_products cannot be derived from other existing products.

        :param trg_elevs: list of target elevations
        :param meta: radar metadata dictionary
        :param select_best_scans: True (default) if best scan from multiple scans at the same elevation should be picked
            based on highest unambiguous velocity interval (for VRADH and WRADH) and lowest PRF (for DBZH).
        :return: names of datasets containing scans closest to trg_elevs
        """
        # Extract elevations from meta dictionary
        elevations = [dataset['elevation'] for key, dataset in self.meta.items() if key.startswith('dataset')]
        elevations = [elevation[0] if type(elevation) is np.ndarray else elevation for elevation in elevations]
        uniq_elevations = list(sorted(set(elevations)))

        # trg_elevs = trg_elevs[meta['country']]
        trg_elevs = self.target_elevations
        if len(uniq_elevations) < len(trg_elevs):
            raise ValueError('Number of available elevations ({}) lower than number of target elevations ({}).'
                             .format(len(elevations), len(trg_elevs)))

        # Pick elevations closest to trg_elevs
        picked_elevs = [min(uniq_elevations, key=lambda x: abs(x - trg_elev)) for trg_elev in trg_elevs]
        if len(set(picked_elevs)) < len(picked_elevs):
            picked_elevs = self.pick_elevations_iteratively(uniq_elevations)

        picked_datasets = {}

        # Find datasets that contain picked_elevs
        for picked_elev in picked_elevs:
            for key, value in self.meta.items():
                if not key.startswith('dataset'):
                    continue

                # Sometimes scans are done at the same elevation with different PRFs, some of which we may filter out
                if key in self.skipped_scans:
                    continue

                # Check if dataset is recorded at picked_elev
                if not value['elevation'] == picked_elev:
                    continue

                # Check if single-pol products are available
                check_sp_products = all(product in value['products'] for product in self.target_sp_products)
                if not check_sp_products:
                    raise Exception('Some of the target single-pol products ({}) are missing at the elevations ({}). '
                                    'closest to the target elevations ({}).'
                                    .format(self.target_sp_products, picked_elevs, self.target_elevations))

                # Check if dual-pol products are available
                check_dp_products = all(product in value['products'] for product in self.target_dp_products)
                if not check_dp_products:

                    # Check if we can derive dual-pol products from existing products
                    if 'ZDR' not in value['products']:
                        if 'DBZH' not in value['products'] or 'DBZV' not in value['products']:
                            raise Exception('ZDR is missing and cannot be computed at target elevation: {}'.
                                            format(picked_elev))
                        else:
                            self.target_dp_products.extend(['DBZH', 'DBZV'])
                            self.target_dp_products.remove('ZDR')

                    # Check if RHOHV is missing, which we cannot compute from existing products
                    if 'RHOHV' not in value['products']:
                        raise Exception('RHOHV is missing and cannot be computed at target elevation: {}'.
                                        format(picked_elev))

                dataset = {product: key for product in self.target_sp_products}
                dataset.update({product: key for product in self.target_dp_products})
                dataset['prf_high'] = value['prf_high']
                dataset['prf_low'] = value['prf_low']
                dataset['dt_end'] = value['dt_end']

                if value['elevation'] in picked_datasets.keys():
                    picked_datasets[value['elevation']].extend([dataset])
                else:
                    picked_datasets[value['elevation']] = [dataset]

        if select_best_scans:
            picked_datasets = self.pick_best_scans(picked_datasets)

        # Now flatten/unpack datasets dictionary
        picked_datasets = {elevation: datasets[0] for elevation, datasets in picked_datasets.items()}

        return picked_datasets

    def pick_elevations_iteratively(self, elevations):
        picked_elevs = []

        for trg_elev in self.target_elevations:
            if len(elevations) == 0:
                break

            picked_elev = min(elevations, key=lambda x: abs(x - trg_elev))
            picked_elevs.append(picked_elev)
            elevations.remove(picked_elev)

        return picked_elevs

    def pick_best_scans(self, datasets):
        """
        Loops over selected datasets and selects the right datasets for the products based on unambiguous velocities and
        time. It starts by selecting the scans with the highest unambiguous velocity interval for VRADH and WRADH data and
        subsequently picks the scan with the lowest prf (and the least range folding) for DBZH data.

        NOTE: Although probably unnecessary, an exception is raised when there are two scans at the same elevation with the
            same highest value of the unambiguous velocity. Future improvement with a nearest-neighbour lookup with the
            unambiguous velocity values and time?

        :param meta: radar metadata dictionary
        :param datasets: picked_datasets dictionary
        :return: best datasets for each elevation based on unambiguous velocities and time of scan
        """
        for elevation, scans in datasets.items():
            # Check if there are multiple scans at a given elevation
            if len(scans) > 1:
                # Now calculate the unambiguous velocity (interval) for all scans at this elevation
                unamb_velocity = [self.calculate_unambiguous_velocities(self.meta['wavelength'], scan['prf_high'],
                                                                        scan['prf_low']) for scan in scans]

                highest_unamb_velocity = max(unamb_velocity)

                # Check if 2 or more scans have the same highest value of the unambiguous velocity
                if unamb_velocity.count(highest_unamb_velocity) > 1:
                    raise Exception(
                        'Two or more scans at elevation {} share the same unambiguous velocity. Add one of the '
                        'scans to skipped_scans, so only a single scan can be selected based on the '
                        'highest unambiguous velocity.'.format(elevation))

                hi = unamb_velocity.index(highest_unamb_velocity)  # index of highest
                highest_unamb_velocity_dataset = scans[hi]['VRADH']

                scan_with_lowest_prf = min(scans, key=lambda x: x['prf_high'])  # find lowest value of the high prf
                lowest_prf_scans = [scan for scan in scans if scan['prf_high'] == scan_with_lowest_prf['prf_high']]

                if len(lowest_prf_scans) > 1:
                    time_between_scans = [abs(scans[hi]['dt_end'] - lprf_scan['dt_end']) for lprf_scan in
                                          lowest_prf_scans]
                    least_time = min(time_between_scans)
                    lti = time_between_scans.index(
                        least_time)  # index of scan nearest in time to highest_unamb_velocity
                    lowest_prf_dataset = lowest_prf_scans[lti]['DBZH']
                else:
                    lowest_prf_dataset = lowest_prf_scans[0]['DBZH']

                # Update datasets
                old = datasets[elevation][0]
                datasets[elevation] = [{}]
                datasets[elevation][0]['DBZH'] = lowest_prf_dataset
                datasets[elevation][0]['VRADH'] = highest_unamb_velocity_dataset
                datasets[elevation][0]['WRADH'] = highest_unamb_velocity_dataset
                datasets[elevation][0]['prf_high'] = None
                datasets[elevation][0]['prf_low'] = None
                datasets[elevation][0]['dt_end'] = None
                datasets[elevation][0].update(
                    {key: value for key, value in old.items() if key not in datasets[elevation][0]})

                if 'DBZV' in datasets[elevation][0].keys():
                    datasets[elevation][0]['DBZV'] = lowest_prf_dataset

        return datasets

    def calculate_unambiguous_velocities(self, wavelength, highprf, lowprf):
        """
        Calculates unambiguous velocity interval following Holleman & Beekhuis (2003).

        Holleman, I., & Beekhuis, H. (2003). Analysis and correction of dual PRF velocity data.
            Journal of Atmospheric and Oceanic Technology, 20(4), 443-453.

        :param wavelength: wavelength in cm
        :param highprf: highest prf of a scan
        :param lowprf: lowest prf of a scan
        :return: unambiguous velocity interval in m/s
        """
        wavelength = wavelength / 100
        unamb_vel_high = (wavelength * highprf) / 4
        unamb_vel_low = (wavelength * lowprf) / 4
        dualprf_unamb_vel = (unamb_vel_high * unamb_vel_low) / (unamb_vel_high - unamb_vel_low)
        return dualprf_unamb_vel

    def parse_odim_data(self, dataset, product):
        """
        Converts stored raw data by applying the calibration formula

        :param f: h5py File Object of ODIM formatted radar archive
        :param dataset: dataset path in ODIM archive (dataset1, dataset2, etc.)
        :param product: product identifier (data1, data2, etc.)
        :return: nparray of corrected data and dict of calibration formula elements
        """
        calibration = {}

        calibration['gain'] = self._f[dataset][product]['what'].attrs.get('gain')
        calibration['offset'] = self._f[dataset][product]['what'].attrs.get('offset')
        calibration['nodata'] = self._f[dataset][product]['what'].attrs.get('nodata')
        calibration['undetect'] = self._f[dataset][product]['what'].attrs.get('undetect')

        calibration = {k: v[0] if type(v) is np.ndarray else v for k, v in calibration.items()}  # unpack entirely

        raw_data = self._f[dataset][product]['data'][()]

        missing = np.logical_or(raw_data == calibration['nodata'], raw_data == calibration['undetect'])

        corrected_data = raw_data * calibration['gain'] + calibration['offset']
        corrected_data[missing] = np.nan

        return corrected_data, calibration

    def correct_with_reflectivity(self, products, products_to_correct):
        """
        For some countries certain products contain non-zero/non-nan values where they should in fact be set to np.nan. In
        these cases we set these products to np.nan where DBZH values are np.nan.

        :param products: products dictionary
        :param products_to_correct: list of product names to correct with reflectivity
        :return: products dictionary with corrected values
        """
        DBZH_nan = np.isnan(products['DBZH']['data'])

        for productname, product in products.items():
            if productname != 'DBZH' and productname in products_to_correct:
                products[productname]['data'][DBZH_nan] = np.nan

        return products

    def interpolate_scan_products(self, products, elevation):
        """
        Interpolates all provided products belonging to a single scan at once, so coordinates are calculated just once.
        Based on radar2mat code from the WSRLIB package:
        Sheldon, Daniel. WSRLIB: MATLAB Toolbox for Weather Surveillance Radar. http://bitbucket.org/dsheldon/wsrlib, 2015.

        NOTE: If implemented with NEXRAD radar support: make sure azimuths are always sorted.

        NOTE: If r_max > max range of radar product, all values are set to np.nan. This causes a hard cut-off. Are there
            better solutions?

        :param meta: radar metadata dictionary
        :param products: dictionary of products with names as keys
        :param elevation: elevation in degrees
        :param r_max: maximum range of the interpolation in meters
        :param pix_dims: width and height dimension of the interpolated image
        :return: dictionary of interpolated products
        """
        # Determine query points
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

            azimuths, ranges = self.get_azimuths_ranges_odim(product['dataset'])

            maxrange = self.meta[product['dataset']]['range_scale'] * self.meta[product['dataset']]['range_bins']
            maxrange_mask = R > maxrange

            groundranges = slant2ground(ranges, elevation)

            aa, rr = np.meshgrid(groundranges, azimuths)

            i = interpolate.griddata((aa.ravel(), rr.ravel()), product['data'].ravel(), (R, PHI), method='nearest')

            # Now remove nearest neighbour interpolation outside of product_maxrange
            i[maxrange_mask] = np.nan

            i = np.pad(i, self.padding, mode='constant', constant_values=np.nan)
            interpolated_products[product_name] = i

        return interpolated_products

    def get_azimuths_ranges_odim(self, dataset):
        """
        Generates 1D arrays containing azimuths in degrees and range values in meters for every rangebin along a ray

        :param meta: radar metadata dictionary
        :param dataset: dataset path in ODIM archive (dataset1, dataset2, etc.)
        :return: 1D array for azimuths in degrees and ranges in meters
        """
        r = self.meta[dataset]['range_start'] + self.meta[dataset]['range_scale'] * \
            np.arange(0, self.meta[dataset]['range_bins'])
        az = np.arange(0, 360, 360 / self.meta[dataset]['azim_bins'])

        return az, r

    def stack_interpolated_datasets(self, interpolated_datasets):
        """
        Stacks interpolated datasets in a multidimensional array of size: number of elevations * number of products x
        image pixel dimensions horizontal x image pixel dimensions vertical.

        By default the interpolated datasets will be stacked to a 15 x 608 x 608 dataset, with the first dimension organized
        as follows: 1:5 DBZH, 6:10 VRADH, 11:15 WRADH. Each of these layers contains a 608 x 608 interpolated dataset.

        Stacking order is determined by the order of elements in target_sp_products and target_dp_products

        :param interpolated_datasets: dictionary of interpolated products with product names as keys
        :param target_sp_products: list of target single-pol products in the stacking order
        :param target_dp_products: list of target dual-pol products in the stacking order
        :return: stacked single-pol products (sp_data) and stacked dual-pol products (dp_data)
        """
        # Determine size of final numpy array
        first_dataset = next(iter(interpolated_datasets))  # Assuming all datasets are the same dimensions
        nr_elevations = len(interpolated_datasets)
        sp_products = [data for data in self.target_sp_products if data in interpolated_datasets[first_dataset]]
        dp_products = [data for data in self.target_dp_products if data in interpolated_datasets[first_dataset]]
        first_dim_sp = nr_elevations * len(sp_products)
        first_dim_dp = nr_elevations * len(dp_products)

        image_dims = self.pixel_dimensions + 2 * self.padding

        # Prepopulate single-pol and dual-pol arrays
        sp_data = np.zeros((first_dim_sp, image_dims, image_dims))
        dp_data = np.zeros((first_dim_dp, image_dims, image_dims))

        # Fill single-pol array
        i = 0
        for product in sp_products:
            for dataset in interpolated_datasets:
                sp_data[i, :, :] = interpolated_datasets[dataset][product]
                i += 1

        # Fill dual-pol array
        i = 0
        for product in dp_products:
            for dataset in interpolated_datasets:
                dp_data[i, :, :] = interpolated_datasets[dataset][product]
                i += 1

        return sp_data, dp_data

    def save_file(self, rdata, dp):
        """
        Save file as .mat or .npz file containing single-pol (rdata) and dual-pol (dp) radar products

        :param output_file: name to save file as; the extension determines file type (either .mat or .npz)
        :param rdata: stacked single-pol data
        :param dp: stacked dual-pol data
        """

        if self.output_file.suffix == '.mat':
            io.savemat(self.output_file, {'rdata': rdata, 'dp': dp})
        elif self.output_file.suffix == '.npz':
            np.savez_compressed(self.output_file, rdata=rdata, dp=dp)

        print('Processed: {}'.format(self.pvolfile))


if __name__ == "__main__":
    # r = RadarRenderer('data/raw/mvol_201503200100_10557.h5', output_file='data/raw/mvol_201503200100_10557.h5')
    r = RadarRenderer('data/raw/NLHRW_pvol_20180429T2330_NL52.h5',
                      target_elevations=[0.3, 1.2, 2.0, 2.7, 4.5],
                      output_file='data/processed/NLHRW_pvol_20180429T2330_NL52.mat')

