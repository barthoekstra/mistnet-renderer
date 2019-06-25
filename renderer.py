#!/usr/bin/env python
"""
Renderer of weather radar data for MistNet CNN

@TODO: Implement argparse interface
@TODO: Implement check if dimensions of rendered and stacked files are correct
@TODO: Implement graceful SIGTERM signals
@TODO: Add logger
@TODO: Add test suite

"""
import datetime
import os
import pathlib
import json

import numpy as np
import h5py
from scipy import interpolate, io


class RadarRenderer:

    # MistNet Model Parameters
    range_max = 150000  # in meters
    pixel_dimensions = 600  # width and height of interpolated image in pixels
    padding = 4  # number of pixels to pad interpolated image with on all sides (for the convolutions)

    # Radar metadata
    odim_radar_db = None
    correct_products = {'Germany': ['RHOHV'], 'Netherlands': [], 'United States': []}

    def __init__(self, pvolfile, **kwargs):
        self.pvolfile = pathlib.Path(pvolfile).resolve()
        self.output_type = kwargs.get('output_type', 'npz')
        self.output_file = kwargs.get('output_file', pathlib.Path(self.pvolfile.parent.as_posix() + '/' +
                                                                  self.pvolfile.stem + '.' + self.output_type))
        if not isinstance(self.output_file, pathlib.Path):
            self.output_file = pathlib.Path(self.output_file).resolve()

        self.radar_format = kwargs.get('radar_format', None)
        self.elevations = set()
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
            self.radar = self.load_radar_volume(self._f)
            self.selected_data = self.select_datasets_odim()
            self.render = self.render_to_mistnet()
            sp_data, dp_data = self.stack_interpolated_elevations(self.render)
            self.save_file(sp_data, dp_data)

    def load_odim_file(self, pvolfile):
        return self.load_hdf5_file(pvolfile)

    def load_hdf5_file(self, pvolfile):
        return h5py.File(pvolfile, 'r')

    def load_radar_volume(self, f):
        radar = dict()

        def format_attributes(attributes):
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
                radar['country'] = [radar['country'].strip() for radar in self.odim_radar_db if
                                    radar['odimcode'].strip() == source['NOD']][0]
            else:
                radar['country'] = [radar['country'].strip() for radar in self.odim_radar_db if
                                    radar['wmocode'].strip() == source['WMO']][0]
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
        if len(self.elevations) < len(self.target_elevations):
            raise Exception('Number of available elevations ({}) is lower than the number of target elevations({}).'
                            .format(len(self.elevations), len(self.target_elevations)))

        # Pick elevations closest to target elevations
        picked_elevs = [min(self.elevations, key=lambda x: abs(x - trg_elev)) for trg_elev in self.target_elevations]
        if len(set(picked_elevs)) < len(picked_elevs):
            picked_elevs = self.pick_elevations_iteratively()

        def check_product_availability(products):
            check_sp_products = all(product in products for product in self.target_sp_products)
            if not check_sp_products:
                raise Exception('Some of the target single-pol products ({}) are missing at the elevations ({}) '
                                'closest to the target elevations ({}).'
                                .format(self.target_sp_products, picked_elevs, self.target_elevations))

            check_dp_products = all(product in products for product in self.target_dp_products)
            if not check_dp_products:
                # Check if we can derive dual-pol products from existing products
                if 'ZDR' not in products:
                    if 'DBZH' not in products or 'DBZV' not in products:
                        raise Exception('ZDR is missing and cannot be computed at target elevation: {}.'
                                        .format(elev))
                    else:
                        self.target_dp_products.extend(['DBZH', 'DBZV'])
                        self.target_dp_products = list(set(self.target_dp_products))

                if 'RHOHV' not in products:
                    raise Exception('RHOHV is missing and cannot be computed at target elevation: {}.'.format(elev))

        selected_data = {}

        for elev in picked_elevs:
            if isinstance(self.radar[elev], list):
                for scan in self.radar[elev]:
                    check_product_availability(scan.keys())

                # No exceptions were thrown, so we can assume all data is available in all scans at elevation elev
                if combine_multiple_scans:
                    high_unamb_velocity, lowest_prf = self.pick_best_scans(self.radar[elev])
                    selected_data[elev] = {'DBZH': lowest_prf, 'VRADH': high_unamb_velocity, 'WRADH': high_unamb_velocity,
                                           'RHOHV': lowest_prf}

                    if 'DBZV' in self.target_dp_products:
                        selected_data[elev].update({'DBZV': lowest_prf})
                    else:
                        selected_data[elev].update({'ZDR': lowest_prf})
                else:
                    selected_data[elev] = {product: 0 for product in self.target_sp_products}
                    selected_data[elev].update({product: 0 for product in self.target_dp_products})
            else:
                check_product_availability(self.radar[elev].keys())
                selected_data[elev] = {product: None for product in self.target_sp_products}
                selected_data[elev].update({product: None for product in self.target_dp_products})
                if 'DBZV' in self.target_dp_products:
                    selected_data[elev].pop('ZDR')

        return selected_data

    def pick_elevations_iteratively(self, elevations):
        picked_elevs = []
        available_elevations = elevations

        for trg_elev in self.target_elevations:
            if len(available_elevations) == 0:
                break

            picked_elev = min(available_elevations, key=lambda x: abs(x - trg_elev))
            picked_elevs.append(picked_elev)
            available_elevations.remove(picked_elev)

        return picked_elevs

    def pick_best_scans(self, scans):
        unamb_velocity = [self.calculate_unambiguous_velocities(scan) for scan in scans]
        highest_unamb_velocity = max(unamb_velocity)

        # Check if 2 or more scans have the same highest value of the unambiguous velocity
        if unamb_velocity.count(highest_unamb_velocity) > 1:
            raise Exception(
                'Two or more scans at elevation {} share the same unambiguous velocity. Add one of the scans to '
                'skipped_scans, so only a single scan can be selected based on the highest unambiguous velocity'
                .format(round(scans[0]['elangle'], 2))
            )

        huvi = unamb_velocity.index(highest_unamb_velocity)  # index of scan with highest unambiguous velocity

        lowest_prf = min(scans, key=lambda x: x['highprf'])  # find lowest value of high prf
        lowest_prf_scans = [scan for scan in scans if scan['highprf'] == lowest_prf['highprf']]

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

        :param wavelength: wavelength in cm
        :param highprf: highest prf of a scan
        :param lowprf: lowest prf of a scan
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

            # Calculate depolarization ratio
            if 'DEPOL' in self.target_dp_products:
                pass

            interpolated_volume[elevation] = self.interpolate_elevation(elevation, parsed_products, scan_index=products)

        # With ZDR calculated, we can now remove it from the list of target products
        if 'DBZV' in self.target_dp_products:
            self.target_dp_products.remove('DBZH')
            self.target_dp_products.remove('DBZV')

        return interpolated_volume

    def parse_odim_data(self, elevation, product, index=None):
        e, p, i = elevation, product, index

        if index is None:
            nan = np.logical_or(self.radar[e][p]['values'] == self.radar[e][p]['nodata'],
                                self.radar[e][p]['values'] == self.radar[e][p]['undetect'])

            corrected_data = self.radar[e][p]['values'] * self.radar[e][p]['gain'] + \
                             self.radar[e][p]['offset']
        else:
            nan = np.logical_or(self.radar[e][i][p]['values'] == self.radar[e][i][p]['nodata'],
                                self.radar[e][i][p]['values'] == self.radar[e][i][p]['undetect'])

            corrected_data = self.radar[e][i][p]['values'] * self.radar[e][i][p]['gain'] + self.radar[e][i][p]['offset']

        corrected_data[nan] = np.nan

        return corrected_data

    def correct_with_reflectivity(self, reflectivity, product):
        reflectivity_nan = np.isnan(reflectivity)
        product[reflectivity_nan] = np.nan
        return product

    def interpolate_elevation(self, elevation, products, scan_index=None):
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

        :param meta: radar metadata dictionary
        :param dataset: dataset path in ODIM archive (dataset1, dataset2, etc.)
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

    def stack_interpolated_elevations(self, interpolated_elevations):
        # Determine size of final numpy array
        first_dataset = next(iter(interpolated_elevations))
        nr_elevations = len(interpolated_elevations)
        sp_products = [data for data in self.target_sp_products if data in interpolated_elevations[first_dataset]]
        dp_products = [data for data in self.target_dp_products if data in interpolated_elevations[first_dataset]]
        first_dim_sp = nr_elevations * len(sp_products)
        first_dim_dp = nr_elevations * len(dp_products)

        image_dims = self.pixel_dimensions + 2 * self.padding

        # Prepopulate single-pol and dual-pol arrays
        sp_data = np.zeros((first_dim_sp, image_dims, image_dims))
        dp_data = np.zeros((first_dim_dp, image_dims, image_dims))

        # Fill single-pol array
        i = 0
        for product in sp_products:
            for elevation in interpolated_elevations:
                sp_data[i, :, :] = interpolated_elevations[elevation][product]
                i += 1

        # Fill dual-pol array
        i = 0
        for product in dp_products:
            for elevation in interpolated_elevations:
                dp_data[i, :, :] = interpolated_elevations[elevation][product]
                i += 1

        return sp_data, dp_data

    def save_file(self, rdata, dp):
        if self.output_file.suffix == '.mat':
            io.savemat(self.output_file, {'rdata': rdata, 'dp': dp})
        elif self.output_file.suffix == '.npz':
            np.savez_compressed(self.output_file, rdata=rdata, dp=dp)

        print('Processed: {}'.format(self.pvolfile.name))


if __name__ == "__main__":
    r = RadarRenderer('data/raw/mvol_201503200100_10557.h5', output_file='data/processed/mvol_201503200100_RW.mat')
    # r = RadarRenderer('data/raw/NLHRW_pvol_20181020T2210_NL52.h5', target_elevations=[0.3, 1.2, 2.0, 2.7, 4.5],
    #                   output_file='data/processed/NLHRW_20181020T2210_RW.mat')
