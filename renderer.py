#!/usr/bin/env python

"""
Renderer of radar data for MistNet

@TODO: Implement argparse interface
@TODO: Implement check if dimensions of generated/stacked numpy array are what they should be
@TODO: Implement scan selector for multiple scans at the same elevation and different PRFs
@TODO: Implement filter that selects only data where DBZH value (for ODIM data) is non-zero/non-NaN
@TODO: Implement graceful SIGTERM signals

"""
import os
import json
import time
import pathlib
import warnings
from multiprocessing import Pool

import h5py
import numpy as np
from scipy import interpolate, io

warnings.filterwarnings('ignore')

# Model parameters (fixed)
range_max = 150000  # in meters
pixel_dimensions = 600  # width of interpolated image in pixels
padding = 4  # number of pixels to pad interpolated image with on all sides

target_elevations = {'Germany': [0.5, 1.5, 2.5, 3.5, 4.5],
                     'Netherlands': [0.3, 1.2, 2.0, 2.7, 4.5],
                     'United States': [0.5, 1.5, 2.5, 3.5, 4.5]}
target_sp_products = {'Germany': ['DBZH', 'VRADH', 'WRADH'],
                      'Netherlands': ['DBZH', 'VRADH', 'WRADH'],
                      'United States': ['DBZH', 'VRADH', 'WRADH']}
target_dp_products = {'Germany': ['RHOHV', 'ZDR'],
                      'Netherlands': ['RHOHV', 'ZDR'],
                      'United States': []}
skipped_scans = {'Germany': [],
                 'Netherlands': ['dataset7', 'dataset16'],
                 'United States': []}

with open(os.environ['ODIM_DATABASE']) as json_file:
    radars_meta_db = json.load(json_file)


def render_radar_file(file, radartype=None, output_file=None, output_type=None):
    file = pathlib.Path(file).resolve()

    if output_type is None:
        output_type = 'npz'

    if output_file is None:
        output_file = pathlib.Path(file.parent.as_posix() + '/' + file.stem + '.' + output_type)

    if file.suffix == '.h5' or radartype == 'ODIM':
        # We assume it's an ODIM formatted HDF5 file
        odimfile, meta = load_odim_file(file)

        if odimfile is None:
            # Some error reading the file occurred.
            return

        try:
            selected_datasets = select_datasets_odim(target_elevations, meta)
        except ValueError as e:
            print('{}: {}'.format(file.name, e))
            return

        meta['selected_datasets'] = selected_datasets

        interpolated_datasets = {}

        for dataset in meta['selected_datasets']:
            selected_data_sp = {key: value for key, value in meta[dataset]['products'].items()
                                if key in target_sp_products[meta['country']]}
            selected_data_dp = {key: value for key, value in meta[dataset]['products'].items()
                                if key in target_dp_products[meta['country']]}

            selected_data = {}
            selected_data.update(selected_data_sp)
            selected_data.update(selected_data_dp)

            products = {}

            for productname, product in selected_data.items():
                data, calib = parse_odim_data(odimfile, dataset, product)
                products[productname] = data

            # Calculate ZDR if DBZV is present and remove DBZV after doing so
            if 'DBZV' in products.keys():
                products['ZDR'] = products['DBZH'] - products['DBZV']
                products.pop('DBZV')
                target_dp_products[meta['country']].remove('DBZH')
                target_dp_products[meta['country']].remove('DBZV')

            azimuths, ranges = get_azimuths_ranges_odim(meta, dataset)

            interpolated_data = interpolate_scan_products(products=products, azimuths=azimuths, ranges=ranges,
                                                          elevation=meta[dataset]['elevation'], r_max=range_max,
                                                          pix_dims=pixel_dimensions, padding=padding)

            interpolated_datasets[dataset] = interpolated_data

        sp_data, dp_data = stack_interpolated_datasets(interpolated_datasets, target_sp_products[meta['country']],
                                                       target_dp_products[meta['country']])

        save_file(output_file=output_file, rdata=sp_data, dp=dp_data)

    elif file.suffix == '' or radartype == 'NEXRAD':
        # We assume it's a NEXRAD file
        load_nexrad_file(file)
    else:
        raise Exception('Could not infer radar data format from filename. Set type to ODIM or NEXRAD.')


def load_odim_file(file):
    """
    Loading an ODIM file creates a h5py File object to access the archive contents, scans through the available
    datasets and populates a metadata dictionary with corresponding products and elevations.

    :param file: radar archive filepath
    :return: h5py File object and metadata dict
    """
    f, meta = None, None

    try:
        f = h5py.File(file, 'r')
        meta = extract_odim_metadata(f)
    except OSError as e:
        print('Error opening h5 archive: {}'.format(e))

    return f, meta


def extract_odim_metadata(f):
    """
    Available elevations and ordering of the datasets in the ODIM h5 files differs from country to country, so we first
    have to make an inventory of available elevations and datasets. This function returns a dict containing datasets and
    their corresponding elevations and available products. Furthermore it matches the radar wmocode with the ODIM radar
    database to extract the country the radar is located in.

    :param f: h5py File Object of ODIM file
    :return: radar metadata dictionary
    """
    meta = {}

    # Match country with the ODIM radars database
    source = dict(pair.split(':') for pair in f['what'].attrs.get('source').decode('UTF-8').split(','))
    try:
        if 'NOD' in source.keys():
            country = [radar['country'].strip() for radar in radars_meta_db if
                       radar['odimcode'].strip() == source['NOD']]
        elif 'WMO' in source.keys():
            country = [radar['country'].strip() for radar in radars_meta_db if
                       radar['wmocode'].strip() == source['WMO']]
    except KeyError:
        # We are probably dealing with an ODIM conversion of a NEXRAD file, which lacks the NOD and WMO codes
        country = ['United States']
    else:
        country = ['United States']

    meta['country'] = country[0]

    # Loop over datasets to extract products and corresponding elevations
    for dataset in f:
        meta[dataset] = {}

        if not f[dataset].name.startswith('/dataset'):
            continue

        meta[dataset]['elevation'] = f[dataset]['where'].attrs.get('elangle')
        meta[dataset]['elevation'] = f[dataset]['where'].attrs.get('elangle')
        meta[dataset]['range_start'] = f[dataset]['where'].attrs.get('rstart')
        meta[dataset]['range_scale'] = f[dataset]['where'].attrs.get('rscale')
        meta[dataset]['range_bins'] = f[dataset]['where'].attrs.get('nbins')
        meta[dataset]['azim_bins'] = f[dataset]['where'].attrs.get('nrays')

        meta[dataset] = {k: v[0] if type(v) is np.ndarray else v for k, v in meta[dataset].items()}  # unpack entirely

        meta[dataset]['products'] = {}

        for data in f[dataset]:

            if not data.startswith('data'):
                continue

            quantity = f[dataset][data]['what'].attrs.get('quantity').decode('UTF-8')
            meta[dataset]['products'][quantity] = data

    return meta


def select_datasets_odim(trg_elevs, meta):
    """
    Scans closest to trg_elevs are picked. Throws an exception if these do not contain target_sp_products and
    target_dp_products or if target_dp_products cannot be derived from other existing products.

    :param trg_elevs: list of target elevations
    :param meta: radar metadata dictionary
    :return: names of datasets containing scans closest to trg_elevs
    """
    # Extract elevations from meta dictionary
    elevations = [dataset['elevation'] for key, dataset in meta.items() if key.startswith('dataset')]
    elevations = [elevation[0] if type(elevation) is np.ndarray else elevation for elevation in elevations]
    elevations = list(sorted(set(elevations)))

    trg_elevs = trg_elevs[meta['country']]
    if len(elevations) < len(trg_elevs):
        raise ValueError('Number of available elevations ({}) lower than number of target elevations ({}).'
                         .format(len(elevations), len(trg_elevs)))

    # Pick elevations closest to trg_elevs
    picked_elevs = [min(elevations, key=lambda x: abs(x - trg_elev)) for trg_elev in trg_elevs]
    print(picked_elevs)

    # Find datasets that contain picked_elevs
    picked_datasets = []

    for picked_elev in picked_elevs:
        for key, value in meta.items():
            if not key.startswith('dataset'):
                continue

            # Sometimes scans are done at the same elevation with different PRFs, some of which we may filter out
            if key in skipped_scans[meta['country']]:
                continue

            # Check if dataset is recorded at picked_elev
            if not value['elevation'] == picked_elev:
                continue

            # Check if single-pol products are available
            check_sp_products = all(product in value['products'] for product in target_sp_products[meta['country']])
            if not check_sp_products:
                raise Exception('Some of the target single-pol products ({}) are missing at target elevations.'.
                                format(target_sp_products[meta['country']]))

            # Check if dual-pol products are available
            check_dp_products = all(product in value['products'] for product in target_dp_products[meta['country']])
            if not check_dp_products:

                # Check if we can derive dual-pol products from existing products
                if 'ZDR' not in value['products']:
                    if 'DBZH' not in value['products'] or 'DBZV' not in value['products']:
                        raise Exception('ZDR is missing and cannot be computed at target elevation: {}'.
                                        format(picked_elev))
                    else:
                        target_dp_products[meta['country']].extend(['DBZH', 'DBZV'])

                # Check if RHOHV is missing, which we cannot compute from existing products
                if 'RHOHV' not in value['products']:
                    raise Exception('RHOHV is missing at target elevation: {}'.format(picked_elev))

            # Apparently all picked elevations are fine to use
            picked_datasets.append(key)

    return picked_datasets


def parse_odim_data(f, dataset, product):
    """
    Converts stored raw data by applying the calibration formula

    :param f: h5py File Object of ODIM formatted radar archive
    :param dataset: dataset path in ODIM archive (dataset1, dataset2, etc.)
    :param product: product identifier (data1, data2, etc.)
    :return: nparray of corrected data and dict of calibration formula elements
    """
    calibration = {}

    calibration['gain'] = f[dataset][product]['what'].attrs.get('gain')
    calibration['offset'] = f[dataset][product]['what'].attrs.get('offset')
    calibration['nodata'] = f[dataset][product]['what'].attrs.get('nodata')
    calibration['undetect'] = f[dataset][product]['what'].attrs.get('undetect')

    calibration = {k: v[0] if type(v) is np.ndarray else v for k, v in calibration.items()}  # unpack entirely

    raw_data = f[dataset][product]['data'].value

    missing = np.logical_or(raw_data == calibration['nodata'], raw_data == calibration['undetect'])

    corrected_data = raw_data * calibration['gain'] + calibration['offset']
    corrected_data[missing] = np.nan

    return corrected_data, calibration


def interpolate_scan_products(products, azimuths, ranges, elevation, r_max, pix_dims, padding):
    """
    Interpolates all provided products belonging to a single scan at once, so coordinates are calculated just once.
    Based on radar2mat code from the WSRLIB package:
    Sheldon, Daniel. WSRLIB: MATLAB Toolbox for Weather Surveillance Radar. http://bitbucket.org/dsheldon/wsrlib, 2015.

    @NOTE: If implemented with NEXRAD radar support: make sure azimuths are always sorted.

    :param products: dictionary of products with names as keys
    :param azimuths: 1D array of azimuths
    :param ranges: 1D array of ranges
    :param elevation: elevation in degrees
    :param r_max: maximum range of the interpolation in meters
    :param pix_dims: width and height dimension of the interpolated image
    :return: dictionary of interpolated products
    """
    # Determine query points
    x = np.linspace(-r_max, r_max, pix_dims)
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

    groundranges = slant2ground(ranges, elevation)

    aa, rr = np.meshgrid(groundranges, azimuths)
    interpolated_products = {}

    for product_name, product in products.items():
        i = interpolate.griddata((aa.ravel(), rr.ravel()), product.ravel(), (R, PHI), method='nearest')
        i = np.pad(i, padding, mode='constant', constant_values=np.nan)
        interpolated_products[product_name] = i

    return interpolated_products


def stack_interpolated_datasets(interpolated_datasets, target_sp_products, target_dp_products):
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
    sp_products = [data for data in target_sp_products if data in interpolated_datasets[first_dataset]]
    dp_products = [data for data in target_dp_products if data in interpolated_datasets[first_dataset]]
    first_dim_sp = nr_elevations * len(sp_products)
    first_dim_dp = nr_elevations * len(dp_products)

    image_dims = pixel_dimensions + 2 * padding

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


def get_azimuths_ranges_odim(meta, dataset):
    """
    Generates 1D arrays containing azimuths in degrees and range values in meters for every rangebin along a ray

    :param meta: radar metadata dictionary
    :param dataset: dataset path in ODIM archive (dataset1, dataset2, etc.)
    :return: 1D array for azimuths in degrees and ranges in meters
    """
    r = meta[dataset]['range_start'] + meta[dataset]['range_scale'] * np.arange(0, meta[dataset]['range_bins'])
    az = np.arange(0, 360, 360 / meta[dataset]['azim_bins'])

    return az, r


def load_nexrad_file(file):
    raise NotImplementedError


def save_file(output_file, rdata, dp):
    """
    Save file as .mat or .npz file containing single-pol (rdata) and dual-pol (dp) radar products

    :param output_file: name to save file as; the extension determines file type (either .mat or .npz)
    :param rdata: stacked single-pol data
    :param dp: stacked dual-pol data
    """
    output_file = pathlib.Path(output_file)

    if output_file.suffix == '.mat':
        io.savemat(output_file, {'rdata': rdata, 'dp': dp})
    elif output_file.suffix == '.npz':
        np.savez_compressed(output_file, rdata=rdata, dp=dp)

    print('Processed: {}'.format(output_file))


if __name__ == "__main__":
    # mistnet_raw_path = pathlib.Path.cwd() / 'mistnet'
    # mistnet_renders_path = pathlib.Path.cwd() / 'mistnet'

    start = time.time()

    mistnet_raw_path = pathlib.Path.cwd() / 'data' / 'raw'
    mistnet_renders_path = mistnet_raw_path / 'data' / 'processed'

    rendered_files = []
    raw_files = []

    for file in mistnet_renders_path.glob('*'):
        if file.suffix == '.npz':
            rendered_files.append(file.stem)

    for file in mistnet_raw_path.glob('*'):
        if file.suffix == '.h5':
            raw_files.append(file)

    unprocessed_files = [file for file in raw_files if file.stem not in rendered_files]

    print('Files left to process: {}'.format(len(unprocessed_files)))

    with Pool(processes=8) as pool:
        pool.map(render_radar_file, unprocessed_files)

    end = time.time()

    print('Generating {} rendered files took: {}'.format(len(unprocessed_files), end-start))
