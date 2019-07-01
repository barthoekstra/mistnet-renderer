import pytest
import warnings
from renderer import *


@pytest.fixture()
def renderer_output_germany():
    yield RadarRenderer('tests/data/germany.h5', skip_save=True)


# For some reason this test is not isolated enough if at the bottom of the test suite, but it should be able to pass if
# it is at the top. Not sure what is going on, but not a priority to fix.
def test_renderer_output_germany(renderer_output_germany):
    reference_germany = np.load('tests/data/germany.npz')
    assert np.allclose(renderer_output_germany.sp_data, reference_germany['rdata'], equal_nan=True)
    assert np.allclose(renderer_output_germany.dp_data, reference_germany['dp'], equal_nan=True)


@pytest.fixture()
def renderer():
    yield RadarRenderer('tests/data/germany.h5', skip_render=True)


@pytest.fixture()
def renderer_germany_incomplete():
    yield RadarRenderer('tests/data/germany_incomplete.h5', skip_render=True)


@pytest.fixture()
def renderer_netherlands():
    yield RadarRenderer('tests/data/netherlands.h5', skip_render=True)


def test_renderer_setup(renderer):
    assert renderer.output_type == 'npz'
    assert renderer.output_file == pathlib.Path.cwd() / 'tests/data/germany.npz'
    assert renderer.radar_format is None
    assert renderer.target_elevations == [0.5, 1.5, 2.5, 3.5, 4.5]
    assert renderer.skipped_scans == []
    assert renderer.target_sp_products == ['DBZH', 'VRADH', 'WRADH']
    assert renderer.target_dp_products == ['RHOHV', 'ZDR']
    assert renderer.skip_render
    assert renderer._f.id.id == 0  # Check if file is closed
    assert renderer.odim_radar_db[0]['wmocode'] == '11038'  # Check if ODIM DB is correctly loaded


def test_renderer_radar_dict(renderer):
    required_keys = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.0, 12.0, 17.0, 25.0, 'beamwidth', 'country', 'source']
    assert all(key in renderer.radar.keys() for key in required_keys)

    assert renderer.radar['country'] == 'Germany'

    required_keys = ['DBZH', 'WRADH', 'VRADH', 'RHOHV', 'ZDR', 'DBZV', 'highprf', 'lowprf', 'startdate', 'starttime',
                     'enddate', 'endtime', 'elangle', 'nbins', 'nrays', 'rscale', 'rstart']
    assert all(key in renderer.radar[0.5].keys() for key in required_keys)

    assert renderer.radar[0.5]['DBZH']['gain'] == pytest.approx(0.501, 0.01)
    assert renderer.radar[0.5]['DBZH']['nodata'] == pytest.approx(255.0)
    assert isinstance(renderer.radar[0.5]['DBZH']['values'], np.ndarray)


def test_renderer_select_data_odim_output(renderer):
    selected_elevations = [0.5, 1.5, 2.5, 3.5, 4.5]
    assert list(renderer.selected_data.keys()) == selected_elevations

    datasets = {'DBZH': None, 'VRADH': None, 'WRADH': None, 'RHOHV': None, 'ZDR': None}
    assert renderer.selected_data == {elevation: datasets for elevation in selected_elevations}


def test_incomplete_volume():
    with pytest.raises(RadarException):
        RadarRenderer('tests/data/germany_incomplete.h5', skip_render=True)


def test_renderer_select_data_odim_too_many_target_elevations(renderer):
    with pytest.raises(RadarException):
        renderer.target_elevations = list(range(5000))
        renderer.select_datasets_odim()


def test_renderer_pick_elevations_iteratively_lower(renderer):
    renderer.elevations = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.0, 12.0, 17.0, 25.0]
    target_elevations = [0.1, 0.2, 0.3, 2.5, 3.5]
    assert renderer.pick_elevations_iteratively(target_elevations) == [0.5, 1.5, 2.5, 3.5, 4.5]


def test_renderer_pick_elevations_iteratively_higher(renderer):
        renderer.elevations = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 8.0, 12.0, 17.0, 25.0]
        target_elevations = [0.3, 2.5, 3.5, 30.0, 60.0, 90.0]
        assert renderer.pick_elevations_iteratively(target_elevations) == [0.5, 2.5, 3.5, 12.0, 17.0, 25.0]


def test_renderer_elevations_too_spread_out(renderer):
    with pytest.raises(RadarException):
        renderer.elevations = [0.5, 1.5, 2.5, 3.5, 15.0, 17.0, 25.0]
        renderer.select_datasets_odim()


def test_renderer_select_data_odim_no_target_sp_products(renderer):
    with pytest.raises(RadarException):
        renderer.target_sp_products = ['FooBar']
        renderer.select_datasets_odim()


def test_renderer_select_data_odim_no_target_dp_products(renderer):
    with pytest.raises(RadarException):
        renderer.target_dp_products = ['FooBar']
        renderer.select_datasets_odim()


def test_renderer_select_data_odim_no_rhohv_product(renderer):
    for elevation in renderer.elevations:
        renderer.radar[elevation].pop('RHOHV', None)

    with pytest.raises(RadarException):
        renderer.select_datasets_odim()


def test_renderer_select_data_odim_no_zdr_product(renderer):
    for elevation in renderer.elevations:
        renderer.radar[elevation].pop('ZDR', None)

    renderer.select_datasets_odim()


def test_renderer_select_data_odim_no_zdr_computed_product(renderer):
    for elevation in renderer.elevations:
        renderer.radar[elevation].pop('ZDR', None)
        renderer.radar[elevation].pop('DBZV', None)

    with pytest.raises(RadarException):
        renderer.select_datasets_odim()


def test_unambiguous_velocities(renderer):
    renderer.radar[0.5]['wavelength'] = 5.3
    renderer.radar[0.5]['highprf'] = 1000
    renderer.radar[0.5]['lowprf'] = 750
    assert renderer.calculate_unambiguous_velocities(renderer.radar[0.5]) == pytest.approx(39.75, 0.01)


def test_pick_best_scan(renderer_netherlands):
    datasets = {'DBZH': 1, 'VRADH': 0, 'WRADH': 0, 'RHOHV': 1, 'DBZV': 1}
    assert renderer_netherlands.selected_data[0.3] == datasets


def test_parse_odim_data_nan(renderer):
    renderer.radar[0.5]['DBZH']['values'] = np.asarray([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        assert np.isnan(np.nanmean(renderer.parse_odim_data(0.5, 'DBZH')))


def test_parse_odim_data_normal(renderer):
    assert np.nanmean(renderer.parse_odim_data(0.5, 'DBZH')) == pytest.approx(-4.019440872274833)


def test_parse_odim_data_with_index(renderer_netherlands):
    assert np.nanmean(renderer_netherlands.parse_odim_data(0.3, 'DBZH', index=0)) == pytest.approx(-5.104565218681461)


def test_correct_with_reflectivity(renderer_netherlands):
    reflectivity = np.asarray([[np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    product = np.ones((3, 2))
    assert np.allclose(renderer_netherlands.correct_with_reflectivity(reflectivity, product), reflectivity,
                       equal_nan=True)

    reflectivity = np.asarray([[np.nan, np.nan], [5, np.nan], [np.nan, np.nan]])
    product = np.ones((3, 2)) * 5
    assert np.allclose(renderer_netherlands.correct_with_reflectivity(reflectivity, product), reflectivity,
                       equal_nan=True)





@pytest.fixture()
def renderer_output_us():
    yield RadarRenderer('tests/data/us.h5', skip_save=True)


def test_renderer_output_us(renderer_output_us):
    reference_us = np.load('tests/data/us.npz')
    assert np.allclose(reference_us['rdata'], renderer_output_us.sp_data, equal_nan=True)
    assert np.allclose(reference_us['dp'], renderer_output_us.dp_data, equal_nan=True)


@pytest.fixture()
def renderer_output_netherlands():
    yield RadarRenderer('tests/data/netherlands.h5', skip_save=True)


def test_renderer_output_netherlands(renderer_output_netherlands):
    reference_netherlands = np.load('tests/data/netherlands.npz')
    assert np.allclose(renderer_output_netherlands.sp_data, reference_netherlands['rdata'], equal_nan=True)
    assert np.allclose(renderer_output_netherlands.dp_data, reference_netherlands['dp'], equal_nan=True)
