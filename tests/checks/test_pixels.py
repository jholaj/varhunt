import numpy as np
import pytest

from factories import make_cube, make_time, make_wcs
from varhunt.checks.pixels import Cutout, localise
from varhunt.sources.catalogs import Neighbour

PERIOD, EPOCH = 0.5695, 3000.3


def wave(time: np.ndarray) -> np.ndarray:
    return np.cos(2 * np.pi * (time - EPOCH) / PERIOD)


TARGET_PX = (7.0, 7.0)
NEIGHBOUR_PX = (10.0, 7.0)


def neighbour_at(wcs, source_id: int, x: float, y: float, gmag: float) -> Neighbour:
    coord = wcs.pixel_to_world(x, y)
    return Neighbour(source_id, float(coord.ra.deg), float(coord.dec.deg), gmag)


def scene(variable_px: tuple[float, float] | None, sky_amplitude: float = 5.0):
    wcs = make_wcs()
    time = make_time()
    signal = wave(time)
    still = np.zeros_like(time)
    stars = [
        (*TARGET_PX, 20000.0, 0.01 * signal if variable_px == TARGET_PX else still),
        (*NEIGHBOUR_PX, 8000.0, 0.03 * signal if variable_px == NEIGHBOUR_PX else still),
    ]
    cube = make_cube(time, stars, sky_signal=sky_amplitude * signal)
    target = wcs.pixel_to_world(*TARGET_PX)
    candidates = [
        neighbour_at(wcs, 1, *TARGET_PX, 11.0),
        neighbour_at(wcs, 2, *NEIGHBOUR_PX, 12.0),
        neighbour_at(wcs, 3, 3.0, 12.0, 14.0),
    ]
    return Cutout(time, cube, wcs), float(target.ra.deg), float(target.dec.deg), candidates


def test_signal_on_the_target_despite_a_varying_sky():
    cutout, ra, dec, candidates = scene(TARGET_PX)
    result = localise(cutout, ra, dec, wave, 1 / PERIOD, candidates)
    assert result.verdict == "on_target"
    assert result.offset_px < 0.3
    assert result.centroid_error_px < 0.1
    assert result.likely_source.source_id == 1


def test_signal_from_a_neighbour_is_off_target():
    cutout, ra, dec, candidates = scene(NEIGHBOUR_PX)
    result = localise(cutout, ra, dec, wave, 1 / PERIOD, candidates)
    assert result.verdict == "off_target"
    assert result.centroid == pytest.approx(NEIGHBOUR_PX, abs=0.3)
    assert result.likely_source.source_id == 2


def test_image_motion_is_not_a_brightness_change():
    wcs = make_wcs()
    time = make_time()
    signal = wave(time)
    cube = make_cube(time, [(7.0 + 0.05 * signal, 7.0, 20000.0, np.zeros_like(time))], 5.0 * signal)
    target = wcs.pixel_to_world(*TARGET_PX)
    cutout = Cutout(time, cube, wcs)
    result = localise(cutout, float(target.ra.deg), float(target.dec.deg), wave, 1 / PERIOD, [])
    assert result.verdict == "motion"
    assert result.dipole > 0.5


def test_sky_alone_gives_no_signal():
    cutout, ra, dec, candidates = scene(None)
    result = localise(cutout, ra, dec, wave, 1 / PERIOD, candidates)
    assert result.verdict == "no_signal"


def test_photon_noise_of_a_bright_star_is_not_a_signal():
    wcs = make_wcs()
    time = make_time()
    cube = make_cube(time, [(*TARGET_PX, 400000.0, np.zeros_like(time))], np.zeros_like(time))
    cube += np.random.default_rng(1).normal(0.0, 1.0, cube.shape) * np.sqrt(cube)
    target = wcs.pixel_to_world(*TARGET_PX)
    cutout = Cutout(time, cube, wcs)
    result = localise(cutout, float(target.ra.deg), float(target.dec.deg), wave, 1 / PERIOD, [])
    assert result.verdict == "no_signal"


def test_empty_frames_are_skipped():
    cutout, ra, dec, candidates = scene(TARGET_PX)
    cutout.flux[::50] = np.nan
    result = localise(cutout, ra, dec, wave, 1 / PERIOD, candidates)
    assert result.verdict == "on_target"


def test_a_cutout_without_usable_pixels_has_no_signal():
    cutout, ra, dec, candidates = scene(TARGET_PX)
    cutout.flux[:] = np.nan
    result = localise(cutout, ra, dec, wave, 1 / PERIOD, candidates)
    assert result.verdict == "no_signal"


def test_pointing_jitter_does_not_hide_a_real_signal():
    wcs = make_wcs()
    time = make_time()
    rng = np.random.default_rng(2)
    jitter_x, jitter_y = rng.normal(0.0, 0.02, (2, len(time)))
    star = (7.0 + jitter_x, 7.0 + jitter_y, 400000.0, 0.001 * wave(time))
    cube = make_cube(time, [star], np.zeros_like(time))
    target = wcs.pixel_to_world(*TARGET_PX)
    cutout = Cutout(time, cube, wcs)
    result = localise(cutout, float(target.ra.deg), float(target.dec.deg), wave, 1 / PERIOD, [])
    assert result.verdict == "on_target"
    assert result.offset_px < 3 * result.centroid_error_px


def test_a_star_between_pixels_is_located_without_a_pull_to_the_pixel_centre():
    wcs = make_wcs()
    time = make_time()
    between = (7.45, 6.6)
    cube = make_cube(time, [(*between, 20000.0, 0.01 * wave(time))], np.zeros_like(time))
    target = wcs.pixel_to_world(*between)
    cutout = Cutout(time, cube, wcs)
    result = localise(cutout, float(target.ra.deg), float(target.dec.deg), wave, 1 / PERIOD, [])
    assert result.offset_px < 0.05
