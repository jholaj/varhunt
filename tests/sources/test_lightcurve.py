import numpy as np
import pytest

from factories import make_light_curve, sine, write_qlp
from varhunt.analysis.detect import amplitude_spectrum
from varhunt.sources.lightcurve import SETTLE_DAYS, detrended, load_qlp, load_spoc

WINDOW = 3.0


def test_cleans_and_normalises(tmp_path):
    lc = make_light_curve(noise=1e-4)
    flux = lc.flux * 1234.0
    middle = len(flux) // 4
    flux[middle] = np.nan
    flux[middle + 10] = flux[middle + 10] * 1.5
    flux[middle + 20] = flux[middle + 20] * 0.9
    quality = np.zeros(len(flux), dtype=int)
    quality[middle + 30] = 512
    path = write_qlp(tmp_path / "lc.fits", type(lc)(lc.star, lc.time, flux, lc.orbit), quality)

    loaded = load_qlp(path)
    restart = lc.time[np.argmax(np.diff(lc.time)) + 1]
    first_days = (lc.time < lc.time[0] + SETTLE_DAYS) | (
        (lc.time >= restart) & (lc.time < restart + SETTLE_DAYS)
    )
    assert len(loaded.time) == len(lc.time) - first_days.sum() - 2
    assert np.median(loaded.flux) == pytest.approx(1.0, abs=1e-4)
    assert loaded.flux.max() == pytest.approx(1.5, abs=2e-3)

    clean = detrended(loaded, WINDOW)
    assert len(clean.time) == len(loaded.time) - 1
    assert clean.flux.max() < 1.01
    assert clean.flux.min() == pytest.approx(0.9, abs=2e-3)
    assert loaded.star.tic == lc.star.tic
    assert loaded.star.teff == 7000.0
    assert set(np.unique(loaded.orbit)) == {1, 2}


def test_detrending_removes_slow_drifts(tmp_path):
    path = write_qlp(
        tmp_path / "lc.fits", make_light_curve(lambda time: 0.03 * (time - time[0]) / 24)
    )
    loaded = detrended(load_qlp(path), WINDOW)
    slope = np.polyfit(loaded.time - loaded.time[0], loaded.flux, 1)[0]
    assert abs(slope * loaded.baseline) < 1e-3


def test_detrending_keeps_day_long_periods(tmp_path):
    path = write_qlp(tmp_path / "lc.fits", make_light_curve(sine(1 / 0.7, 0.01)))
    loaded = detrended(load_qlp(path), WINDOW)
    amplitude = amplitude_spectrum(loaded.time, loaded.flux - 1.0, np.array([1 / 0.7]), "cython")
    assert amplitude[0] == pytest.approx(0.01, rel=0.1)


def test_first_day_after_every_gap_is_dropped(tmp_path):
    loaded = load_qlp(write_qlp(tmp_path / "lc.fits", make_light_curve()))
    first = loaded.time[0]
    after_gap = loaded.time[np.argmax(np.diff(loaded.time)) + 1]
    assert first >= 3000.0 + SETTLE_DAYS
    assert 3012.5 + SETTLE_DAYS <= after_gap < 3012.5 + SETTLE_DAYS + 0.01


def test_missing_temperature(tmp_path):
    path = write_qlp(tmp_path / "lc.fits", make_light_curve(), teff=None)
    assert load_qlp(path).star.teff is None


def test_the_qlp_placeholder_for_an_unknown_temperature_is_no_temperature(tmp_path):
    path = write_qlp(tmp_path / "lc.fits", make_light_curve(), teff=-1.0)
    assert load_qlp(path).star.teff is None


def test_a_known_signal_survives_a_short_detrending_window():
    lc = make_light_curve(sine(1 / 1.2, 0.01), noise=1e-4)
    model = sine(1 / 1.2, 0.01)(lc.time)
    clean = detrended(lc, 1.0, model)
    amplitude = amplitude_spectrum(clean.time, clean.flux - 1.0, np.array([1 / 1.2]), "cython")
    assert amplitude[0] == pytest.approx(0.01, rel=0.02)


def test_timing_properties():
    lc = make_light_curve()
    assert lc.cadence == pytest.approx(200 / 86400)
    assert lc.nyquist == pytest.approx(216.0)
    assert lc.baseline == pytest.approx(24.0, abs=0.01)


def test_tess_spoc_light_curves_use_the_pdc_flux(tmp_path):
    lc = make_light_curve(noise=1e-4)
    path = write_qlp(tmp_path / "spoc.fits", lc, flux_column="PDCSAP_FLUX")
    loaded = load_spoc(path)
    assert np.median(loaded.flux) == pytest.approx(1.0, abs=1e-4)
    assert set(np.unique(loaded.orbit)) == {0}
