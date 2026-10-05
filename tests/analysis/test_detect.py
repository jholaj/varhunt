import numpy as np
import pytest

from factories import eclipses, make_light_curve, sine
from varhunt.analysis.detect import (
    Detection,
    Eclipse,
    amplitude_spectrum,
    detect,
    signal_model,
    survives_wider_detrending,
)
from varhunt.config import Settings

SETTINGS = Settings()


def test_amplitude_spectrum_is_calibrated_in_semi_amplitude():
    lc = make_light_curve(sine(7.3, 2e-3), noise=1e-4)
    spectrum = amplitude_spectrum(lc.time, lc.flux - 1.0, np.array([7.3]), method="cython")
    assert spectrum[0] == pytest.approx(2e-3, rel=0.02)


def test_recovers_frequency_amplitude_and_snr_of_a_sinusoid():
    detection = detect(make_light_curve(sine(7.3, 2e-3)), SETTINGS, eclipses=False)
    primary = detection.primary
    assert primary.frequency == pytest.approx(7.3, abs=1e-3)
    assert primary.amplitude == pytest.approx(2e-3, rel=0.05)
    assert primary.snr > 20
    assert detection.is_consistent(SETTINGS)
    assert 0 < detection.period_error < 1e-4


def test_epoch_is_a_time_of_maximum_light():
    signal = sine(3.1, 5e-3, phase=1.0)
    detection = detect(make_light_curve(signal, noise=2e-4), SETTINGS, eclipses=False)
    assert signal(np.array([detection.primary.epoch]))[0] == pytest.approx(5e-3, rel=0.02)


@pytest.mark.parametrize("seed", range(5))
def test_pure_noise_is_not_variable(seed):
    detection = detect(make_light_curve(noise=1e-3, seed=seed), SETTINGS)
    assert not detection.is_variable


def test_prewhitening_separates_two_modes():
    def signal(time):
        return sine(12.1, 3e-3)(time) + sine(17.6, 1e-3)(time)

    detection = detect(make_light_curve(signal), SETTINGS, eclipses=False)
    found = sorted(round(f.frequency, 1) for f in detection.frequencies)
    assert found == [12.1, 17.6]


def test_signal_in_one_orbit_only_is_inconsistent():
    lc = make_light_curve(lambda time: sine(4.2, 3e-3)(time) * (time < 3011.5))
    detection = detect(lc, SETTINGS, eclipses=False)
    assert detection.is_variable
    assert not detection.is_consistent(SETTINGS)


def test_finds_detached_eclipses():
    detection = detect(make_light_curve(eclipses(1.37, 0.1, 0.01)), SETTINGS)
    assert detection.eclipse is not None
    assert detection.eclipse.period == pytest.approx(1.37, rel=0.005)
    assert detection.eclipse.depth == pytest.approx(0.01, rel=0.15)
    assert detection.eclipse.events >= 3


def test_eclipses_beyond_the_sinusoid_period_limit_are_found_at_their_period():
    detection = detect(make_light_curve(eclipses(2.37, 0.1, 0.01)), SETTINGS)
    assert detection.eclipse.period == pytest.approx(2.37, rel=0.005)


def test_alternating_eclipse_depths_double_the_period():
    primary = eclipses(4.04, 0.1, 0.05, epoch=3001.7)
    secondary = eclipses(4.04, 0.1, 0.02, epoch=3003.72)
    lc = make_light_curve(lambda time: primary(time) + secondary(time))
    eclipse = detect(lc, SETTINGS).eclipse
    assert eclipse.period == pytest.approx(4.04, rel=0.005)
    assert eclipse.depth == pytest.approx(0.05, rel=0.15)
    assert eclipse.secondary_depth == pytest.approx(0.02, rel=0.2)
    assert (eclipse.epoch - 3001.7) / 4.04 == pytest.approx(
        round((eclipse.epoch - 3001.7) / 4.04), abs=0.01
    )


def test_eclipses_are_reported_at_the_true_period_not_a_multiple():
    detection = detect(make_light_curve(eclipses(1.5059, 0.08, 0.02)), SETTINGS)
    assert detection.eclipse.period == pytest.approx(1.5059, rel=0.003)


def test_multiperiodic_pulsator_is_not_mistaken_for_eclipses():
    def signal(time):
        return sine(12.3, 0.02)(time) + sine(15.7, 0.01, 1.0)(time) + sine(9.1, 0.008, 2.0)(time)

    detection = detect(make_light_curve(signal), SETTINGS)
    assert detection.eclipse is None
    assert round(detection.primary.frequency, 1) == 12.3


def test_sinusoid_is_not_mistaken_for_eclipses():
    detection = detect(make_light_curve(sine(1.3, 5e-3)), SETTINGS)
    assert detection.eclipse is None


def test_too_short_light_curve_gives_empty_detection():
    lc = make_light_curve()
    short = type(lc)(star=lc.star, time=lc.time[:100], flux=lc.flux[:100], orbit=lc.orbit[:100])
    assert not detect(short, SETTINGS).is_variable


def test_signal_model_reproduces_the_fitted_sinusoids():
    signal = sine(5.2, 4e-3, phase=0.3)
    lc = make_light_curve(signal, noise=2e-4)
    model = signal_model(detect(lc, SETTINGS, eclipses=False))
    assert np.std(model(lc.time) - signal(lc.time)) < 2e-4


def test_signal_model_of_eclipses_is_a_box():
    eclipse = Eclipse(
        period=2.0, duration=0.2, depth=0.01, snr=20.0, epoch=10.0, events=5, odd_even_sigma=0.0
    )
    model = signal_model(Detection(eclipse=eclipse))
    assert model(np.array([10.0, 12.05, 11.0, 13.0])).tolist() == [-0.01, -0.01, 0.0, 0.0]


def test_red_noise_wiggles_are_not_eclipses():
    rng = np.random.default_rng(5)
    periods = rng.uniform(0.5, 3.0, 25)
    phases = rng.uniform(0, 2 * np.pi, 25)

    def wiggles(time):
        return sum(
            7e-4 * np.sin(2 * np.pi * time / p + ph) for p, ph in zip(periods, phases, strict=True)
        )

    detection = detect(make_light_curve(wiggles, noise=5e-4), SETTINGS)
    assert detection.eclipse is None


def test_periods_near_the_detrending_window_keep_their_amplitude():
    detection = detect(make_light_curve(sine(1 / 1.2, 0.01)), SETTINGS)
    assert detection.primary.period == pytest.approx(1.2, rel=0.01)
    assert detection.primary.amplitude == pytest.approx(0.01, rel=0.05)
    assert len(detection.frequencies) == 1


def test_a_narrow_period_range_does_not_break_the_noise_estimate():
    narrow = Settings(min_period=0.19, max_period=0.21)
    detection = detect(make_light_curve(sine(5.0, 3e-3)), narrow)
    assert detection.primary.frequency == pytest.approx(5.0, abs=0.01)


def test_a_signal_slower_than_the_search_range_is_not_reported_at_its_edge():
    slow_and_fast = make_light_curve(lambda time: sine(1 / 4.1, 0.02)(time) + sine(5.0, 3e-3)(time))
    detection = detect(slow_and_fast, SETTINGS)
    assert all(f.period < SETTINGS.max_period * 0.99 for f in detection.frequencies)
    assert any(abs(f.frequency - 5.0) < 0.01 for f in detection.frequencies)


def test_a_signal_inside_the_range_is_not_flagged_as_beyond_it():
    assert not detect(make_light_curve(sine(1 / 1.9, 0.01)), SETTINGS).beyond_range


def test_peaks_left_behind_by_a_slow_signal_do_not_survive_wider_detrending():
    lc = make_light_curve(sine(1 / 4.1, 0.02))
    detection = detect(lc, SETTINGS)
    assert detection.primary is not None
    assert not survives_wider_detrending(lc, detection, SETTINGS)


def test_a_real_day_long_signal_survives_wider_detrending():
    lc = make_light_curve(sine(1 / 1.29, 0.01))
    assert survives_wider_detrending(lc, detect(lc, SETTINGS), SETTINGS)


def test_an_edge_peak_marks_a_signal_beyond_the_range():
    detection = detect(make_light_curve(sine(1 / 2.1, 0.02), noise=1e-4), SETTINGS)
    assert detection.beyond_range
