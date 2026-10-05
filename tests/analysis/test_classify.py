from dataclasses import replace

import numpy as np
import pytest

from factories import make_light_curve, make_star
from varhunt.analysis.classify import ephemeris, type_hint
from varhunt.analysis.detect import Detection, Eclipse, Frequency


def frequencies(*pairs: tuple[float, float]) -> tuple[Frequency, ...]:
    return tuple(Frequency(f, a, 20.0, 0.0) for f, a in pairs)


def hint(detection: Detection, teff: float | None = 7000.0) -> str:
    return type_hint(make_light_curve(star=make_star(teff=teff)), detection)


def test_eclipse_wins():
    eclipse = Eclipse(2.0, 0.1, 0.01, 30.0, 3000.0, 10, 0.0)
    assert hint(Detection(frequencies((1.0, 0.003)), eclipse)) == "EA"


def test_subharmonic_marks_a_contact_binary():
    assert hint(Detection(frequencies((5.0, 0.05), (2.5, 0.01))), teff=5500) == "EW"


def test_short_period_hot_star_is_delta_scuti():
    assert hint(Detection(frequencies((14.0, 0.002)))) == "DSCT"


def test_unknown_temperature_still_allows_delta_scuti():
    assert hint(Detection(frequencies((14.0, 0.002))), teff=None) == "DSCT"


def test_multiperiodic_f_star_with_day_period_is_gamma_dor():
    assert hint(Detection(frequencies((1.2, 0.002), (1.5, 0.001))), teff=7000) == "GDOR"


def test_large_amplitude_with_harmonics_is_rr_lyrae():
    assert hint(Detection(frequencies((1.8, 0.3), (3.6, 0.1))), teff=6800) == "RR"


def test_cool_star_is_rotational():
    assert hint(Detection(frequencies((0.7, 0.004))), teff=4800) == "ROT"


def test_a_cool_star_cannot_spin_faster_than_breakup():
    subgiant = replace(make_star(teff=5745.0), radius=2.15, mass=1.2)
    lc = make_light_curve(star=subgiant)
    assert type_hint(lc, Detection(frequencies((15.7, 0.0016)))) == "VAR"


def test_without_a_size_only_very_short_periods_rule_out_spots():
    assert hint(Detection(frequencies((15.7, 0.0016))), teff=5745.0) == "VAR"


def test_nothing_detected_gives_no_hint():
    assert hint(Detection()) == ""


def test_contact_binary_is_reported_at_the_orbital_period_and_deeper_minimum():
    lc = make_light_curve(star=make_star(teff=5500))
    detection = Detection(
        (Frequency(5.0, 0.05, 30.0, 3000.0), Frequency(2.5, 0.01, 10.0, 3000.1)), period_error=1e-5
    )
    orbit = ephemeris(lc, detection)
    assert orbit.kind == "minimum"
    assert orbit.period == 0.4
    assert orbit.period_error == 2e-5
    model_at = sum(
        f.amplitude * np.cos(2 * np.pi * f.frequency * (orbit.epoch - f.epoch))
        for f in detection.frequencies
    )
    assert model_at < -0.055


def test_eclipsing_binary_uses_the_eclipse_ephemeris():
    eclipse = Eclipse(2.0, 0.1, 0.01, 30.0, 3000.4, 10, 0.0)
    detection = Detection(frequencies((1.0, 0.003)), eclipse)
    lc = make_light_curve()
    orbit = ephemeris(lc, detection)
    assert (orbit.period, orbit.epoch, orbit.kind) == (2.0, 3000.4, "minimum")


def test_pulsator_is_reported_at_maximum_light():
    detection = Detection(frequencies((14.0, 0.002)), period_error=1e-6)
    lc = make_light_curve()
    orbit = ephemeris(lc, detection)
    assert orbit.kind == "maximum"
    assert orbit.period == 1 / 14.0


def test_equal_minima_binary_is_found_from_its_flat_topped_shape():
    # Fourier series of narrow equal dips every 0.6 d, so 1.2 d per orbit.
    f1, width, minimum = 1 / 0.6, 0.03, 3000.2
    waves = tuple(
        Frequency(
            k * f1,
            0.01 * np.exp(-((2 * np.pi * k * f1 * width) ** 2) / 2),
            20.0,
            minimum + 0.5 / (k * f1),
        )
        for k in range(1, 6)
    )
    detection = Detection(waves, period_error=1e-5)
    lc = make_light_curve(star=make_star(teff=5800))
    orbit = ephemeris(lc, detection)
    assert type_hint(lc, detection) == "EA"
    assert orbit.period == pytest.approx(1.2)
    phase = ((orbit.epoch - minimum) / 0.6 + 0.5) % 1.0 - 0.5
    assert phase == pytest.approx(0.0, abs=0.01)


def test_smooth_harmonic_wave_is_not_a_binary():
    detection = Detection(frequencies((1.8, 0.3), (3.6, 0.1)), period_error=1e-5)
    lc = make_light_curve(star=make_star(teff=6800))
    assert ephemeris(lc, detection).period == pytest.approx(1 / 1.8)


def test_small_double_wave_is_more_likely_spots_than_a_contact_binary():
    lc = make_light_curve(star=make_star(teff=9400))
    tiny = Detection((Frequency(5.15, 0.0005, 20.0, 3000.0), Frequency(2.575, 0.0003, 9.0, 3000.1)))
    assert type_hint(lc, tiny) == "ROT"
    assert ephemeris(lc, tiny).period == pytest.approx(2 / 5.15)


def test_very_fast_pulsation_in_an_a_star_is_a_roap_candidate():
    assert hint(Detection(frequencies((142.0, 0.0008))), teff=7800) == "ROAP"
    assert hint(Detection(frequencies((142.0, 0.0008))), teff=4500) != "ROAP"
