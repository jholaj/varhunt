from dataclasses import replace

import pytest

from factories import make_light_curve, make_star, sine
from varhunt.analysis.classify import rotation
from varhunt.analysis.detect import detect
from varhunt.checks.evidence import (
    is_variable_type,
    mentions_variability,
    refine_frequency,
    repeats,
)
from varhunt.config import Settings

SETTINGS = Settings()


def test_sector_with_the_same_frequency_confirms():
    candidate = make_light_curve(sine(5.2, 3e-3))
    same = make_light_curve(
        sine(5.2, 3e-3, phase=1.3), seed=1, star=replace(make_star(), sector=45)
    )
    other = make_light_curve(sine(7.9, 3e-3), seed=2, star=replace(make_star(), sector=46))
    quiet = make_light_curve(seed=3, star=replace(make_star(), sector=47))
    assert repeats(5.2, candidate, same, detect(same, SETTINGS))
    assert not repeats(5.2, candidate, other, detect(other, SETTINGS))
    assert not repeats(5.2, candidate, quiet, detect(quiet, SETTINGS))


def test_a_weak_side_signal_of_another_sector_does_not_count():
    candidate = make_light_curve(sine(5.2, 3e-3))
    waves = [sine(2.1, 6e-3), sine(7.7, 5e-3), sine(5.2, 1e-3)]
    busy = make_light_curve(lambda t: sum(w(t) for w in waves), seed=5)
    assert not repeats(5.2, candidate, busy, detect(busy, SETTINGS))


def test_half_the_frequency_counts_for_binaries():
    candidate = make_light_curve(sine(5.2, 3e-3))
    other = make_light_curve(sine(2.6, 3e-3), seed=6)
    assert repeats(5.2, candidate, other, detect(other, SETTINGS))


def test_variability_catalogues_are_recognised_by_name_or_description():
    assert mentions_variability(
        "I/358/vclassre Variability classification results of all classifiers"
    )
    assert mentions_variability("J/ApJS/276/57/table5 TESS periodic variable star catalog")
    assert mentions_variability("B/gcvs/gcvs_cat GCVS catalog (GCVS 5.1, version Oct, 2020)")
    assert mentions_variability("J/AJ/138/466/variables Results of the automated classification")
    assert not mentions_variability(
        "J/AJ/155/39/table5 Variability Upper Limits for Non-Variable Sources"
    )
    assert not mentions_variability("II/246/out 2MASS Point Source Catalogue, output")


def test_simbad_variable_types():
    assert is_variable_type("EB*")
    assert is_variable_type("dS*")
    assert not is_variable_type("*")
    assert not is_variable_type("PM*")


def test_joining_sectors_years_apart_sharpens_the_frequency():
    truth = 5.123456
    first = make_light_curve(sine(truth, 3e-3, phase=0.2))
    shifted = make_light_curve(lambda time: sine(truth, 3e-3, phase=0.2)(time + 700.0), seed=4)
    later = replace(shifted, time=shifted.time + 700.0, star=replace(shifted.star, sector=99))
    refined = refine_frequency([first, later], 5.1235)
    assert refined.span == pytest.approx(724.0, abs=1.0)
    assert abs(refined.frequency - truth) < 3 * refined.error
    assert refined.error < 1e-5
    assert not refined.ambiguous


def test_weak_signal_across_a_long_gap_leaves_the_cycle_count_open():
    truth = 5.123456
    first = make_light_curve(sine(truth, 5e-4))
    shifted = make_light_curve(lambda time: sine(truth, 5e-4)(time + 700.0), seed=4)
    later = replace(shifted, time=shifted.time + 700.0, star=replace(shifted.star, sector=99))
    assert refine_frequency([first, later], 5.1235).ambiguous


def test_rotation_of_a_fast_a_star_matches_its_measured_v_sin_i():
    star = replace(make_star(teff=10249.0), radius=1.944, mass=2.635)
    spin = rotation(star, 0.38855, vsini=236.0)
    assert spin.equatorial_velocity == pytest.approx(253.0, abs=2.0)
    assert spin.breakup_velocity == pytest.approx(508.0, abs=3.0)
    assert spin.vsini / spin.equatorial_velocity == pytest.approx(0.93, abs=0.01)


def test_rotation_needs_a_radius():
    assert rotation(make_star(), 0.5, vsini=None) is None
