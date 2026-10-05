from dataclasses import replace

import numpy as np
import pytest

from factories import make_light_curve
from varhunt.analysis.ephemeris import (
    heliocentric_date,
    linear_ephemeris,
    minima,
    phase_of,
    profile,
)

PERIOD, EPOCH, DURATION = 4.04, 3001.7, 0.17


def trapezoid(period: float, duration: float, depth: float, epoch: float):
    def signal(time: np.ndarray) -> np.ndarray:
        distance = np.abs(((time - epoch) / period + 0.5) % 1.0 - 0.5) * period
        return -depth * np.clip((duration / 2 - distance) / (duration / 4), 0.0, 1.0)

    return signal


def binary(noise: float = 3e-4):
    primary = trapezoid(PERIOD, DURATION, 0.05, EPOCH)
    secondary = trapezoid(PERIOD, DURATION, 0.02, EPOCH + PERIOD / 2)
    return make_light_curve(lambda time: primary(time) + secondary(time), noise=noise)


def test_every_eclipse_is_timed_and_measured():
    found = minima(binary(), 0.01, DURATION)
    deep = sorted(m.time for m in found if m.depth > 0.035)
    shallow = [m for m in found if m.depth <= 0.035]
    assert len(deep) >= 4
    assert len(shallow) >= 3
    assert all(abs((t - EPOCH) / PERIOD - round((t - EPOCH) / PERIOD)) < 0.002 for t in deep)


def test_minima_spread_over_years_pin_the_period_down():
    truth = 4.0412345
    noise = np.random.default_rng(0).normal(0, 2e-4, 5)
    times = [EPOCH + n * truth + dt for n, dt in zip((0, 1, 2, 170, 171), noise, strict=True)]
    elements = linear_ephemeris(times, 4.04)
    assert elements.period == pytest.approx(truth, abs=5e-6)
    assert elements.cycles == 171
    assert elements.rms_minutes < 1.0


def test_a_false_dip_does_not_bend_the_ephemeris():
    times = [EPOCH + n * PERIOD for n in range(6)] + [EPOCH + 2.4 * PERIOD]
    elements = linear_ephemeris(times, PERIOD)
    assert elements.period == pytest.approx(PERIOD, abs=1e-6)
    assert elements.rms_minutes < 0.01


def test_the_secondary_sits_halfway_and_both_eclipses_have_their_depth():
    lc = binary()
    elements = linear_ephemeris([EPOCH + n * PERIOD for n in range(6)], PERIOD)
    secondary = [m.time for m in minima(lc, 0.01, DURATION) if m.depth <= 0.035]
    assert phase_of(secondary, elements) == pytest.approx(0.5, abs=0.002)
    first = profile([lc], elements, 0.0)
    assert first.depth == pytest.approx(0.05, rel=0.05)
    assert first.duration == pytest.approx(DURATION / PERIOD, rel=0.2)
    assert profile([lc], elements, 0.5).depth == pytest.approx(0.02, rel=0.1)


def test_an_eclipse_cut_by_a_gap_is_not_timed():
    lc = binary()
    cut = lc.time < EPOCH + 0.01
    timed = minima(
        replace(lc, time=lc.time[cut], flux=lc.flux[cut], orbit=lc.orbit[cut]), 0.01, DURATION
    )
    assert all(abs(m.time - EPOCH) > 0.05 for m in timed)


def test_the_heliocentric_date_runs_about_seventy_seconds_behind_tdb():
    bjd = 2461069.74315
    hjd = heliocentric_date(bjd, 148.10071, -26.41831)
    assert (bjd - hjd) * 86400 == pytest.approx(69.5, abs=5.0)
