"""Rough VSX type and the ephemeris to report, for a human to confirm."""

from collections.abc import Callable
from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy import constants

from varhunt.analysis.detect import Detection, is_harmonic, waves
from varhunt.sources.lightcurve import LightCurve, Star

DSCT_MAX_PERIOD = 0.25
# Rapidly oscillating Ap stars pulsate above 60 c/d, faster than delta Scuti stars normally do.
ROAP_MIN_FREQUENCY = 60.0
ROAP_TEFF = (6000.0, 10000.0)
GDOR_PERIODS = (0.25, 3.0)
RR_PERIODS = (0.2, 1.0)
RR_MIN_AMPLITUDE = 0.1
FTYPE_TEFF = (6300.0, 9500.0)
GDOR_TEFF = (6500.0, 7800.0)
COOL_TEFF = 6300.0
MIN_ROTATION_PERIOD = 0.1
MINIMUM_SEARCH_STEPS = 2000
# A detached binary stays within this share of its range from maximum for most of the orbit.
FLAT_LEVEL = 0.1
FLAT_SHARE = 0.4
# Contact binaries swing by a tenth of a magnitude, a double wave this small is more likely spots.
CONTACT_MIN_RANGE = 0.01


@dataclass(frozen=True)
class Ephemeris:
    period: float
    period_error: float | None
    epoch: float
    kind: str


@dataclass(frozen=True)
class Rotation:
    """Equatorial speed the period would need if it were rotation, with what is measured."""

    equatorial_velocity: float
    breakup_velocity: float | None
    vsini: float | None


def type_hint(lc: LightCurve, detection: Detection) -> str:
    if detection.eclipse is not None:
        return "EA"
    primary = detection.primary
    if primary is None:
        return ""
    tolerance = 1.0 / lc.baseline
    others = [f.frequency for f in detection.frequencies[1:]]
    if _is_binary(lc, detection):
        orbit = 2 * primary.period
        if _flat_share(lc, detection, orbit) > FLAT_SHARE:
            return "EA"
        model = _orbital_model(lc, detection, orbit)(_one_cycle(lc, orbit))
        if model.max() - model.min() >= CONTACT_MIN_RANGE:
            return "EW"
        return "ROT" if _can_rotate(lc.star, orbit) else "VAR"

    period = primary.period
    teff = lc.star.teff
    harmonics = [f for f in others if is_harmonic(f, primary.frequency, tolerance, lowest=2)]
    independent = len(others) > len(harmonics)
    if harmonics and primary.amplitude >= RR_MIN_AMPLITUDE and _within(period, RR_PERIODS):
        return "RR"
    if primary.frequency > ROAP_MIN_FREQUENCY and (teff is None or _within(teff, ROAP_TEFF)):
        return "ROAP"
    if period < DSCT_MAX_PERIOD and (teff is None or _within(teff, FTYPE_TEFF)):
        return "DSCT"
    if (
        independent
        and _within(period, GDOR_PERIODS)
        and teff is not None
        and _within(teff, GDOR_TEFF)
    ):
        return "GDOR"
    if teff is not None and teff < COOL_TEFF and _can_rotate(lc.star, period):
        return "ROT"
    return "VAR"


def ephemeris(lc: LightCurve, detection: Detection) -> Ephemeris | None:
    """Period and epoch the way VSX expects them, binaries at the orbital period and minimum."""
    if detection.eclipse is not None:
        eclipse = detection.eclipse
        return Ephemeris(eclipse.period, None, eclipse.epoch, "minimum")
    primary = detection.primary
    if primary is None:
        return None
    if _is_binary(lc, detection):
        period = 2 * primary.period
        times = _one_cycle(lc, period)
        deepest = float(times[np.argmin(_orbital_model(lc, detection, period)(times))])
        error = 2 * detection.period_error if detection.period_error is not None else None
        return Ephemeris(period, error, deepest, "minimum")
    return Ephemeris(primary.period, detection.period_error, primary.epoch, "maximum")


def rotation(star: Star, period: float, vsini: float | None) -> Rotation | None:
    if star.radius is None:
        return None
    radius = star.radius * constants.R_sun
    # One turn per period at the equator, v = 2 pi R / P.
    equatorial = float((2 * np.pi * radius / (period * u.day)).to_value(u.km / u.s))
    if star.mass is None:
        return Rotation(equatorial, None, vsini)
    # Faster than v = sqrt(G M / R) the equator would fly off.
    breakup = np.sqrt(constants.G * star.mass * constants.M_sun / radius).to_value(u.km / u.s)
    return Rotation(equatorial, float(breakup), vsini)


def _is_binary(lc: LightCurve, detection: Detection) -> bool:
    """Unequal minima show up as a subharmonic, equal ones as harmonics with a flat top."""
    primary = detection.primary
    tolerance = 1.0 / lc.baseline
    others = [f.frequency for f in detection.frequencies[1:]]
    if any(abs(f - primary.frequency / 2) <= tolerance for f in others):
        return True
    harmonics = any(is_harmonic(f, primary.frequency, tolerance, lowest=2) for f in others)
    return harmonics and _flat_share(lc, detection, primary.period) > FLAT_SHARE


def _flat_share(lc: LightCurve, detection: Detection, period: float) -> float:
    model = _orbital_model(lc, detection, period)(_one_cycle(lc, period))
    return float(np.mean(model > model.max() - FLAT_LEVEL * (model.max() - model.min())))


def _orbital_model(
    lc: LightCurve, detection: Detection, period: float
) -> Callable[[np.ndarray], np.ndarray]:
    """Only the harmonics of the period, unrelated signals would shift the minima and the range."""
    tolerance = 1.0 / lc.baseline
    return waves(
        [f for f in detection.frequencies if is_harmonic(f.frequency, 1.0 / period, tolerance)]
    )


def _one_cycle(lc: LightCurve, period: float) -> np.ndarray:
    return lc.time[0] + np.linspace(0.0, period, MINIMUM_SEARCH_STEPS, endpoint=False)


def _can_rotate(star: Star, period: float) -> bool:
    """Spots need a spin below breakup, without a size anything above a tenth of a day will do."""
    spin = rotation(star, period, None)
    if spin is None or spin.breakup_velocity is None:
        return period >= MIN_ROTATION_PERIOD
    return spin.equatorial_velocity < spin.breakup_velocity


def _within(value: float, bounds: tuple[float, float]) -> bool:
    return bounds[0] <= value <= bounds[1]
