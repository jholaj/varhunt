"""Times of minimum of eclipsing binaries, the linear ephemeris they give and the eclipse shape."""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from varhunt.sources.lightcurve import MAD_TO_SIGMA, LightCurve

# Dips closer than this in time belong to one eclipse.
DIP_GAP = 0.3
MIN_DIP_POINTS = 4
FIT_PASSES = 3
MIN_FIT_MINIMA = 3
# Minima off the line by more than this many robust sigma, and at least a few minutes, are false.
OUTLIER_SIGMA = 5.0
OUTLIER_FLOOR = 0.005
# The folded eclipse is measured within this share of the period around its centre.
PROFILE_WINDOW = 0.1
PROFILE_BINS = 400
MIN_BIN_POINTS = 3
# An eclipse starts and ends where it is this share of its depth below the out-of-eclipse level.
CONTACT_SHARE = 0.05
GEOCENTRE = EarthLocation.from_geocentric(0, 0, 0, unit=u.m)


@dataclass(frozen=True)
class Minimum:
    time: float
    depth: float
    sector: int


@dataclass(frozen=True)
class Elements:
    period: float
    period_error: float | None
    epoch: float
    rms_minutes: float
    cycles: int


@dataclass(frozen=True)
class Profile:
    depth: float
    duration: float


def minima(lc: LightCurve, threshold: float, duration: float) -> list[Minimum]:
    """Every dip deeper than the threshold, timed by a parabola through its lower half."""
    low = np.flatnonzero(lc.flux < 1.0 - threshold)
    if not len(low):
        return []
    found = []
    for group in np.split(low, np.flatnonzero(np.diff(lc.time[low]) > DIP_GAP) + 1):
        centre = lc.time[group][np.argmin(lc.flux[group])]
        near = np.abs(lc.time - centre) < duration
        offsets, flux = lc.time[near] - centre, lc.flux[near]
        lower = flux < 1.0 - 0.5 * (1.0 - flux.min())
        covered = offsets.max() > duration / 4 and offsets.min() < -duration / 4
        if lower.sum() < MIN_DIP_POINTS or not covered:
            continue
        # y = a x^2 + b x + c has its vertex at x = -b / 2a and y = c - b^2 / 4a.
        curve, slope, level = np.polyfit(offsets[lower], flux[lower], 2)
        if curve <= 0:
            continue
        shift = -slope / (2 * curve)
        bottom = level - slope**2 / (4 * curve)
        found.append(Minimum(float(centre + shift), float(1.0 - bottom), lc.star.sector))
    return found


def linear_ephemeris(times: list[float], period: float) -> Elements:
    """Least-squares period and epoch with cycles counted from the latest minimum. A dip far off
    the line is a false one, it is dropped and the line fitted again."""
    kept = np.sort(np.asarray(times, dtype=float))
    elements, residual = _fit(kept, period)
    while len(kept) > MIN_FIT_MINIMA:
        # A robust sigma of the O-C, the false dip itself does not inflate it.
        spread = MAD_TO_SIGMA * np.median(np.abs(residual - np.median(residual)))
        worst = int(np.argmax(np.abs(residual)))
        if abs(residual[worst]) <= max(OUTLIER_SIGMA * spread, OUTLIER_FLOOR):
            break
        kept = np.delete(kept, worst)
        elements, residual = _fit(kept, elements.period)
    return elements


def phase_of(times: list[float], elements: Elements) -> float:
    phases = ((np.asarray(times) - elements.epoch) / elements.period) % 1.0
    return float(np.median(phases))


def profile(lcs: list[LightCurve], elements: Elements, phase: float) -> Profile | None:
    """Depth and duration, as a share of the period, of the folded eclipse at this phase."""
    time = np.concatenate([lc.time for lc in lcs])
    flux = np.concatenate([lc.flux for lc in lcs])
    offset = ((time - elements.epoch) / elements.period - phase + 0.5) % 1.0 - 0.5
    near = np.abs(offset) < PROFILE_WINDOW
    edges = np.linspace(-PROFILE_WINDOW, PROFILE_WINDOW, PROFILE_BINS + 1)
    index = np.digitize(offset[near], edges) - 1
    counts = np.bincount(index, minlength=PROFILE_BINS)[:PROFILE_BINS]
    medians = np.array(
        [
            np.median(flux[near][index == i]) if counts[i] >= MIN_BIN_POINTS else np.nan
            for i in range(PROFILE_BINS)
        ]
    )
    centres = 0.5 * (edges[1:] + edges[:-1])
    outside = np.abs(centres) > 0.75 * PROFILE_WINDOW
    if not np.isfinite(medians).any() or not np.isfinite(medians[outside]).any():
        return None
    level = np.nanmedian(medians[outside])
    depth = level - np.nanmin(medians)
    inside = centres[medians < level - CONTACT_SHARE * depth]
    duration = float(inside.max() - inside.min()) if len(inside) else 0.0
    return Profile(depth=float(depth / level), duration=duration)


def heliocentric_date(bjd: float, ra: float, dec: float) -> float:
    """HJD in UTC, which the VSX form asks for, from a barycentric TDB date."""
    where = SkyCoord(ra, dec, unit="deg")
    # BJD adds the barycentric light delay in TDB, HJD the heliocentric one in UTC.
    barycentric = Time(bjd, format="jd", scale="tdb", location=GEOCENTRE)
    observed = barycentric - barycentric.light_travel_time(where, "barycentric")
    for _ in range(FIT_PASSES):
        observed = barycentric - observed.light_travel_time(where, "barycentric")
    return float((observed.utc + observed.light_travel_time(where, "heliocentric")).jd)


def _fit(times: np.ndarray, period: float) -> tuple[Elements, np.ndarray]:
    epoch = times[-1]
    error = None
    for _ in range(FIT_PASSES):
        cycles = np.round((times - epoch) / period)
        if len(times) > 3:
            # Times of minimum follow T = T0 + P E, with E the cycle number.
            (period, epoch), covariance = np.polyfit(cycles, times, 1, cov=True)
            error = float(np.sqrt(covariance[0, 0]))
        elif len(times) > 1:
            period, epoch = np.polyfit(cycles, times, 1)
    residual = times - (epoch + period * cycles)
    elements = Elements(
        period=float(period),
        period_error=error,
        epoch=float(epoch),
        rms_minutes=float(np.std(residual) * 24 * 60),
        cycles=int(cycles.max() - cycles.min()),
    )
    return elements, residual
