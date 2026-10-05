"""QLP and TESS-SPOC light curves, loaded clean and normalised, and their gentle detrending."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits

# Raw aperture flux, the QLP detrended flux suppresses periods above a few tenths of a day.
FLUX_COLUMN = "SAP_FLUX"
# The mission pipeline's own systematics correction, VSX moderators know it best.
SPOC_FLUX_COLUMN = "PDCSAP_FLUX"
SEGMENT_GAP = 0.5
# Every stretch after a gap starts with a thermal ramp that takes about a day to settle.
SETTLE_DAYS = 1.0
TREND_BIN = 0.1
# Only upward outliers are clipped so eclipses survive.
OUTLIER_SIGMA = 5.0
# Gaussian noise has sigma = 1.4826 MAD, which eclipses and outliers barely move.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Star:
    tic: int
    ra: float
    dec: float
    tmag: float
    teff: float | None
    sector: int
    radius: float | None = None
    mass: float | None = None


@dataclass(frozen=True)
class LightCurve:
    star: Star
    time: np.ndarray
    flux: np.ndarray
    orbit: np.ndarray

    @property
    def baseline(self) -> float:
        return float(self.time[-1] - self.time[0]) if len(self.time) > 1 else 0.0

    @property
    def cadence(self) -> float:
        return float(np.median(np.diff(self.time)))

    @property
    def nyquist(self) -> float:
        return 0.5 / self.cadence


def load_qlp(path: Path) -> LightCurve:
    """Good cadences of the raw aperture flux over its median, without the settling days."""
    return _load(path, FLUX_COLUMN, orbits=True)


def load_spoc(path: Path) -> LightCurve:
    """Good cadences of a TESS-SPOC light curve's PDC flux over its median."""
    return _load(path, SPOC_FLUX_COLUMN, orbits=False)


def detrended(lc: LightCurve, window: float, model: np.ndarray | None = None) -> LightCurve:
    """Divides out slow drifts, estimated with a known signal model removed so it stays intact."""
    if not len(lc.flux):
        return lc
    signal = 1.0 + model if model is not None else 1.0
    # The trend is fitted to flux / (1 + model), so it follows the drifts and not the star.
    flux = lc.flux / trend(lc.time, lc.flux / signal, window)
    keep = ~_upper_outliers(flux)
    return LightCurve(star=lc.star, time=lc.time[keep], flux=flux[keep], orbit=lc.orbit[keep])


def trend(time: np.ndarray, flux: np.ndarray, window: float) -> np.ndarray:
    """Running median of short-bin medians, computed separately for every stretch between gaps."""
    result = np.empty_like(flux)
    segments = np.cumsum(np.r_[0, np.diff(time) > SEGMENT_GAP])
    for segment in np.unique(segments):
        inside = segments == segment
        times, values = time[inside], flux[inside]
        bins = np.floor((times - times[0]) / TREND_BIN).astype(int)
        starts = np.flatnonzero(np.r_[True, np.diff(bins) > 0])
        centres = np.array([chunk.mean() for chunk in np.split(times, starts[1:])])
        # Medians of 0.1 d bins first, so an outlier or a short eclipse cannot move the trend.
        medians = np.array([np.median(chunk) for chunk in np.split(values, starts[1:])])
        smooth = np.array(
            [np.median(medians[np.abs(centres - centre) <= window / 2]) for centre in centres]
        )
        result[inside] = np.interp(times, centres, smooth)
    return result


def _load(path: Path, column: str, orbits: bool) -> LightCurve:
    with fits.open(path) as hdul:
        header = hdul[0].header
        data = hdul[1].data
        time = np.asarray(data["TIME"], dtype=float)
        flux = np.asarray(data[column], dtype=float)
        orbit = np.asarray(data["ORBITID"], dtype=int) if orbits else np.zeros(len(time), int)
        quality = np.asarray(data["QUALITY"])

    star = Star(
        tic=int(header["TICID"]),
        ra=float(header["RA_OBJ"]),
        dec=float(header["DEC_OBJ"]),
        tmag=float(header["TESSMAG"]),
        teff=_positive_or_none(header.get("TEFF")),
        sector=int(header["SECTOR"]),
        radius=_positive_or_none(header.get("RADIUS")),
        mass=_positive_or_none(header.get("MASS")),
    )
    good = (quality == 0) & np.isfinite(time) & np.isfinite(flux) & (flux > 0)
    order = np.argsort(time[good])
    time, flux, orbit = time[good][order], flux[good][order], orbit[good][order]
    settled = ~_settling(time)
    time, flux, orbit = time[settled], flux[settled], orbit[settled]
    if len(flux):
        flux = flux / np.median(flux)
    return LightCurve(star=star, time=time, flux=flux, orbit=orbit)


def _settling(time: np.ndarray) -> np.ndarray:
    if not len(time):
        return np.zeros(0, dtype=bool)
    starts = time[np.r_[True, np.diff(time) > SEGMENT_GAP]]
    since = time - starts[np.searchsorted(starts, time, side="right") - 1]
    return since < SETTLE_DAYS


def _upper_outliers(flux: np.ndarray) -> np.ndarray:
    median = np.median(flux)
    sigma = MAD_TO_SIGMA * np.median(np.abs(flux - median))
    return flux > median + OUTLIER_SIGMA * sigma


def _positive_or_none(value: object) -> float | None:
    # QLP writes -1 where the TIC has no value.
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) and number > 0 else None
