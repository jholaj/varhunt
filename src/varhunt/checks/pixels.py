"""Pixel-level localisation, which star in a TESScut cutout carries a periodic signal."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.mast import Tesscut

from varhunt.sources.archive import retried
from varhunt.sources.catalogs import Neighbour, counterpart
from varhunt.sources.lightcurve import MAD_TO_SIGMA

CUTOUT_SIZE = 15
SEGMENT_GAP = 1.0
MIN_FINITE_SHARE = 0.99
# Pixels this dim in the median image are sky, they define the common-mode signal.
FAINT_PERCENTILE = 30
MIN_FRAMES = 100
# Null templates sit between this many resolutions and a c/d from the signal, as the local noise.
NULL_TEMPLATES = 20
MIN_NULL_SHIFT = 3.0
MAX_NULL_SHIFT = 1.0
MAX_NULL_STRETCH = 0.5
# Signal is summed around its peak where pointing jitter cancels, small apertures suit faint stars.
APERTURE_RADII_PX = (1.0, 2.0, 3.0)
MIN_SIGNIFICANCE = 6.0
ON_TARGET_PX = 0.5
OFF_TARGET_PX = 1.0
SOURCE_RADIUS_PX = 1.0
# Brightness changes are positive in every pixel, image motion leaves a negative lobe by the peak.
DIPOLE_RADIUS_PX = 2.5
MAX_DIPOLE = 0.3
MOTION_SIGMA = 3.0
# The signal is located by a round Gaussian fitted to this many pixels around its peak.
PSF_HALF_WIDTH = 3
PSF_START_PX = 0.9
PSF_WIDTH_PX = (0.5, 1.5)
PSF_ITERATIONS = 30
PSF_CONVERGED_PX = 1e-4
PSF_MAX_SHIFT_PX = 1.5
# Median distance from the centre of a round Gaussian scatter, in its per-axis sigma.
RAYLEIGH_MEDIAN = float(np.sqrt(2 * np.log(2)))


@dataclass(frozen=True)
class Cutout:
    time: np.ndarray
    flux: np.ndarray
    wcs: WCS


@dataclass(frozen=True)
class Localisation:
    verdict: str
    significance: float
    dipole: float
    offset_px: float
    centroid_error_px: float
    centroid: tuple[float, float]
    target: tuple[float, float]
    likely_source: Neighbour | None
    companions: tuple[Neighbour, ...]
    signal_map: np.ndarray
    sources: tuple[tuple[Neighbour, float, float], ...]


def cutout_path(ra: float, dec: float, sector: int, cache_dir: Path) -> Path:
    return cache_dir / "tesscut" / f"{ra:.5f}{dec:+.5f}_s{sector:04d}_{CUTOUT_SIZE}.fits"


def cached_cutouts(ra: float, dec: float, cache_dir: Path) -> list[Path]:
    return sorted((cache_dir / "tesscut").glob(f"{ra:.5f}{dec:+.5f}_s*_{CUTOUT_SIZE}.fits"))


@retried
def fetch_cutout(ra: float, dec: float, sector: int, cache_dir: Path) -> Path:
    path = cutout_path(ra, dec, sector, cache_dir)
    if not path.exists():
        hduls = Tesscut.get_cutouts(
            coordinates=SkyCoord(ra, dec, unit="deg"), size=CUTOUT_SIZE, sector=sector
        )
        if not hduls:
            raise FileNotFoundError(f"TESScut has no sector {sector} data at {ra} {dec}")
        hdul = hduls[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".part")
        hdul.writeto(partial, overwrite=True)
        partial.replace(path)
    return path


def load_cutout(path: Path) -> Cutout:
    with fits.open(path) as hdul:
        data = hdul[1].data
        time = np.asarray(data["TIME"], dtype=float)
        good = (np.asarray(data["QUALITY"]) == 0) & np.isfinite(time)
        flux = np.asarray(data["FLUX"], dtype=float)[good]
        wcs = WCS(hdul[2].header)
    return Cutout(time=time[good], flux=flux, wcs=wcs)


def localise(
    cutout: Cutout,
    ra: float,
    dec: float,
    template: Callable[[np.ndarray], np.ndarray],
    frequency: float,
    candidates: list[Neighbour],
) -> Localisation:
    """Fits the signal shape to every pixel, removes the sky share and finds what is left. Noise
    comes from the same shape stretched to nearby frequencies, so jitter and drifts count too."""
    coefficients = _coefficients(cutout.time, cutout.flux, template, frequency)
    target_x, target_y = cutout.wcs.world_to_pixel(SkyCoord(ra, dec, unit="deg"))
    target = (float(target_x), float(target_y))
    sources = _place(cutout.wcs, candidates)
    image_shape = cutout.flux.shape[1:]
    valid = np.isfinite(coefficients[0])
    if not valid.any():
        empty = coefficients[0].reshape(image_shape)
        return Localisation(
            verdict="no_signal",
            significance=0.0,
            dipole=0.0,
            offset_px=float("nan"),
            centroid_error_px=float("nan"),
            centroid=target,
            target=target,
            likely_source=None,
            companions=(),
            signal_map=empty,
            sources=sources,
        )

    median_image = np.nanmedian(cutout.flux, axis=0).ravel()
    faint = valid & (median_image <= np.nanpercentile(median_image[valid], FAINT_PERCENTILE))
    coefficients = coefficients - np.median(coefficients[:, faint], axis=1)[:, None]
    signal, nulls = coefficients[0], coefficients[1:]
    # Pixel to pixel scatter of the sky at the signal frequency is a floor under every pixel.
    floor = MAD_TO_SIGMA * np.median(np.abs(signal[faint] - np.median(signal[faint])))
    sigma = np.hypot(_spread(nulls), floor)

    # Brightness changes add flux in every pixel of the star, so the signal map peaks on it.
    peak = int(np.nanargmax(signal))
    peak_y, peak_x = np.unravel_index(peak, image_shape)
    rows, cols = np.indices(image_shape)
    distance = np.hypot(rows - peak_y, cols - peak_x).ravel()
    significance = max(
        _aperture_significance(signal, nulls, floor, (distance <= radius) & valid)
        for radius in APERTURE_RADII_PX
    )
    near_peak = np.flatnonzero((distance <= DIPOLE_RADIUS_PX) & valid)
    lobe = near_peak[np.argmin(signal[near_peak])]
    dipole = float(-signal[lobe] / signal[peak])
    motion = dipole > MAX_DIPOLE and -signal[lobe] >= MOTION_SIGMA * sigma[lobe]
    signal_map = signal.reshape(image_shape)
    sigma_map = sigma.reshape(image_shape)
    centroid = _centroid(signal_map, sigma_map, int(peak_y), int(peak_x))
    offset = float(np.hypot(centroid[0] - target[0], centroid[1] - target[1]))
    null_maps = nulls.reshape(-1, *image_shape)
    centroid_error = _centroid_error(
        signal_map, sigma_map, null_maps, int(peak_y), int(peak_x), centroid
    )

    itself = counterpart(ra, dec, candidates)
    companions = tuple(
        n
        for n, x, y in sources
        if (itself is None or n.source_id != itself.source_id)
        and np.hypot(x - centroid[0], y - centroid[1]) <= ON_TARGET_PX
    )
    return Localisation(
        verdict=_verdict(significance, motion, offset),
        significance=significance,
        dipole=dipole,
        offset_px=offset,
        centroid_error_px=centroid_error,
        centroid=centroid,
        target=target,
        likely_source=_likely_source(sources, centroid),
        companions=companions,
        signal_map=signal_map,
        sources=sources,
    )


def _coefficients(
    time: np.ndarray,
    cube: np.ndarray,
    template: Callable[[np.ndarray], np.ndarray],
    frequency: float,
) -> np.ndarray:
    """Template and null template fits per pixel, one row each, NaN for unusable pixels."""
    pixels = cube.reshape(len(time), -1)
    coefficients = np.full((1 + 2 * NULL_TEMPLATES, pixels.shape[1]), np.nan)
    alive = np.any(np.isfinite(pixels), axis=1)
    time, pixels = time[alive], pixels[alive]
    if not len(time):
        return coefficients
    usable = np.mean(np.isfinite(pixels), axis=0) >= MIN_FINITE_SHARE
    rows = np.all(np.isfinite(pixels[:, usable]), axis=1)
    time, pixels = time[rows], pixels[rows][:, usable]
    if not usable.any() or len(time) < MIN_FRAMES:
        return coefficients

    # Removing the nuisance part from pixels and templates leaves the fit c = sum(t y) / sum(t t).
    basis = np.linalg.qr(_nuisance(time))[0]
    shapes = _shapes(time, template, frequency)
    pixels = pixels - basis @ (basis.T @ pixels)
    shapes = shapes - basis @ (basis.T @ shapes)
    coefficients[:, usable] = (shapes.T @ pixels) / np.sum(shapes**2, axis=0)[:, None]
    return coefficients


def _nuisance(time: np.ndarray) -> np.ndarray:
    segments = np.cumsum(np.r_[0, np.diff(time) > SEGMENT_GAP])
    columns = []
    for segment in np.unique(segments):
        inside = segments == segment
        centred = np.where(inside, time - time[inside].mean(), 0.0)
        columns += [inside.astype(float), centred, centred**2]
    return np.column_stack(columns)


def _shapes(
    time: np.ndarray, template: Callable[[np.ndarray], np.ndarray], frequency: float
) -> np.ndarray:
    span = time[-1] - time[0]
    middle = 0.5 * (time[0] + time[-1])
    closest = MIN_NULL_SHIFT / span
    farthest = max(min(MAX_NULL_SHIFT, MAX_NULL_STRETCH * frequency), 2 * closest)
    shifts = np.linspace(closest, farthest, NULL_TEMPLATES)
    # A template stretched by 1 + e has its frequencies at f (1 + e), same shape but off the signal.
    stretches = np.r_[shifts, -shifts] / frequency
    stretched = [template(middle + (time - middle) * (1 + stretch)) for stretch in stretches]
    return np.column_stack([template(time), *stretched])


def _aperture_significance(
    signal: np.ndarray, nulls: np.ndarray, floor: float, aperture: np.ndarray
) -> float:
    # Noise of the sum from the null fits summed the same way, the sky floor added in quadrature.
    noise = np.hypot(_spread(nulls[:, aperture].sum(axis=1)), floor * np.sqrt(aperture.sum()))
    return float(signal[aperture].sum() / noise) if noise > 0 else float("inf")


def _spread(values: np.ndarray) -> np.ndarray:
    # Null fits have zero mean, so 1.4826 times their median absolute value is their sigma.
    return MAD_TO_SIGMA * np.median(np.abs(values), axis=0)


def _centroid(
    signal: np.ndarray, sigma: np.ndarray, peak_y: int, peak_x: int
) -> tuple[float, float]:
    """Centre of a round Gaussian fitted around the peak, the weighted mean where it fails."""
    start = _weighted_centroid(signal, peak_y, peak_x)
    rows = slice(max(peak_y - PSF_HALF_WIDTH, 0), min(peak_y + PSF_HALF_WIDTH + 1, signal.shape[0]))
    cols = slice(max(peak_x - PSF_HALF_WIDTH, 0), min(peak_x + PSF_HALF_WIDTH + 1, signal.shape[1]))
    ys, xs = np.mgrid[rows, cols]
    values, weights = signal[rows, cols], 1.0 / sigma[rows, cols]
    usable = np.isfinite(values) & np.isfinite(weights)
    xs, ys, values, weights = xs[usable], ys[usable], values[usable], weights[usable]
    params = np.array([signal[peak_y, peak_x], *start, PSF_START_PX, 0.0])
    for _ in range(PSF_ITERATIONS):
        amplitude, x0, y0, width, background = params
        dx, dy = xs - x0, ys - y0
        # Gauss-Newton step for a exp(-r^2 / 2 s^2) + b, each pixel weighted by 1 / sigma.
        profile = np.exp(-(dx**2 + dy**2) / (2 * width**2))
        bump = amplitude * profile
        jacobian = np.column_stack(
            [
                profile,
                bump * dx / width**2,
                bump * dy / width**2,
                bump * (dx**2 + dy**2) / width**3,
                np.ones_like(profile),
            ]
        )
        residual = values - bump - background
        step = np.linalg.lstsq(jacobian * weights[:, None], residual * weights, rcond=None)[0]
        params = params + step
        if not PSF_WIDTH_PX[0] <= params[3] <= PSF_WIDTH_PX[1] or params[0] <= 0:
            return start
        if np.hypot(step[1], step[2]) < PSF_CONVERGED_PX:
            break
    x0, y0 = params[1], params[2]
    if np.hypot(x0 - peak_x, y0 - peak_y) > PSF_MAX_SHIFT_PX:
        return start
    return float(x0), float(y0)


def _weighted_centroid(signal: np.ndarray, peak_y: int, peak_x: int) -> tuple[float, float]:
    rows = slice(max(peak_y - 1, 0), min(peak_y + 2, signal.shape[0]))
    cols = slice(max(peak_x - 1, 0), min(peak_x + 2, signal.shape[1]))
    weights = np.nan_to_num(np.clip(signal[rows, cols], 0.0, None))
    ys, xs = np.mgrid[rows, cols]
    total = weights.sum()
    if total <= 0:
        return float(peak_x), float(peak_y)
    return float((xs * weights).sum() / total), float((ys * weights).sum() / total)


def _centroid_error(
    signal_map: np.ndarray,
    sigma_map: np.ndarray,
    null_maps: np.ndarray,
    peak_y: int,
    peak_x: int,
    centroid: tuple[float, float],
) -> float:
    """Per-axis scatter of the centroid when null fits are added as fresh noise."""
    moved = np.array(
        [_centroid(signal_map + null, sigma_map, peak_y, peak_x) for null in null_maps]
    )
    distance = np.hypot(moved[:, 0] - centroid[0], moved[:, 1] - centroid[1])
    return float(np.median(distance) / RAYLEIGH_MEDIAN)


def _place(wcs: WCS, candidates: list[Neighbour]) -> tuple[tuple[Neighbour, float, float], ...]:
    if not candidates:
        return ()
    coords = SkyCoord([n.ra for n in candidates], [n.dec for n in candidates], unit="deg")
    xs, ys = wcs.world_to_pixel(coords)
    return tuple((n, float(x), float(y)) for n, x, y in zip(candidates, xs, ys, strict=True))


def _likely_source(
    sources: tuple[tuple[Neighbour, float, float], ...], centroid: tuple[float, float]
) -> Neighbour | None:
    if not sources:
        return None
    distances = [(n, float(np.hypot(x - centroid[0], y - centroid[1]))) for n, x, y in sources]
    near = [n for n, distance in distances if distance <= SOURCE_RADIUS_PX]
    if near:
        return min(near, key=lambda n: n.gmag)
    return min(distances, key=lambda item: item[1])[0]


def _verdict(significance: float, motion: bool, offset: float) -> str:
    if significance < MIN_SIGNIFICANCE:
        return "no_signal"
    if motion:
        return "motion"
    if offset <= ON_TARGET_PX:
        return "on_target"
    if offset >= OFF_TARGET_PX:
        return "off_target"
    return "ambiguous"
