"""Synthetic light curves, QLP files and TESScut cubes for the tests."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from varhunt.output.report import CANDIDATE, Result
from varhunt.sources.lightcurve import LightCurve, Star

START = 3000.0
CADENCE = 200 / 86400
GAP = (11.5, 12.5)
PIXEL_DEG = 21 / 3600


def make_star(tic: int = 1, tmag: float = 11.0, teff: float | None = 7000.0) -> Star:
    return Star(tic=tic, ra=100.0, dec=20.0, tmag=tmag, teff=teff, sector=72)


def make_time(days: float = 24.0, cadence: float = CADENCE) -> np.ndarray:
    time = START + np.arange(0, days, cadence)
    offset = time - START
    return time[(offset < GAP[0]) | (offset > GAP[1])]


def make_light_curve(
    signal: Callable[[np.ndarray], np.ndarray] | None = None,
    noise: float = 1e-3,
    seed: int = 0,
    star: Star | None = None,
) -> LightCurve:
    rng = np.random.default_rng(seed)
    time = make_time()
    flux = 1.0 + rng.normal(0.0, noise, len(time))
    if signal is not None:
        flux = flux + signal(time)
    orbit = np.where(time - START < GAP[0], 1, 2)
    return LightCurve(star=star or make_star(), time=time, flux=flux, orbit=orbit)


def sine(frequency: float, amplitude: float, phase: float = 0.0):
    return lambda time: amplitude * np.sin(2 * np.pi * frequency * time + phase)


def eclipses(period: float, duration: float, depth: float, epoch: float = START + 0.7):
    def signal(time: np.ndarray) -> np.ndarray:
        distance = np.abs(((time - epoch) / period + 0.5) % 1.0 - 0.5) * period
        return -depth * (distance < duration / 2)

    return signal


def write_qlp(
    path: Path,
    lc: LightCurve,
    quality: np.ndarray | None = None,
    teff: float | None = 7000.0,
    flux_column: str = "SAP_FLUX",
) -> Path:
    primary = fits.PrimaryHDU()
    primary.header.update(
        TICID=lc.star.tic,
        RA_OBJ=lc.star.ra,
        DEC_OBJ=lc.star.dec,
        TESSMAG=lc.star.tmag,
        SECTOR=lc.star.sector,
    )
    if teff is not None:
        primary.header["TEFF"] = teff
    columns = [
        fits.Column(name="TIME", format="D", array=lc.time),
        fits.Column(name=flux_column, format="E", array=lc.flux),
        fits.Column(name="ORBITID", format="J", array=lc.orbit),
        fits.Column(
            name="QUALITY",
            format="J",
            array=quality if quality is not None else np.zeros(len(lc.time), dtype=int),
        ),
    ]
    fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(path)
    return path


def make_wcs(size: int = 15, ra: float = 100.0, dec: float = 20.0) -> WCS:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [size // 2 + 1, size // 2 + 1]
    wcs.wcs.cdelt = [-PIXEL_DEG, PIXEL_DEG]
    return wcs


def make_cube(
    time: np.ndarray,
    stars: list[tuple[float | np.ndarray, float | np.ndarray, float, np.ndarray]],
    sky_signal: np.ndarray,
    size: int = 15,
    psf_sigma: float = 0.8,
    noise: float = 2.0,
    seed: int = 0,
) -> np.ndarray:
    """Pixels of stars given as (x, y, flux, relative variation) on a sky with its own signal.

    Positions may be arrays over time to make a star move.
    """
    rng = np.random.default_rng(seed)
    ys, xs = np.mgrid[0:size, 0:size]
    cube = np.full((len(time), size, size), 100.0) + sky_signal[:, None, None]
    for x, y, flux, variation in stars:
        column = np.broadcast_to(np.asarray(x, dtype=float), time.shape)[:, None, None]
        row = np.broadcast_to(np.asarray(y, dtype=float), time.shape)[:, None, None]
        psf = np.exp(-((xs - column) ** 2 + (ys - row) ** 2) / (2 * psf_sigma**2))
        psf /= psf.sum(axis=(1, 2), keepdims=True)
        cube += flux * (1.0 + variation)[:, None, None] * psf
    return cube + rng.normal(0.0, noise, cube.shape)


def candidate(**overrides) -> Result:
    values = {
        "tic": 194990179,
        "ra": 100.1258,
        "dec": 20.0667,
        "tmag": 11.59,
        "teff": None,
        "file": "lc.fits",
        "n_points": 8641,
        "rms_ppt": 7.9,
        "period": 0.5695,
        "period_error": 0.00004,
        "epoch": 3300.12,
        "epoch_kind": "maximum",
        "light_max_ppt": 10.9,
        "light_min_ppt": -10.9,
        "amplitude_ppt": 10.9,
        "snr": 45.0,
        "periods": "0.5695000",
        "type_hint": "ROT",
        "status": CANDIDATE,
        "gaia_id": "3371989308108214144",
    }
    return Result(**values | overrides)


def sure_candidate(**overrides) -> Result:
    evidence = {
        "vet_verdict": "on_target",
        "vet_offset_px": 0.1,
        "vet_centroid_error_px": 0.05,
        "other_sectors": "43;44;45",
        "confirmed_sectors": "43;45",
        "other_verdicts": "43:on_target;45:no_signal",
        "survives_wider_detrending": True,
    }
    return candidate(**evidence | overrides)
