"""Independent evidence for a candidate, from other TESS sectors, other catalogues and Gaia."""

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from varhunt.analysis.detect import Detection, amplitude_spectrum
from varhunt.sources.archive import retried
from varhunt.sources.catalogs import GAIA_SOURCES
from varhunt.sources.lightcurve import LightCurve

CATALOG_RADIUS_ARCSEC = 5.0
# Entries this close belong to other stars, but they may have caught the candidate's light.
NEARBY_CATALOG_ARCSEC = 30.0
# Only the strongest signals of another sector count as repeating the candidate.
DOMINANT_SIGNALS = 2
# A catalogue whose name or description has one of these words is about variability.
VARIABILITY_WORDS = (
    "variab",
    "eclips",
    "pulsat",
    "periodic",
    "rotation period",
    "light curve",
    "light-curve",
    "flare",
    "delta sct",
    "rr lyr",
    "rrab",
    "rrl",
    "cepheid",
    "gcvs",
)
NOT_VARIABILITY_WORDS = ("non-variable", "upper limit", "spurious")
# SIMBAD object types of variable stars.
VARIABLE_TYPES = frozenset(
    {
        "V*",
        "EB*",
        "dS*",
        "RR*",
        "Ro*",
        "BY*",
        "RS*",
        "Pu*",
        "gD*",
        "El*",
        "WU*",
        "Al*",
        "bL*",
        "Ce*",
        "cC*",
        "LP*",
        "Mi*",
        "sr*",
        "Ir*",
        "Er*",
        "Fl*",
        "SX*",
        "ZZ*",
        "a2*",
        "RV*",
    }
)
# Search this many single-sector resolutions around the frequency when joining sectors.
REFINE_WINDOW = 1.0
COARSE_POINTS = 101
FINE_POINTS = 201
# Sectors closer than this form one unbroken run.
RUN_GAP_DAYS = 10.0
# The cycle count across a gap is safe when one alias fits within this many errors.
ALIAS_SIGMA = 3.0
GAIA_PARAMETERS = "I/355/paramp"
VSINI_ARCSEC = 2.0
SCATTER_FIELD_ARCSEC = 300.0
SCATTER_MAG_WINDOW = 0.5
SCATTER_MIN_COMPARISON = 5
# Gaia photometry of sources with a close companion is unreliable.
MAX_RUWE = 1.4
MAX_MULTI_PEAK_PERCENT = 10


@dataclass(frozen=True)
class Refinement:
    frequency: float
    error: float
    span: float
    ambiguous: bool


@dataclass(frozen=True)
class GaiaQuality:
    """Astrometric fit quality and the epoch scatter of the G flux relative to the field."""

    ruwe: float | None
    scatter: float | None
    scatter_reliable: bool


def repeats(signal: float, candidate: LightCurve, lc: LightCurve, detection: Detection) -> bool:
    """Whether another sector's strongest signals repeat the reported one, or its double or half."""
    tolerance = 1.0 / min(lc.baseline, candidate.baseline)
    dominant = [f.frequency for f in detection.frequencies[:DOMINANT_SIGNALS]]
    if detection.eclipse is not None:
        dominant.append(1.0 / detection.eclipse.period)
    # A binary shows its orbit or half of it, so double and half count as the same signal.
    return any(abs(f - factor * signal) <= tolerance for f in dominant for factor in (0.5, 1, 2))


def refine_frequency(lcs: list[LightCurve], frequency: float) -> Refinement:
    """The frequency from all sectors together. The longest unbroken run of sectors fixes it well
    enough to count cycles across the gaps, the flag is raised when it is not."""
    runs = _unbroken_runs(lcs)
    anchor = max(runs, key=lambda run: run[-1].time[-1] - run[0].time[0])
    single = min(lc.baseline for lc in lcs)
    # One cycle more or less over a span T shifts f by 1 / T, the anchor must be sharper.
    anchored, anchor_error = _peak(anchor, frequency, REFINE_WINDOW / single)
    joined, error = _peak(lcs, anchored, ALIAS_SIGMA * anchor_error)
    everything = _joined(lcs)[0]
    span = float(everything[-1] - everything[0])
    return Refinement(joined, error, span, ambiguous=1.0 / span < 2 * ALIAS_SIGMA * anchor_error)


@retried
def measured_vsini(ra: float, dec: float) -> float | None:
    """Gaia line broadening, or the hot-star v sin i of the Apsis chain."""
    where = SkyCoord(ra, dec, unit="deg")
    radius = VSINI_ARCSEC * u.arcsec
    for catalog, column in ((GAIA_SOURCES, "Vbroad"), (GAIA_PARAMETERS, "vsini-HS")):
        tables = Vizier(columns=[column], row_limit=1).query_region(
            where, radius=radius, catalog=catalog
        )
        if tables and len(tables[0]) and not np.ma.is_masked(tables[0][column][0]):
            return float(tables[0][column][0])
    return None


@retried
def variability_catalogs(
    ra: float, dec: float, radius_arcsec: float = CATALOG_RADIUS_ARCSEC
) -> list[str]:
    """VizieR catalogues about variability that list something within the radius."""
    tables = Vizier(row_limit=1).query_region(
        SkyCoord(ra, dec, unit="deg"), radius=radius_arcsec * u.arcsec
    )
    return [
        name
        for name, table in zip(tables.keys(), tables, strict=True)
        if mentions_variability(f"{name} {table.meta.get('description', '')}")
    ]


def mentions_variability(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in VARIABILITY_WORDS) and not any(
        word in lowered for word in NOT_VARIABILITY_WORDS
    )


@retried
def simbad_objects(ra: float, dec: float) -> list[tuple[str, str]]:
    """Main identifier and object type of every SIMBAD object nearby, nearest first."""
    simbad = Simbad()
    simbad.add_votable_fields("otype")
    where = SkyCoord(ra, dec, unit="deg")
    found = simbad.query_region(where, radius=CATALOG_RADIUS_ARCSEC * u.arcsec)
    if found is None or not len(found):
        return []
    distance = where.separation(SkyCoord(found["ra"], found["dec"], unit="deg")).arcsec
    return [
        (" ".join(str(found["main_id"][i]).split()), str(found["otype"][i]))
        for i in np.argsort(distance)
    ]


@retried
def gaia_quality(source_id: int, ra: float, dec: float) -> GaiaQuality | None:
    vizier = Vizier(
        columns=["Source", "Gmag", "FG", "e_FG", "o_Gmag", "RUWE", "IPDfmp"], row_limit=-1
    )
    tables = vizier.query_region(
        SkyCoord(ra, dec, unit="deg"), radius=SCATTER_FIELD_ARCSEC * u.arcsec, catalog=GAIA_SOURCES
    )
    if not tables:
        return None
    field = tables[0]
    own = np.flatnonzero(np.asarray(field["Source"]) == source_id)
    if not len(own):
        return None
    index = own[0]
    ruwe = float(np.ma.filled(field["RUWE"][index], np.nan))
    reliable = bool(
        ruwe <= MAX_RUWE and np.ma.filled(field["IPDfmp"][index], 100) <= MAX_MULTI_PEAK_PERCENT
    )
    # e_FG is the error of the mean, sqrt(n) e_FG / FG is the epoch scatter relative to the flux.
    proxy = np.asarray(np.sqrt(field["o_Gmag"]) * field["e_FG"] / field["FG"], dtype=float)
    similar = (np.abs(field["Gmag"] - field["Gmag"][index]) < SCATTER_MAG_WINDOW) & np.isfinite(
        proxy
    )
    similar[index] = False
    scatter = None
    if np.isfinite(proxy[index]) and similar.sum() >= SCATTER_MIN_COMPARISON:
        scatter = float(proxy[index] / np.median(proxy[similar]))
    return GaiaQuality(ruwe if np.isfinite(ruwe) else None, scatter, reliable)


def is_variable_type(otype: str) -> bool:
    return otype in VARIABLE_TYPES


def _unbroken_runs(lcs: list[LightCurve]) -> list[list[LightCurve]]:
    ordered = sorted(lcs, key=lambda lc: lc.time[0])
    runs = [[ordered[0]]]
    for lc in ordered[1:]:
        if lc.time[0] - runs[-1][-1].time[-1] <= RUN_GAP_DAYS:
            runs[-1].append(lc)
        else:
            runs.append([lc])
    return runs


def _joined(lcs: list[LightCurve]) -> tuple[np.ndarray, np.ndarray]:
    time = np.concatenate([lc.time for lc in lcs])
    flux = np.concatenate([lc.flux - 1.0 for lc in lcs])
    order = np.argsort(time)
    return time[order], flux[order]


def _peak(lcs: list[LightCurve], centre: float, half_width: float) -> tuple[float, float]:
    """Highest peak near a frequency, found on a coarse then a fine grid, and its formal error
    over the span of the light curves, never below the fine grid step."""
    time, flux = _joined(lcs)
    span = float(time[-1] - time[0])
    points = max(COARSE_POINTS, int(2 * half_width * span * 20) + 1)
    coarse = np.linspace(centre - half_width, centre + half_width, points)
    best = float(coarse[np.argmax(amplitude_spectrum(time, flux, coarse))])
    step = coarse[1] - coarse[0]
    fine = np.linspace(best - step, best + step, FINE_POINTS)
    best = float(fine[np.argmax(amplitude_spectrum(time, flux, fine, method="cython"))])
    phase = 2 * np.pi * best * time
    design = np.column_stack([np.sin(phase), np.cos(phase), np.ones_like(time)])
    coefficients, *_ = np.linalg.lstsq(design, flux, rcond=None)
    amplitude = float(np.hypot(coefficients[0], coefficients[1]))
    noise = float(np.std(flux - design @ coefficients))
    # Montgomery and O'Donoghue (1999): sigma_f = sqrt(6 / N) sigma / (pi T A).
    error = np.sqrt(6.0 / len(time)) * noise / (np.pi * span * amplitude)
    return best, float(max(error, fine[1] - fine[0]))
