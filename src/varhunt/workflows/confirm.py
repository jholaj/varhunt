"""Independent evidence for the candidates that passed the pixel check."""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from varhunt.analysis.classify import rotation
from varhunt.analysis.detect import (
    MIN_POINTS,
    Detection,
    cleaned,
    detect,
    survives_wider_detrending,
)
from varhunt.checks.evidence import (
    NEARBY_CATALOG_ARCSEC,
    gaia_quality,
    is_variable_type,
    measured_vsini,
    refine_frequency,
    repeats,
    simbad_objects,
    variability_catalogs,
)
from varhunt.config import Resources, Run
from varhunt.output.plots import sectors_figure
from varhunt.output.report import CANDIDATE, Result, read_results
from varhunt.sources.lightcurve import LightCurve, load_qlp
from varhunt.sources.mast import download, light_curves_of
from varhunt.workflows.shared import light_curve, locate, write

log = logging.getLogger(__name__)

OTHER_SECTORS = 3
CONFIRM_THREADS = 4
# Other sectors shorter than this cannot pin a frequency down.
MIN_BASELINE = 5.0
PASSED_VERDICTS = ("on_target", "ambiguous")
HARMONIC_SLACK = 0.05


def confirm(run: Run, resources: Resources) -> list[Result]:
    """Other sectors, catalogues and Gaia for every candidate that passed the pixel check."""
    results = read_results(run.stars_csv)
    chosen = [r for r in results if r.status == CANDIDATE and r.vet_verdict in PASSED_VERDICTS]
    with ThreadPoolExecutor(CONFIRM_THREADS) as pool:
        list(pool.map(partial(_confirm_safely, run, resources), chosen))
    write(run, results)
    return chosen


def _confirm_safely(run: Run, resources: Resources, result: Result) -> None:
    try:
        _confirm_one(run, resources, result)
    except OSError:
        # Half gathered evidence must not look like a clean record, so the candidate stays unsure.
        log.exception("confirming TIC %d failed", result.tic)
        result.confirmed_sectors = ""
        result.other_verdicts = ""


def _confirm_one(run: Run, resources: Resources, result: Result) -> None:
    log.info("confirming TIC %d", result.tic)
    raw = light_curve(resources, result)
    detection = detect(raw, run.settings)
    signal = detection.signal_frequency
    if signal is None:
        return
    result.survives_wider_detrending = survives_wider_detrending(raw, detection, run.settings)
    products = [p for p in light_curves_of(result.tic) if p.sector != run.sector]
    latest = sorted(products, key=lambda p: p.sector)[-OTHER_SECTORS:]
    paths = download(latest, resources.cache_dir, resources.workers)
    others = [load_qlp(path) for path in paths.values()]
    usable = [o for o in others if len(o.time) >= MIN_POINTS and o.baseline >= MIN_BASELINE]
    checked = [(o, detect(o, run.settings)) for o in usable]
    confirming = [(o, d) for o, d in checked if repeats(signal, raw, o, d)]
    result.other_sectors = ";".join(str(o.star.sector) for o in usable)
    result.confirmed_sectors = ";".join(str(o.star.sector) for o, _ in confirming)
    if confirming and detection.primary is not None:
        joined = [cleaned(raw, detection, run.settings)]
        joined += [cleaned(o, d, run.settings) for o, d in confirming]
        _join_sectors(run, result, detection.primary.frequency, joined)
    hits = variability_catalogs(result.ra, result.dec)
    nearby = simbad_objects(result.ra, result.dec)
    result.simbad_name, result.simbad_type = nearby[0] if nearby else ("", "")
    hits += [f"SIMBAD {kind}" for _, kind in nearby if is_variable_type(kind)]
    result.catalog_hits = ";".join(hits)
    around = variability_catalogs(result.ra, result.dec, NEARBY_CATALOG_ARCSEC)
    result.nearby_catalog_hits = ";".join(name for name in around if name not in hits)
    if not hits:
        result.other_verdicts = ";".join(
            f"{o.star.sector}:{_verdict_in(result, o, d, resources)}" for o, d in confirming
        )
    vsini = measured_vsini(result.ra, result.dec)
    spin = rotation(raw.star, result.joint_period or result.period, vsini)
    if spin is not None:
        result.equatorial_velocity = spin.equatorial_velocity
        result.breakup_velocity = spin.breakup_velocity
        result.vsini = spin.vsini
    if result.gaia_id:
        quality = gaia_quality(int(result.gaia_id), result.ra, result.dec)
        if quality is not None:
            result.gaia_ruwe = quality.ruwe
            result.gaia_scatter = quality.scatter
            result.gaia_scatter_reliable = quality.scatter_reliable


def _join_sectors(run: Run, result: Result, frequency: float, lcs: list[LightCurve]) -> None:
    """Refines the reported period over all sectors, binaries keep their doubling."""
    factor = round(result.period * frequency)
    if factor < 1 or abs(result.period * frequency - factor) > HARMONIC_SLACK:
        return
    refined = refine_frequency(lcs, frequency)
    result.joint_period = factor / refined.frequency
    result.joint_period_error = factor * refined.error / refined.frequency**2
    result.joint_span = refined.span
    result.joint_aliases = refined.ambiguous
    sectors_figure(
        lcs, result.joint_period, result.epoch, run.plots_dir / f"TIC{result.tic}_sectors.png"
    )


def _verdict_in(result: Result, lc: LightCurve, detection: Detection, resources: Resources) -> str:
    try:
        return locate(result, lc, detection, resources, []).verdict
    except OSError:
        log.exception("no cutout for TIC %d in sector %d", result.tic, lc.star.sector)
        return "unavailable"
