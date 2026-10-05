"""Searching one field, every QLP light curve analysed and the variables sorted by status."""

import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from varhunt.analysis.blend import blend_risk
from varhunt.analysis.classify import Ephemeris, ephemeris, type_hint
from varhunt.analysis.detect import Detection, cleaned, detect, signal_model
from varhunt.analysis.field import Attribution, Signal, attribute
from varhunt.config import Resources, Run, Settings
from varhunt.output.plots import candidate_figure
from varhunt.output.report import (
    CANDIDATE,
    INCONSISTENT,
    KNOWN,
    KNOWN_NEARBY,
    QUIET,
    SHARED,
    SLOW,
    SYSTEMATIC,
    Result,
)
from varhunt.sources.catalogs import KnownCatalog, neighbours
from varhunt.sources.lightcurve import Star, load_qlp
from varhunt.sources.mast import download, find_light_curves
from varhunt.workflows.shared import PPT, amplitude, light_curve, plot_path, write

log = logging.getLogger(__name__)

BLEND_RADIUS_ARCSEC = 75.0


@dataclass(frozen=True)
class Analysis:
    star: Star
    detection: Detection
    file: str
    n_points: int
    baseline: float
    type_hint: str
    consistent: bool
    ephemeris: Ephemeris | None
    light_range: tuple[float, float] | None


def search(run: Run, resources: Resources) -> list[Result]:
    run.save()
    products = find_light_curves(run.field, run.sector)
    log.info("%d QLP light curves in sector %d", len(products), run.sector)
    paths = download(products, resources.cache_dir, resources.workers)
    log.info("%d on disk, analysing", len(paths))
    analyses = _analyse_all(sorted(paths.values()), run.settings, resources.workers)

    variable = [a for a in analyses if a.detection.is_variable]
    attribution = attribute([_signal(a) for a in variable], run.settings)
    known = KnownCatalog.query(run.field, margin_arcsec=2 * run.settings.match_radius_arcsec)
    results = [_result(a, attribution, known, run.settings) for a in analyses]
    candidates = [r for r in results if r.status == CANDIDATE]
    log.info("%d variable stars, %d candidates", len(variable), len(candidates))

    detections = {a.star.tic: a.detection for a in analyses}
    for result in candidates:
        _assess_blend(result)
        detection = detections[result.tic]
        candidate_figure(
            cleaned(light_curve(resources, result), detection, run.settings),
            detection,
            result,
            run.settings,
            plot_path(run, result),
        )
    write(run, results)
    return results


def _analyse_all(paths: list[Path], settings: Settings, workers: int) -> list[Analysis]:
    with ProcessPoolExecutor(workers) as pool:
        analyses = list(pool.map(partial(_analyse, settings=settings), paths, chunksize=4))
    failed = sum(a is None for a in analyses)
    if failed:
        log.warning("%d light curves could not be read", failed)
    return [a for a in analyses if a is not None]


def _analyse(path: Path, settings: Settings) -> Analysis | None:
    try:
        lc = load_qlp(path)
    except (OSError, KeyError, ValueError):
        return None
    detection = detect(lc, settings)
    hint = type_hint(lc, detection)
    model = signal_model(detection)(lc.time) if detection.is_variable else None
    return Analysis(
        star=lc.star,
        detection=detection,
        file=path.name,
        n_points=len(lc.time),
        baseline=lc.baseline,
        type_hint=hint,
        consistent=detection.is_consistent(settings),
        ephemeris=ephemeris(lc, detection),
        light_range=(float(model.min()), float(model.max())) if model is not None else None,
    )


def _signal(analysis: Analysis) -> Signal:
    detection = analysis.detection
    if detection.primary is not None:
        strength = detection.primary.amplitude
        frequencies = tuple(f.frequency for f in detection.frequencies)
    else:
        strength = detection.eclipse.depth / 2
        frequencies = (1.0 / detection.eclipse.period,)
    return Signal(
        tic=analysis.star.tic,
        ra=analysis.star.ra,
        dec=analysis.star.dec,
        tmag=analysis.star.tmag,
        amplitude=strength,
        frequencies=frequencies,
        baseline=analysis.baseline,
    )


def _result(
    analysis: Analysis, attribution: Attribution, known: KnownCatalog, settings: Settings
) -> Result:
    star, detection = analysis.star, analysis.detection
    result = Result(
        tic=star.tic,
        ra=star.ra,
        dec=star.dec,
        tmag=star.tmag,
        teff=star.teff,
        file=analysis.file,
        n_points=analysis.n_points,
        rms_ppt=detection.rms * PPT,
        type_hint=analysis.type_hint,
        shared_with=attribution.sources.get(star.tic),
        echoes=attribution.echoes(star.tic),
    )
    if (orbit := analysis.ephemeris) is not None:
        result.period = orbit.period
        result.period_error = orbit.period_error
        result.epoch = orbit.epoch
        result.epoch_kind = orbit.kind
    if analysis.light_range is not None:
        result.light_min_ppt = analysis.light_range[0] * PPT
        result.light_max_ppt = analysis.light_range[1] * PPT
    if (primary := detection.primary) is not None:
        result.amplitude_ppt = primary.amplitude * PPT
        result.snr = primary.snr
        result.periods = ";".join(f"{f.period:.7f}" for f in detection.frequencies)
        result.orbit_ratio = detection.orbit_ratio
        result.orbit_coherence = detection.orbit_coherence
    if (eclipse := detection.eclipse) is not None:
        result.eclipse_period = eclipse.period
        result.eclipse_duration = eclipse.duration
        result.eclipse_depth_ppt = eclipse.depth * PPT
        result.eclipse_secondary_ppt = eclipse.secondary_depth * PPT
        result.eclipse_snr = eclipse.snr
        result.eclipse_epoch = eclipse.epoch
    nearest = known.nearest(star.ra, star.dec)
    if nearest is not None and nearest[1] <= 2 * settings.match_radius_arcsec:
        variable, separation = nearest
        result.known_name = variable.name
        result.known_type = variable.kind
        result.known_period = variable.period
        result.known_separation = separation
    result.status = _status(analysis, attribution, result.known_separation, settings)
    return result


def _status(
    analysis: Analysis,
    attribution: Attribution,
    known_separation: float | None,
    settings: Settings,
) -> str:
    if analysis.detection.beyond_range and analysis.detection.eclipse is None:
        return SLOW
    if not analysis.detection.is_variable:
        return QUIET
    if analysis.star.tic in attribution.systematic:
        return SYSTEMATIC
    if not analysis.consistent:
        return INCONSISTENT
    if analysis.star.tic in attribution.sources:
        return SHARED
    if known_separation is not None:
        return KNOWN if known_separation <= settings.match_radius_arcsec else KNOWN_NEARBY
    return CANDIDATE


def _assess_blend(result: Result) -> None:
    nearby = neighbours(result.ra, result.dec, BLEND_RADIUS_ARCSEC)
    risk = blend_risk(result.ra, result.dec, result.tmag, amplitude(result), nearby)
    if risk.counterpart is not None:
        result.gaia_id = str(risk.counterpart.source_id)
    if risk.neighbour is not None:
        result.blend_neighbour = risk.neighbour.name
        result.blend_required_ppt = risk.required_amplitude * PPT
