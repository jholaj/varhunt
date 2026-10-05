"""Pixel check of a run's candidates in the TESScut cutouts of their sector."""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from varhunt.analysis.blend import blend_risk
from varhunt.analysis.detect import cleaned, detect
from varhunt.checks.pixels import fetch_cutout
from varhunt.config import Resources, Run
from varhunt.output.plots import candidate_figure
from varhunt.output.report import CANDIDATE, Result, read_results
from varhunt.sources.catalogs import counterpart, neighbours
from varhunt.workflows.shared import PPT, amplitude, light_curve, locate, plot_path, write

log = logging.getLogger(__name__)

CUTOUT_RADIUS_ARCSEC = 170.0
CUTOUT_THREADS = 4


def vet(run: Run, resources: Resources, tics: set[int]) -> list[Result]:
    results = read_results(run.stars_csv)
    chosen = [r for r in results if r.status == CANDIDATE and (not tics or r.tic in tics)]
    with ThreadPoolExecutor(CUTOUT_THREADS) as pool:
        list(pool.map(partial(_prefetch, run.sector, resources), chosen))
    for result in chosen:
        log.info("pixel check of TIC %d", result.tic)
        raw = light_curve(resources, result)
        detection = detect(raw, run.settings)
        if detection.signal_frequency is None:
            continue
        nearby = neighbours(result.ra, result.dec, CUTOUT_RADIUS_ARCSEC)
        try:
            localisation = locate(result, raw, detection, resources, nearby)
        except OSError:
            log.exception("no cutout for TIC %d", result.tic)
            continue
        result.vet_verdict = localisation.verdict
        result.vet_offset_px = round(localisation.offset_px, 3)
        result.vet_centroid_error_px = round(localisation.centroid_error_px, 3)
        source = localisation.likely_source
        result.vet_source = source.name if source is not None else ""
        result.vet_companions = ";".join(n.name for n in localisation.companions)
        suspects = [counterpart(result.ra, result.dec, nearby), *localisation.companions]
        risk = blend_risk(
            result.ra, result.dec, result.tmag, amplitude(result), [s for s in suspects if s]
        )
        if risk.neighbour is not None:
            result.vet_companion_needs_ppt = risk.required_amplitude * PPT
        candidate_figure(
            cleaned(raw, detection, run.settings),
            detection,
            result,
            run.settings,
            plot_path(run, result),
            localisation,
        )
    write(run, results)
    return chosen


def _prefetch(sector: int, resources: Resources, result: Result) -> None:
    try:
        fetch_cutout(result.ra, result.dec, sector, resources.cache_dir)
    except Exception:
        # The pixel check fetches it again and reports what went wrong.
        log.warning("prefetching the cutout of TIC %d failed", result.tic)
