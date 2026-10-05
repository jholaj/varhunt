"""What every workflow needs, a candidate's light curve, its pixel check and the run files."""

from pathlib import Path

from varhunt.analysis.detect import Detection, signal_model
from varhunt.checks.pixels import Localisation, fetch_cutout, load_cutout, localise
from varhunt.config import Resources, Run
from varhunt.output.report import CANDIDATE, Result, write_results, write_summary
from varhunt.sources.catalogs import Neighbour
from varhunt.sources.lightcurve import LightCurve, load_qlp
from varhunt.sources.mast import local_path

PPT = 1e3


def light_curve(resources: Resources, result: Result) -> LightCurve:
    return load_qlp(local_path(resources.cache_dir, result.file))


def locate(
    result: Result,
    lc: LightCurve,
    detection: Detection,
    resources: Resources,
    nearby: list[Neighbour],
) -> Localisation:
    """Pixel check of the signal in the sector the light curve comes from."""
    path = fetch_cutout(result.ra, result.dec, lc.star.sector, resources.cache_dir)
    template = signal_model(detection)
    return localise(
        load_cutout(path), result.ra, result.dec, template, detection.signal_frequency, nearby
    )


def amplitude(result: Result) -> float:
    if result.eclipse_depth_ppt is not None:
        return result.eclipse_depth_ppt / 2 / PPT
    return result.amplitude_ppt / PPT


def plot_path(run: Run, result: Result) -> Path:
    return run.plots_dir / f"TIC{result.tic}.png"


def write(run: Run, results: list[Result]) -> None:
    write_results(run.stars_csv, results)
    write_results(run.candidates_csv, [r for r in results if r.status == CANDIDATE])
    write_summary(run, results)
