"""A VSX submission for one candidate, with elements from every TESS sector and VSX-style plots."""

import logging
from pathlib import Path

import numpy as np
from astropy.time import Time

from varhunt.analysis.detect import amplitude_spectrum, cleaned, detect, signal_model
from varhunt.analysis.ephemeris import (
    Minimum,
    heliocentric_date,
    linear_ephemeris,
    minima,
    phase_of,
    profile,
)
from varhunt.config import Resources, Run
from varhunt.output.plots import submission_figures
from varhunt.output.report import BTJD_OFFSET, Result, read_results
from varhunt.output.vsx import Attachment, Submission, write_draft
from varhunt.sources.catalogs import Identity, identity
from varhunt.sources.lightcurve import LightCurve, load_qlp, load_spoc
from varhunt.sources.mast import download, light_curves_of, spoc_light_curves
from varhunt.workflows.shared import PPT

log = logging.getLogger(__name__)

# A TESS-SPOC sector confirms the signal when its amplitude stands this far above the noise.
MIN_SPOC_SNR = 5.0
NOISE_WINDOW = 1.0
NOISE_POINTS = 2000
ECLIPSE_BINS = 400
WAVE_BINS = 60
MODEL_STEPS = 20000
# Amplitudes below this read better in millimagnitudes.
MILLIMAG_BELOW = 0.05
ON_TARGET = "on_target"


def prepare(run: Run, resources: Resources, tic: int, out: Path, discoverer: str) -> Path:
    """Writes the draft and its plots into a folder named after the star, returns the draft."""
    result = next((r for r in read_results(run.stars_csv) if r.tic == tic), None)
    if result is None or not result.gaia_id:
        raise LookupError(f"TIC {tic} is not a candidate with a Gaia counterpart in {run.root}")
    if not result.sure:
        log.warning("TIC %d is not sure, check it twice", tic)
    who = identity(tic, int(result.gaia_id), result.ra, result.dec)
    qlp = _qlp_sectors(run, resources, result)
    spoc_paths = download(spoc_light_curves(tic), resources.cache_dir, resources.workers)
    spoc = [lc for lc in (load_spoc(path) for path in spoc_paths.values()) if len(lc.time)]
    folder = out / who.name.replace(" ", "")
    if result.eclipse_depth_ppt is not None:
        submission = _eclipsing(result, who, qlp, spoc, folder, discoverer)
    else:
        submission = _periodic(run, result, who, qlp, spoc, folder, discoverer)
    path = folder / "draft.md"
    write_draft(path, submission)
    return path


def _qlp_sectors(run: Run, resources: Resources, result: Result) -> list[LightCurve]:
    paths = download(light_curves_of(result.tic), resources.cache_dir, resources.workers)
    raws = [load_qlp(path) for path in paths.values()]
    return [cleaned(raw, detect(raw, run.settings), run.settings) for raw in raws if len(raw.time)]


def _eclipsing(
    result: Result,
    who: Identity,
    qlp: list[LightCurve],
    spoc: list[LightCurve],
    folder: Path,
    discoverer: str,
) -> Submission:
    depth = result.eclipse_depth_ppt / PPT
    secondary = (result.eclipse_secondary_ppt or 0.0) / PPT
    covered = {lc.star.sector for lc in spoc}
    timed = spoc + [lc for lc in qlp if lc.star.sector not in covered]
    threshold = 0.5 * (secondary if secondary > 0 else depth)
    dips = [m for lc in timed for m in minima(lc, threshold, result.eclipse_duration)]
    primaries, secondaries = _split(dips, depth, secondary)
    elements = linear_ephemeris([m.time for m in primaries], result.period)
    shapes = spoc or qlp
    first = profile(shapes, elements, 0.0)
    second_phase = phase_of([m.time for m in secondaries], elements) if secondaries else None
    second = profile(shapes, elements, second_phase) if second_phase is not None else None
    # An eclipse that removes a share d of the light is -2.5 log10(1 - d) magnitudes deep.
    amplitude = -2.5 * np.log10(1.0 - first.depth)
    epoch_bjd = elements.epoch + BTJD_OFFSET
    years = sorted({Time(m.time + BTJD_OFFSET, format="jd").datetime.year for m in primaries})
    sectors = sorted({m.sector for m in primaries})

    remarks = []
    if elements.period_error is not None:
        remarks.append(f"P uncertainty {elements.period_error:.7f} d.")
    if second is not None:
        remarks.append(
            f"Min II = {-2.5 * np.log10(1.0 - second.depth):.3f} TESS (amplitude) "
            f"at phase {second_phase:.4f}."
        )
    remarks.append(
        f"Elements from {len(primaries)} TESS minima {years[0]}-{years[-1]} "
        f"(sectors {', '.join(map(str, sectors))}), rms O-C {elements.rms_minutes:.1f} min."
    )
    remarks.append(f"Epoch = BJD(TDB) {epoch_bjd:.4f}. Max V from {who.v_source}.")
    source = "TESS-SPOC PDCSAP" if spoc else "MIT QLP"
    phase_plot, timeline = submission_figures(
        shapes,
        result.tmag,
        elements.period,
        epoch_bjd,
        who.name,
        folder / who.name.replace(" ", ""),
        ECLIPSE_BINS,
        millimag=amplitude < MILLIMAG_BELOW,
    )
    evidence = [
        f"{len(primaries)} primary and {len(secondaries)} secondary minima timed over "
        f"{elements.cycles} cycles, rms O-C {elements.rms_minutes:.1f} min.",
        _spoc_line(spoc),
        _pixel_line(result),
        *_common_evidence(result),
    ]
    return Submission(
        identity=who,
        variable_type="EA",
        amplitude=float(amplitude),
        period=elements.period,
        period_error=elements.period_error,
        epoch_hjd=heliocentric_date(epoch_bjd, result.ra, result.dec),
        epoch_bjd=epoch_bjd,
        epoch_kind="primary minimum",
        duration_percent=100 * first.duration,
        discoverer=discoverer,
        remarks=" ".join(remarks),
        attachments=(
            Attachment(
                phase_plot,
                f"Phase plot, {source}, sectors {_sector_list(shapes)}",
                f"TESS full-frame-image photometry ({source}) folded with "
                f"P = {elements.period:.7f} d. {_pixel_line(result)}",
            ),
            Attachment(
                timeline,
                f"Light curve, {source}, sector {max(lc.star.sector for lc in shapes)}",
                "TESS magnitude against BJD, brightest at the top.",
            ),
        ),
        evidence=tuple(evidence),
    )


def _periodic(
    run: Run,
    result: Result,
    who: Identity,
    qlp: list[LightCurve],
    spoc: list[LightCurve],
    folder: Path,
    discoverer: str,
) -> Submission:
    own = next(lc for lc in qlp if lc.star.sector == run.sector)
    detection = detect(own, run.settings)
    joint = result.joint_period is not None and not result.joint_aliases
    period = result.joint_period if joint else result.period
    period_error = result.joint_period_error if joint else result.period_error
    model = signal_model(detection)
    grid = result.epoch + np.linspace(0.0, period, MODEL_STEPS, endpoint=False)
    # VSX wants the time of maximum light for pulsating and rotating stars.
    epoch_bjd = float(grid[np.argmax(model(grid))]) + BTJD_OFFSET
    # Full range of the fitted model, from its brightest to its faintest point, in magnitudes.
    amplitude = 2.5 * np.log10((1 + result.light_max_ppt / PPT) / (1 + result.light_min_ppt / PPT))
    strongest = detection.primary.frequency
    confirming = [lc.star.sector for lc in spoc if _snr_at(lc, strongest) >= MIN_SPOC_SNR]
    frequencies = ", ".join(f"{1 / float(p):.4f}" for p in result.periods.split(";"))

    remarks = []
    if period_error is not None:
        remarks.append(f"P uncertainty {period_error:.7f} d.")
    remarks.append(f"Frequencies {frequencies} c/d.")
    if confirming:
        remarks.append(f"Present in TESS-SPOC PDCSAP sectors {', '.join(map(str, confirming))}.")
    remarks.append(f"Epoch of maximum = BJD(TDB) {epoch_bjd:.4f}. Max V from {who.v_source}.")
    shapes = spoc or qlp
    source = "TESS-SPOC PDCSAP" if spoc else "MIT QLP"
    phase_plot, timeline = submission_figures(
        shapes,
        result.tmag,
        period,
        epoch_bjd,
        who.name,
        folder / who.name.replace(" ", ""),
        WAVE_BINS,
        millimag=amplitude < MILLIMAG_BELOW,
    )
    spoc_line = (
        f"TESS-SPOC PDCSAP shows the {strongest:.4f} c/d signal in sectors "
        f"{', '.join(map(str, confirming))} of {_sector_list(spoc)}."
        if spoc
        else "No TESS-SPOC light curves, the plots use QLP."
    )
    evidence = [
        f"Type hint {result.type_hint} is a rule of thumb, decide the type yourself.",
        spoc_line,
        _pixel_line(result),
        "The period comes from all sectors."
        if joint
        else "The period comes from one sector, the cycle count across gaps is not certain.",
        *_common_evidence(result),
    ]
    return Submission(
        identity=who,
        variable_type=result.type_hint or "VAR",
        amplitude=float(amplitude),
        period=float(period),
        period_error=period_error,
        epoch_hjd=heliocentric_date(epoch_bjd, result.ra, result.dec),
        epoch_bjd=epoch_bjd,
        epoch_kind="maximum",
        duration_percent=None,
        discoverer=discoverer,
        remarks=" ".join(remarks),
        attachments=(
            Attachment(
                phase_plot,
                f"Phase plot, {source}, sectors {_sector_list(shapes)}",
                f"TESS full-frame-image photometry ({source}) folded with P = {period:.7f} d. "
                f"{_pixel_line(result)}",
            ),
            Attachment(
                timeline,
                f"Light curve, {source}, sector {max(lc.star.sector for lc in shapes)}",
                "TESS magnitude against BJD, brightest at the top.",
            ),
        ),
        evidence=tuple(evidence),
    )


def _split(dips: list[Minimum], depth: float, secondary: float) -> tuple[list, list]:
    """Primary and secondary minima, told apart within each sector since dilution differs."""
    if secondary <= 0:
        return dips, []
    primaries, secondaries = [], []
    for sector in {m.sector for m in dips}:
        own = [m for m in dips if m.sector == sector]
        cut = max(m.depth for m in own) * (1 + secondary / depth) / 2
        primaries += [m for m in own if m.depth > cut]
        secondaries += [m for m in own if m.depth <= cut]
    return primaries, secondaries


def _snr_at(lc: LightCurve, frequency: float) -> float:
    grid = np.linspace(max(frequency - NOISE_WINDOW, 0.05), frequency + NOISE_WINDOW, NOISE_POINTS)
    noise = np.median(amplitude_spectrum(lc.time, lc.flux - 1.0, grid))
    signal = amplitude_spectrum(lc.time, lc.flux - 1.0, np.array([frequency]), "cython")[0]
    return float(signal / noise) if noise > 0 else 0.0


def _pixel_line(result: Result) -> str:
    others = [item for item in result.other_verdicts.split(";") if item.endswith(ON_TARGET)]
    count = len(others) + (result.vet_verdict == ON_TARGET)
    return (
        "A pixel-level analysis of TESScut images places the signal on the target "
        f"in {count} sectors."
    )


def _spoc_line(spoc: list[LightCurve]) -> str:
    if not spoc:
        return "No TESS-SPOC light curves, the minima come from QLP."
    return f"TESS-SPOC PDCSAP light curves of sectors {_sector_list(spoc)} were used."


def _common_evidence(result: Result) -> list[str]:
    lines = []
    if result.gaia_ruwe is not None and result.gaia_ruwe > 1.4:
        lines.append(f"Gaia RUWE {result.gaia_ruwe:.2f}, probably an unresolved binary.")
    if result.nearby_catalog_hits:
        lines.append(
            f"Variability catalogues list another star within 30 arcsec: "
            f"{result.nearby_catalog_hits.replace(';', ', ')}."
        )
    lines.append("Look the star up in ASAS-SN Sky Patrol, a phase plot of ground-based data helps.")
    return lines


def _sector_list(lcs: list[LightCurve]) -> str:
    return ", ".join(str(s) for s in sorted({lc.star.sector for lc in lcs}))
