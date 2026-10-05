"""Diagnostic figures for candidates, injection completeness and VSX submissions."""

from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from varhunt.analysis.detect import Detection, amplitude_spectrum, frequency_grid
from varhunt.checks.pixels import Localisation
from varhunt.config import Settings
from varhunt.evaluation.injection import Trial
from varhunt.output.report import BTJD_OFFSET, Result
from varhunt.sources.lightcurve import LightCurve

PPT = 1e3
TIME_BIN = 30 / 1440
PHASE_BINS = 60
DPI = 110
TIME_ORIGIN = 2460000.0


def candidate_figure(
    lc: LightCurve,
    detection: Detection,
    result: Result,
    settings: Settings,
    path: Path,
    localisation: Localisation | None = None,
) -> None:
    columns = 3 if localisation is not None else 2
    figure = Figure(figsize=(6.5 * columns, 8), layout="constrained")
    axes = figure.subplots(2, columns)
    _light_curve(axes[0, 0], lc)
    _spectrum(axes[0, 1], lc, detection, settings)
    _folded(axes[1, 0], lc, result.period, result.epoch)
    _folded(axes[1, 1], lc, 2 * result.period, result.epoch)
    if localisation is not None:
        _pixels(axes[0, 2], localisation)
        axes[1, 2].axis("off")
    figure.suptitle(
        f"TIC {result.tic}, Tmag {result.tmag:.2f}, P = {result.period:.6f} d, "
        f"S/N {result.strength:.1f}, {result.type_hint}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)


def sectors_figure(lcs: list[LightCurve], period: float, epoch: float, path: Path) -> None:
    """Every sector folded on one ephemeris, the strongest proof the signal is the same."""
    figure = Figure(figsize=(7, 2.2 * len(lcs)), layout="constrained")
    axes = np.atleast_1d(figure.subplots(len(lcs), 1, sharex=True))
    for ax, lc in zip(axes, sorted(lcs, key=lambda lc: lc.star.sector), strict=True):
        _folded(ax, lc, period, epoch)
        ax.set_title(f"sector {lc.star.sector}")
    figure.suptitle(f"TIC {lcs[0].star.tic} folded at {period:.8f} d")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)


def submission_figures(
    lcs: list[LightCurve],
    magnitude: float,
    period: float,
    epoch: float,
    title: str,
    prefix: Path,
    bins: int,
    millimag: bool,
) -> tuple[Path, Path]:
    """Light curve of the latest sector and every sector folded, in magnitudes, brightest at the
    top as VSX asks. Small amplitudes read better in millimagnitudes around the mean."""
    ordered = sorted(lcs, key=lambda lc: lc.star.sector)
    label = "TESS (mmag from the mean)" if millimag else "TESS magnitude"

    def scaled(lc: LightCurve) -> np.ndarray:
        offset = -2.5 * np.log10(lc.flux)
        return offset * PPT if millimag else magnitude + offset

    figure = Figure(figsize=(9, 4), layout="constrained")
    ax = figure.subplots()
    latest = ordered[-1]
    ax.plot(latest.time + BTJD_OFFSET - TIME_ORIGIN, scaled(latest), ".", ms=2, color="k")
    ax.invert_yaxis()
    ax.set_xlabel(f"BJD - {TIME_ORIGIN:.0f}")
    ax.set_ylabel(label)
    ax.set_title(f"{title}, TESS sector {latest.star.sector}")
    timeline = prefix.with_name(prefix.name + "_lightcurve.png")
    timeline.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(timeline, dpi=DPI)

    figure = Figure(figsize=(9, 5), layout="constrained")
    ax = figure.subplots()
    for colour, lc in enumerate(ordered):
        phase = ((lc.time + BTJD_OFFSET - epoch) / period) % 1.0
        values = scaled(lc)
        centres, medians = _phase_bins(phase, values, bins)
        for shift in (0.0, 1.0):
            ax.plot(phase + shift, values, ".", ms=1, alpha=0.2, color=f"C{colour}")
            ax.plot(
                centres + shift,
                medians,
                "o",
                ms=3,
                color=f"C{colour}",
                label=f"sector {lc.star.sector}" if shift == 0.0 else None,
            )
    ax.invert_yaxis()
    ax.set_xlabel(f"phase (P = {period:.7f} d, epoch BJD {epoch:.4f})")
    ax.set_ylabel(label)
    ax.set_title(f"{title}, all TESS sectors folded")
    ax.legend(markerscale=2, fontsize=8)
    folded = prefix.with_name(prefix.name + "_phase.png")
    figure.savefig(folded, dpi=DPI)
    return folded, timeline


def completeness_figure(trials: list[Trial], path: Path) -> None:
    figure = Figure(figsize=(8, 6), layout="constrained")
    ax = figure.subplots()
    for recovered, colour, label in (
        (True, "tab:green", "recovered"),
        (False, "tab:red", "missed"),
    ):
        chosen = [t for t in trials if t.recovered == recovered]
        ax.scatter(
            [t.period for t in chosen],
            [t.amplitude * PPT for t in chosen],
            s=10,
            c=colour,
            label=label,
        )
    ax.set(xscale="log", yscale="log", xlabel="period (d)", ylabel="semi-amplitude (ppt)")
    ax.legend()
    ax.set_title("Injected sinusoids")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=DPI)


def _light_curve(ax: Axes, lc: LightCurve) -> None:
    flux = (lc.flux - 1.0) * PPT
    ax.plot(lc.time, flux, ",", color="0.6")
    times, values = _binned(lc.time, flux, TIME_BIN)
    ax.plot(times, values, ".", color="tab:blue", markersize=2)
    ax.set(xlabel="time (BTJD)", ylabel="relative flux (ppt)", title="light curve")


def _spectrum(ax: Axes, lc: LightCurve, detection: Detection, settings: Settings) -> None:
    grid = frequency_grid(lc, settings)
    ax.plot(grid, amplitude_spectrum(lc.time, lc.flux - 1.0, grid) * PPT, lw=0.6, color="black")
    for frequency in detection.frequencies:
        ax.axvline(frequency.frequency, color="tab:red", alpha=0.4, lw=1)
    ax.set(xlabel="frequency (1/d)", ylabel="amplitude (ppt)", title="amplitude spectrum")


def _folded(ax: Axes, lc: LightCurve, period: float, epoch: float) -> None:
    phase = ((lc.time - epoch) / period) % 1.0
    flux = (lc.flux - 1.0) * PPT
    ax.plot(phase, flux, ",", color="0.6")
    edges = np.linspace(0, 1, PHASE_BINS + 1)
    index = np.clip(np.digitize(phase, edges) - 1, 0, PHASE_BINS - 1)
    medians = [
        np.median(flux[index == k]) if np.any(index == k) else np.nan for k in range(PHASE_BINS)
    ]
    ax.plot((edges[:-1] + edges[1:]) / 2, medians, "o", color="tab:blue", markersize=3)
    ax.set(xlabel="phase", ylabel="relative flux (ppt)", title=f"folded at {period:.6f} d")


def _pixels(ax: Axes, localisation: Localisation) -> None:
    ax.imshow(localisation.signal_map, origin="lower", cmap="viridis")
    sources = localisation.sources
    if sources:
        sizes = [max(4.0, 140.0 * 10 ** (-0.2 * (n.gmag - 10))) for n, _, _ in sources]
        ax.scatter(
            [x for _, x, _ in sources],
            [y for _, _, y in sources],
            s=sizes,
            facecolors="none",
            edgecolors="white",
            linewidths=0.8,
        )
    ax.plot(*localisation.target, "x", color="red", markersize=10, label="target")
    ax.plot(*localisation.centroid, "*", color="orange", markersize=12, label="signal centroid")
    size = localisation.signal_map.shape
    ax.set(xlim=(-0.5, size[1] - 0.5), ylim=(-0.5, size[0] - 0.5))
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"pixel check: {localisation.verdict}, {localisation.offset_px:.2f} px off")


def _binned(time: np.ndarray, values: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    index = np.floor((time - time[0]) / width).astype(int)
    counts = np.bincount(index)
    filled = counts > 0
    return (
        np.bincount(index, time)[filled] / counts[filled],
        np.bincount(index, values)[filled] / counts[filled],
    )


def _phase_bins(phase: np.ndarray, values: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.digitize(phase, edges) - 1, 0, bins - 1)
    counts = np.bincount(index, minlength=bins)
    sums = np.bincount(index, values, minlength=bins)
    filled = counts > 0
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres[filled], sums[filled] / counts[filled]
