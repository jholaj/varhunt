"""Synthetic sinusoids injected into real light curves to measure what a run can detect."""

from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path

import numpy as np

from varhunt.analysis.detect import detect
from varhunt.config import Settings
from varhunt.sources.lightcurve import load_qlp

AMPLITUDE_EDGES_PPT = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
TMAG_EDGES = (6.0, 10.0, 11.0, 12.0, 13.0, 16.0)
MIN_INJECTED_PERIOD = 0.01


@dataclass(frozen=True)
class Injection:
    path: Path
    period: float
    amplitude: float
    phase: float


@dataclass(frozen=True)
class Trial:
    tmag: float
    period: float
    amplitude: float
    recovered: bool


def plan(paths: list[Path], trials: int, settings: Settings, seed: int) -> list[Injection]:
    """Random quiet stars, log-uniform periods over the searched range and amplitudes."""
    rng = np.random.default_rng(seed)
    shortest = max(settings.min_period, MIN_INJECTED_PERIOD)
    periods = np.exp(rng.uniform(np.log(shortest), np.log(settings.max_period), trials))
    amplitudes = np.exp(
        rng.uniform(np.log(AMPLITUDE_EDGES_PPT[0]), np.log(AMPLITUDE_EDGES_PPT[-1]), trials)
    )
    return [
        Injection(
            path=paths[int(rng.integers(len(paths)))],
            period=float(period),
            amplitude=float(amplitude) / 1e3,
            phase=float(rng.uniform(0, 2 * np.pi)),
        )
        for period, amplitude in zip(periods, amplitudes, strict=True)
    ]


def run_trial(injection: Injection, settings: Settings) -> Trial:
    lc = load_qlp(injection.path)
    frequency = 1.0 / injection.period
    signal = injection.amplitude * np.sin(2 * np.pi * frequency * lc.time + injection.phase)
    detection = detect(replace(lc, flux=lc.flux + signal), settings, eclipses=False)
    found = any(abs(f.frequency - frequency) <= 1.0 / lc.baseline for f in detection.frequencies)
    return Trial(
        tmag=lc.star.tmag,
        period=injection.period,
        amplitude=injection.amplitude,
        recovered=found and detection.is_consistent(settings),
    )


def completeness_table(trials: list[Trial]) -> str:
    header = ["Semi-amplitude (ppt)"] + [
        f"Tmag {low:g} to {high:g}" for low, high in pairwise(TMAG_EDGES)
    ]
    lines = ["| " + " | ".join(header) + " |", "| --- |" + " ---: |" * (len(header) - 1)]
    for low, high in pairwise(AMPLITUDE_EDGES_PPT):
        cells = [f"{low:g} to {high:g}"]
        for tmag_low, tmag_high in pairwise(TMAG_EDGES):
            group = [
                t
                for t in trials
                if low <= t.amplitude * 1e3 < high and tmag_low <= t.tmag < tmag_high
            ]
            share = sum(t.recovered for t in group) / len(group) if group else None
            cells.append(f"{share:.0%} ({len(group)})" if share is not None else "")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
