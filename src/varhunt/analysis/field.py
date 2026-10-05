"""Signals shared by several stars of a field, split into instrument systematics and blends."""

from dataclasses import dataclass

import numpy as np
from astropy.coordinates import angular_separation

from varhunt.config import Settings

# Peaks of one signal seen in different stars agree to a fraction of the frequency resolution.
FREQUENCY_TOLERANCE = 0.25


@dataclass(frozen=True)
class Signal:
    tic: int
    ra: float
    dec: float
    tmag: float
    amplitude: float
    frequencies: tuple[float, ...]
    baseline: float

    @property
    def variable_flux(self) -> float:
        # Relative amplitude times the star's flux, a magnitude m is a flux of 10^(-0.4 m).
        return self.amplitude * 10 ** (-0.4 * self.tmag)


@dataclass(frozen=True)
class Attribution:
    systematic: frozenset[int]
    sources: dict[int, int]

    def echoes(self, tic: int) -> int:
        """How many other stars show the signal attributed to this one."""
        return sum(source == tic for source in self.sources.values())


def attribute(signals: list[Signal], settings: Settings) -> Attribution:
    """Flags field-wide frequencies as systematic and gives each shared local signal to the star
    with the largest variable flux."""
    if not signals:
        return Attribution(frozenset(), {})
    separation = _separations(signals)
    systematic = _systematic(signals, separation, settings)

    parent = list(range(len(signals)))
    for i, j in zip(*np.triu_indices(len(signals), k=1), strict=True):
        if i in systematic or j in systematic:
            continue
        if separation[i, j] <= settings.link_radius_arcsec and _shared(signals[i], signals[j]):
            parent[_root(parent, int(i))] = _root(parent, int(j))

    groups: dict[int, list[int]] = {}
    for index in range(len(signals)):
        groups.setdefault(_root(parent, index), []).append(index)
    sources: dict[int, int] = {}
    for members in groups.values():
        # Light leaks from the star that varies most in absolute flux, the others see a share.
        source = max(members, key=lambda k: signals[k].variable_flux)
        for member in members:
            if member != source:
                sources[signals[member].tic] = signals[source].tic
    return Attribution(frozenset(signals[i].tic for i in systematic), sources)


def _separations(signals: list[Signal]) -> np.ndarray:
    ra = np.radians([s.ra for s in signals])
    dec = np.radians([s.dec for s in signals])
    radians = angular_separation(ra[:, None], dec[:, None], ra[None, :], dec[None, :])
    return np.degrees(radians) * 3600.0


def _systematic(signals: list[Signal], separation: np.ndarray, settings: Settings) -> set[int]:
    primary = np.array([s.frequencies[0] for s in signals])
    tolerance = FREQUENCY_TOLERANCE / min(s.baseline for s in signals)
    flagged: set[int] = set()
    for index in range(len(signals)):
        members = np.flatnonzero(np.abs(primary - primary[index]) <= tolerance)
        if len(members) < settings.systematic_min_stars:
            continue
        if separation[np.ix_(members, members)].max() > 2 * settings.link_radius_arcsec:
            flagged.update(int(m) for m in members)
    return flagged


def _shared(a: Signal, b: Signal) -> bool:
    tolerance = FREQUENCY_TOLERANCE / min(a.baseline, b.baseline)
    for one, others in ((a.frequencies[0], b.frequencies), (b.frequencies[0], a.frequencies)):
        for other in others:
            if min(abs(one - other), abs(one - 2 * other), abs(2 * one - other)) <= tolerance:
                return True
    return False


def _root(parent: list[int], index: int) -> int:
    while parent[index] != index:
        parent[index] = parent[parent[index]]
        index = parent[index]
    return index
