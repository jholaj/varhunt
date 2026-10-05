"""How strongly a neighbour would have to vary to fake the signal seen on the target."""

from dataclasses import dataclass

import numpy as np

from varhunt.sources.catalogs import Neighbour, counterpart, separations_arcsec

PIXEL_ARCSEC = 21.0
# A neighbour's share of light in the target aperture falls off like a Gaussian of this width.
APERTURE_SIGMA_PX = 1.2
SEARCH_RADIUS_PX = 3.0
# Typical G minus T colour, used when the target has no Gaia counterpart.
G_MINUS_T = 0.5


@dataclass(frozen=True)
class BlendRisk:
    counterpart: Neighbour | None
    neighbour: Neighbour | None
    required_amplitude: float | None


def blend_risk(
    ra: float, dec: float, tmag: float, amplitude: float, candidates: list[Neighbour]
) -> BlendRisk:
    """The neighbour that needs the smallest amplitude of its own to explain the observed one."""
    target = counterpart(ra, dec, candidates)
    others = [n for n in candidates if target is None or n.source_id != target.source_id]
    if not others:
        return BlendRisk(target, None, None)
    pixels = separations_arcsec(ra, dec, others) / PIXEL_ARCSEC
    near = pixels <= SEARCH_RADIUS_PX
    if not near.any():
        return BlendRisk(target, None, None)

    # Share of a neighbour's light in the aperture, a Gaussian in its distance in pixels.
    weights = np.exp(-0.5 * (pixels / APERTURE_SIGMA_PX) ** 2)
    fluxes = np.array([n.flux for n in others])
    own = target.flux if target is not None else 10 ** (-0.4 * (tmag + G_MINUS_T))
    total = own + np.sum(weights[near] * fluxes[near])
    required = np.full(len(others), np.inf)
    # A neighbour diluted by w F_n among the total flux needs A_n = A F_total / (w F_n) to fake A.
    required[near] = amplitude * total / (weights[near] * fluxes[near])
    index = int(np.argmin(required))
    return BlendRisk(target, others[index], float(required[index]))
