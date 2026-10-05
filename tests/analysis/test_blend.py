import numpy as np
import pytest

from varhunt.analysis.blend import APERTURE_SIGMA_PX, PIXEL_ARCSEC, blend_risk
from varhunt.sources.catalogs import Neighbour

RA, DEC = 100.0, 20.0


def neighbour(source_id: int, offset_arcsec: float, gmag: float) -> Neighbour:
    return Neighbour(source_id, RA + offset_arcsec / 3600 / np.cos(np.radians(DEC)), DEC, gmag)


def test_no_neighbours_means_no_risk():
    risk = blend_risk(RA, DEC, 11.0, 0.01, [neighbour(1, 0.5, 11.5)])
    assert risk.counterpart.source_id == 1
    assert risk.neighbour is None


def test_required_amplitude_follows_the_light_share():
    target, other = neighbour(1, 0.0, 11.0), neighbour(2, 21.0, 13.5)
    risk = blend_risk(RA, DEC, 10.5, 0.01, [target, other])
    weight = np.exp(-0.5 * (1.0 / APERTURE_SIGMA_PX) ** 2)
    expected = 0.01 * (target.flux + weight * other.flux) / (weight * other.flux)
    assert risk.neighbour.source_id == 2
    assert risk.required_amplitude == pytest.approx(expected, rel=1e-3)


def test_the_easiest_explanation_is_reported():
    candidates = [neighbour(1, 0.0, 11.0), neighbour(2, 40.0, 12.0), neighbour(3, 15.0, 12.0)]
    assert blend_risk(RA, DEC, 10.5, 0.01, candidates).neighbour.source_id == 3


def test_neighbours_beyond_the_search_radius_are_ignored():
    risk = blend_risk(
        RA, DEC, 10.5, 0.01, [neighbour(1, 0.0, 11.0), neighbour(2, 5 * PIXEL_ARCSEC, 9.0)]
    )
    assert risk.neighbour is None
