import pytest

from varhunt.sources.catalogs import KnownCatalog, KnownVariable, Neighbour, counterpart, johnson_v


def test_nearest_known_variable():
    catalog = KnownCatalog(
        [
            KnownVariable("V1", "EW", 0.31, 100.0, 20.0),
            KnownVariable("V2", "DSCT", None, 100.01, 20.0),
        ]
    )
    variable, separation = catalog.nearest(100.0095, 20.0)
    assert variable.name == "V2"
    assert separation == pytest.approx(1.69, abs=0.05)


def test_empty_catalog_has_no_nearest():
    assert KnownCatalog([]).nearest(100.0, 20.0) is None


def test_counterpart_needs_to_be_close():
    near = Neighbour(1, 100.0005, 20.0, 11.0)
    far = Neighbour(2, 100.01, 20.0, 9.0)
    assert counterpart(100.0, 20.0, [far, near]) == near
    assert counterpart(100.0, 20.0, [far]) is None
    assert counterpart(100.0, 20.0, []) is None


def test_gaia_colours_give_johnson_v():
    assert johnson_v(11.169, 0.784) == pytest.approx(11.295, abs=0.03)
    assert johnson_v(9.491, 0.036) == pytest.approx(9.518, abs=0.01)
