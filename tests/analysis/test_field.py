from varhunt.analysis.field import Signal, attribute
from varhunt.config import Settings

SETTINGS = Settings()
ARCSEC = 1 / 3600


def signal(tic: int, dra: float, tmag: float, amplitude: float, *frequencies: float) -> Signal:
    return Signal(
        tic=tic,
        ra=100.0 + dra * ARCSEC,
        dec=20.0,
        tmag=tmag,
        amplitude=amplitude,
        frequencies=frequencies,
        baseline=24.0,
    )


def test_shared_local_signal_goes_to_the_star_with_most_variable_flux():
    signals = [
        signal(1, 0, 11.6, 0.011, 1.756),
        signal(2, 50, 12.8, 0.014, 1.757),
        signal(3, 100, 11.8, 0.002, 1.755),
    ]
    attribution = attribute(signals, SETTINGS)
    assert attribution.sources == {2: 1, 3: 1}
    assert attribution.echoes(1) == 2
    assert attribution.echoes(2) == 0
    assert not attribution.systematic


def test_far_apart_stars_are_not_linked():
    signals = [signal(1, 0, 11.0, 0.01, 2.0), signal(2, 1000, 11.0, 0.01, 2.0)]
    assert attribute(signals, SETTINGS).sources == {}


def test_harmonic_counts_as_the_same_signal():
    signals = [signal(1, 0, 10.0, 0.02, 4.0), signal(2, 40, 12.0, 0.002, 8.0)]
    assert attribute(signals, SETTINGS).sources == {2: 1}


def test_unrelated_frequencies_stay_separate():
    signals = [signal(1, 0, 10.0, 0.02, 4.0), signal(2, 40, 12.0, 0.002, 5.3)]
    assert attribute(signals, SETTINGS).sources == {}


def test_frequency_shared_across_the_field_is_systematic():
    signals = [signal(tic, tic * 200, 12.0, 0.001, 3.0) for tic in range(1, 7)]
    attribution = attribute(signals, SETTINGS)
    assert attribution.systematic == frozenset(range(1, 7))
    assert attribution.sources == {}


def test_empty_field():
    attribution = attribute([], SETTINGS)
    assert attribution.sources == {}
    assert not attribution.systematic
