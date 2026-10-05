"""Periodic signals in one light curve, sinusoids by prewhitening and eclipses by a box search."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from astropy.timeseries import BoxLeastSquares, LombScargle

from varhunt.config import Settings
from varhunt.sources.lightcurve import MAD_TO_SIGMA, LightCurve, detrended

MIN_POINTS = 500
# Signals slower than this are bent by the first running median, the trend is redone around them.
RETREND_FREQUENCY = 1.25
RETREND_PASSES = 3
# The robustness check repeats the search with periods and detrending this many times longer.
WIDER_DETRENDING = 3.0
# Periods this far below the detrending window are left untouched by it.
WIDER_MIN_SHARE = 0.1
# The retrend has converged once the strongest amplitude moves by less than this share.
RETREND_SETTLED = 0.02
# Noise is the median amplitude within this many c/d of a peak, the peak itself left out.
NOISE_WINDOW = 1.0
NOISE_EXCLUSION = 3.0
MIN_NOISE_BINS = 20
REFINE_FACTOR = 20
MIN_ORBIT_SHARE = 0.2
# Peaks this many resolutions from the primary stay in the data for the orbit check, side lobes too.
ORBIT_RESOLUTION = 4.0
ECLIPSE_BIN = 10 / 1440
ECLIPSE_MIN_PERIOD = 0.35
ECLIPSE_DURATIONS = (0.02, 0.04, 0.08, 0.16)
ECLIPSE_GRID = 20000
MIN_ECLIPSES = 3
ALIAS_DIVISORS = (1.5, 2, 3)
ALIAS_WINDOW = 2e-3
ALIAS_STEPS = 101
ALIAS_TOLERANCE = 0.9
# Every other dip this much shallower is a secondary eclipse, the orbit is twice as long.
ODD_EVEN_SIGMA = 5.0
ODD_EVEN_SHARE = 0.2
SECONDARY_SIGMA = 3.0
# Eclipse depths are read off the middle quarter of the eclipse.
CENTRAL_SHARE = 0.25
MIN_CENTRAL_POINTS = 10


@dataclass(frozen=True)
class Frequency:
    frequency: float
    amplitude: float
    snr: float
    epoch: float

    @property
    def period(self) -> float:
        return 1.0 / self.frequency


@dataclass(frozen=True)
class Eclipse:
    period: float
    duration: float
    depth: float
    snr: float
    epoch: float
    events: int
    odd_even_sigma: float
    secondary_depth: float = 0.0


@dataclass(frozen=True)
class Detection:
    frequencies: tuple[Frequency, ...] = ()
    eclipse: Eclipse | None = None
    rms: float = float("nan")
    period_error: float | None = None
    orbit_ratio: float | None = None
    orbit_coherence: float | None = None
    # The strongest signal lies outside the searched periods, what is left inside may be its echo.
    beyond_range: bool = False

    @property
    def primary(self) -> Frequency | None:
        return self.frequencies[0] if self.frequencies else None

    @property
    def signal_frequency(self) -> float | None:
        """Frequency of the reported signal, the eclipses when there are any."""
        if self.eclipse is not None:
            return 1.0 / self.eclipse.period
        return self.primary.frequency if self.primary is not None else None

    @property
    def is_variable(self) -> bool:
        return bool(self.frequencies) or self.eclipse is not None

    def is_consistent(self, settings: Settings) -> bool:
        """True when the strongest sinusoid looks alike in every orbit, or eclipses repeat."""
        if self.primary is None:
            return self.eclipse is not None
        if self.orbit_ratio is None or self.orbit_coherence is None:
            return True
        return (
            self.orbit_ratio >= settings.min_orbit_ratio
            and self.orbit_coherence >= settings.min_orbit_coherence
        )


def detect(raw: LightCurve, settings: Settings, eclipses: bool = True) -> Detection:
    if len(raw.time) < MIN_POINTS:
        return Detection()
    lc = detrended(raw, settings.detrend_window)
    frequencies, models, beyond = _prewhiten(lc, settings)
    for _ in range(RETREND_PASSES):
        if not _needs_retrend(frequencies):
            break
        previous = frequencies
        lc = detrended(raw, settings.detrend_window, waves(frequencies)(raw.time))
        frequencies, models, beyond = _prewhiten(lc, settings)
        if _settled(previous, frequencies):
            break
    rms = float(np.std(lc.flux))
    residual = lc.flux - 1.0 - sum(models, np.zeros_like(lc.flux))
    eclipse = _search_eclipse(lc, settings, frequencies, models) if eclipses else None
    if not frequencies:
        return Detection(eclipse=eclipse, rms=rms, beyond_range=beyond)

    primary = frequencies[0]
    distant = [
        model
        for frequency, model in zip(frequencies[1:], models[1:], strict=True)
        if abs(frequency.frequency - primary.frequency) > ORBIT_RESOLUTION / lc.baseline
    ]
    ratio, coherence = _orbit_consistency(
        lc, lc.flux - 1.0 - sum(distant, np.zeros_like(lc.flux)), primary.frequency
    )
    return Detection(
        frequencies=tuple(frequencies),
        eclipse=eclipse,
        rms=rms,
        period_error=_period_error(primary, residual, lc.baseline),
        orbit_ratio=ratio,
        orbit_coherence=coherence,
        beyond_range=beyond,
    )


def survives_wider_detrending(raw: LightCurve, detection: Detection, settings: Settings) -> bool:
    """Whether the strongest sinusoid stays when slower signals are let through, a slow
    variation half removed by the running median leaves peaks that then vanish."""
    primary = detection.primary
    if detection.eclipse is not None or primary is None:
        return True
    if primary.period < WIDER_MIN_SHARE * settings.detrend_window:
        return True
    wider = replace(
        settings,
        max_period=WIDER_DETRENDING * settings.max_period,
        max_frequencies=2 * settings.max_frequencies,
    )
    found = detect(raw, wider, eclipses=False).frequencies
    tolerance = 1.0 / raw.baseline
    return any(abs(f.frequency - primary.frequency) <= tolerance for f in found)


def cleaned(raw: LightCurve, detection: Detection, settings: Settings) -> LightCurve:
    """The detrended light curve the detection converged on."""
    if _needs_retrend(list(detection.frequencies)):
        return detrended(raw, settings.detrend_window, waves(detection.frequencies)(raw.time))
    return detrended(raw, settings.detrend_window)


def signal_model(detection: Detection) -> Callable[[np.ndarray], np.ndarray]:
    """Fitted relative flux variation over time, the eclipses if found, else the sinusoids."""
    if detection.eclipse is not None:
        eclipse = detection.eclipse
        return lambda time: (
            -eclipse.depth * _in_eclipse(time, eclipse)
            - eclipse.secondary_depth * _in_eclipse(time, eclipse, phase=0.5)
        )
    return waves(detection.frequencies)


def waves(
    frequencies: tuple[Frequency, ...] | list[Frequency],
) -> Callable[[np.ndarray], np.ndarray]:
    """Sum of the given sinusoids as relative flux over time."""

    def model(time: np.ndarray) -> np.ndarray:
        waves = [
            f.amplitude * np.cos(2 * np.pi * f.frequency * (time - f.epoch)) for f in frequencies
        ]
        return sum(waves, np.zeros_like(time))

    return model


def is_harmonic(frequency: float, fundamental: float, tolerance: float, lowest: int = 1) -> bool:
    order = round(frequency / fundamental)
    return order >= lowest and abs(frequency - order * fundamental) <= tolerance


def frequency_grid(lc: LightCurve, settings: Settings) -> np.ndarray:
    low = 1.0 / settings.max_period
    high = min(0.95 * lc.nyquist, 1.0 / settings.min_period)
    return np.arange(low, high, 1.0 / (lc.baseline * settings.oversample))


def amplitude_spectrum(
    time: np.ndarray, flux: np.ndarray, grid: np.ndarray, method: str = "fast"
) -> np.ndarray:
    """Semi-amplitude of the best-fitting sinusoid at every grid frequency."""
    periodogram = LombScargle(time, flux, fit_mean=False, center_data=True, normalization="psd")
    power = periodogram.power(grid, method=method, assume_regular_frequency=True)
    # With psd normalisation the power P of N points is a semi-amplitude A = sqrt(4 P / N).
    return np.sqrt(4.0 * np.clip(power, 0.0, None) / len(time))


def _needs_retrend(frequencies: list[Frequency]) -> bool:
    return any(f.frequency < RETREND_FREQUENCY for f in frequencies)


def _settled(previous: list[Frequency], current: list[Frequency]) -> bool:
    if not current or len(previous) != len(current):
        return False
    change = abs(current[0].amplitude - previous[0].amplitude)
    return change <= RETREND_SETTLED * current[0].amplitude


def _prewhiten(
    lc: LightCurve, settings: Settings
) -> tuple[list[Frequency], list[np.ndarray], bool]:
    grid = frequency_grid(lc, settings)
    reference = float(lc.time[0])
    shifted = lc.time - reference
    residual = lc.flux - 1.0
    frequencies: list[Frequency] = []
    models: list[np.ndarray] = []
    beyond = False
    for _ in range(settings.max_frequencies if len(grid) else 0):
        spectrum = amplitude_spectrum(lc.time, residual, grid)
        peak = _refine_peak(
            lc.time, residual, grid, float(grid[np.argmax(spectrum)]), lc.baseline, settings
        )
        sine, cosine = _fit_sinusoid(shifted, residual, peak)
        amplitude = float(np.hypot(sine, cosine))
        snr = amplitude / _local_noise(grid, spectrum, peak, lc.baseline)
        if not snr >= settings.min_snr:
            break
        model = _sinusoid(shifted, peak, sine, cosine)
        residual = residual - model
        if not grid[0] < peak < grid[-1]:
            # The spectrum still rises at the edge, the signal lies outside the searched range.
            beyond = beyond or not frequencies
            continue
        frequencies.append(
            Frequency(peak, amplitude, snr, _epoch_of_maximum(reference, peak, sine, cosine))
        )
        models.append(model)
    return frequencies, models, beyond


def _refine_peak(
    time: np.ndarray,
    flux: np.ndarray,
    grid: np.ndarray,
    peak: float,
    baseline: float,
    settings: Settings,
) -> float:
    step = 1.0 / (baseline * settings.oversample * REFINE_FACTOR)
    low = max(peak - 1.0 / baseline, grid[0])
    high = min(peak + 1.0 / baseline, grid[-1])
    fine = np.arange(low, high + step / 2, step)
    return float(fine[np.argmax(amplitude_spectrum(time, flux, fine, method="cython"))])


def _fit_sinusoid(time: np.ndarray, flux: np.ndarray, frequency: float) -> tuple[float, float]:
    phase = 2 * np.pi * frequency * time
    # Linear least squares for flux = a sin(2 pi f t) + b cos(2 pi f t) + c.
    design = np.column_stack([np.sin(phase), np.cos(phase), np.ones_like(time)])
    coefficients = np.linalg.lstsq(design, flux, rcond=None)[0]
    return float(coefficients[0]), float(coefficients[1])


def _sinusoid(time: np.ndarray, frequency: float, sine: float, cosine: float) -> np.ndarray:
    phase = 2 * np.pi * frequency * time
    return sine * np.sin(phase) + cosine * np.cos(phase)


def _epoch_of_maximum(reference: float, frequency: float, sine: float, cosine: float) -> float:
    phase = np.arctan2(cosine, sine)
    # a sin(x) + b cos(x) = A sin(x + phi) with phi = atan2(b, a), it peaks where x + phi = pi / 2.
    return reference + float(((np.pi / 2 - phase) / (2 * np.pi)) % 1.0) / frequency


def _local_noise(grid: np.ndarray, spectrum: np.ndarray, peak: float, baseline: float) -> float:
    distance = np.abs(grid - peak)
    # The peak itself is left out, it is a few resolutions 1 / T wide.
    outside = distance >= NOISE_EXCLUSION / baseline
    window = outside & (distance <= NOISE_WINDOW)
    if window.sum() < MIN_NOISE_BINS:
        window = outside
    return float(np.median(spectrum[window]))


def _period_error(primary: Frequency, residual: np.ndarray, baseline: float) -> float:
    # Montgomery and O'Donoghue (1999) frequency uncertainty, propagated to period.
    sigma = np.sqrt(6.0 / len(residual)) * np.std(residual) / (np.pi * baseline * primary.amplitude)
    return float(sigma / primary.frequency**2)


def _orbit_consistency(
    lc: LightCurve, signal: np.ndarray, frequency: float
) -> tuple[float | None, float | None]:
    orbits = [
        orbit
        for orbit in np.unique(lc.orbit)
        if np.sum(lc.orbit == orbit) >= MIN_ORBIT_SHARE * len(lc.orbit)
    ]
    if len(orbits) < 2:
        return None, None
    reference = lc.time[0]
    amplitudes = []
    for orbit in orbits:
        inside = lc.orbit == orbit
        sine, cosine = _fit_sinusoid(lc.time[inside] - reference, signal[inside], frequency)
        amplitudes.append(complex(sine, cosine))
    # Each orbit's sinusoid is a complex amplitude a + ib, equal ones give ratio 1 and coherence 1.
    moduli = np.abs(amplitudes)
    return float(moduli.min() / moduli.max()), float(abs(sum(amplitudes)) / moduli.sum())


def _search_eclipse(
    lc: LightCurve, settings: Settings, frequencies: list[Frequency], models: list[np.ndarray]
) -> Eclipse | None:
    """Best box-shaped dip train, kept only if eclipses plus the unrelated sinusoids explain the
    light curve better than all the sinusoids do."""
    time, flux = _bin(lc.time, lc.flux, max(ECLIPSE_BIN, lc.cadence))
    longest = lc.baseline / (MIN_ECLIPSES - 1)
    if longest <= ECLIPSE_MIN_PERIOD or len(time) < MIN_POINTS / 4:
        return None
    sigma = MAD_TO_SIGMA * np.median(np.abs(flux - np.median(flux)))
    if not sigma > 0:
        return None
    bls = BoxLeastSquares(time, flux, dy=np.full_like(flux, sigma))
    periods = 1.0 / np.linspace(1.0 / longest, 1.0 / ECLIPSE_MIN_PERIOD, ECLIPSE_GRID)
    result = bls.power(periods, ECLIPSE_DURATIONS, objective="snr")
    best = int(np.nanargmax(result.power))
    period, duration, epoch = _shortest_alias(
        bls,
        float(result.period[best]),
        float(result.duration[best]),
        float(result.transit_time[best]),
        float(result.power[best]),
    )

    stats = bls.compute_stats(period, duration, epoch)
    depth = float(stats["depth"][0])
    if not depth > 0:
        return None
    events = int(np.sum(np.asarray(stats["per_transit_count"]) > 0))
    inside = bls.transit_mask(time, period, duration, epoch)
    snr = _pink_snr(time, flux, inside, depth, duration, events)
    if snr < settings.eclipse_min_snr or events < MIN_ECLIPSES:
        return None
    box = bls.model(lc.time, period, duration, epoch)
    unrelated = [
        model
        for frequency, model in zip(frequencies, models, strict=True)
        if not is_harmonic(frequency.frequency, 0.5 / period, 1.0 / lc.baseline)
    ]
    with_eclipses = lc.flux - box - sum(unrelated, np.zeros_like(lc.flux))
    if np.var(with_eclipses) >= np.var(lc.flux - 1.0 - sum(models, np.zeros_like(lc.flux))):
        return None
    period, epoch = _alternating(bls, period, duration, epoch, longest)
    stats = bls.compute_stats(period, duration, epoch)
    secondary, secondary_error = stats["depth_phased"]
    if not secondary > SECONDARY_SIGMA * secondary_error:
        secondary = 0.0
    odd, even = stats["depth_odd"], stats["depth_even"]
    spread = np.hypot(odd[1], even[1])
    return Eclipse(
        period=period,
        duration=duration,
        depth=_central_depth(lc, period, duration, epoch, 0.0) or float(stats["depth"][0]),
        snr=float(snr),
        epoch=float(epoch - period * np.floor((epoch - lc.time[0]) / period)),
        events=events,
        odd_even_sigma=float(abs(odd[0] - even[0]) / spread) if spread > 0 else 0.0,
        secondary_depth=(_central_depth(lc, period, duration, epoch, 0.5) or float(secondary))
        if secondary
        else 0.0,
    )


def _central_depth(
    lc: LightCurve, period: float, duration: float, epoch: float, phase: float
) -> float | None:
    """Depth in the middle of the eclipses, which a box averages away when they are V-shaped."""
    cycles = (lc.time - epoch) / period - phase
    offset = np.abs((cycles + 0.5) % 1.0 - 0.5) * period
    centre = offset < CENTRAL_SHARE * duration / 2
    if centre.sum() < MIN_CENTRAL_POINTS:
        return None
    return float(np.median(lc.flux) - np.median(lc.flux[centre]))


def _alternating(
    bls: BoxLeastSquares, period: float, duration: float, epoch: float, longest: float
) -> tuple[float, float]:
    """Doubles the period while every other dip is shallower, as secondary eclipses are, and
    moves the epoch to the deeper primary eclipse."""
    while 2 * period <= longest:
        stats = bls.compute_stats(period, duration, epoch)
        (odd, odd_error), (even, even_error) = stats["depth_odd"], stats["depth_even"]
        difference = abs(odd - even)
        if not (
            difference > ODD_EVEN_SIGMA * np.hypot(odd_error, even_error)
            and difference > ODD_EVEN_SHARE * max(odd, even)
        ):
            break
        if odd > even:
            epoch += period
        period *= 2
    return period, epoch


def _pink_snr(
    time: np.ndarray,
    flux: np.ndarray,
    inside: np.ndarray,
    depth: float,
    duration: float,
    events: int,
) -> float:
    """Depth over white noise in the eclipse points plus red noise on the eclipse timescale,
    following Pont et al. (2006)."""
    outside = flux[~inside]
    white = MAD_TO_SIGMA * np.median(np.abs(outside - np.median(outside)))
    chunks = np.floor((time[~inside] - time[0]) / duration).astype(int)
    counts = np.bincount(chunks)
    full = counts >= 2
    means = np.bincount(chunks, outside)[full] / counts[full]
    per_chunk = float(np.mean(counts[full])) if full.any() else 1.0
    red = max(float(np.var(means)) - white**2 / per_chunk, 0.0) if full.sum() > 2 else 0.0
    # Pont et al. (2006): sigma^2 = sigma_white^2 / n_points + sigma_red^2 / n_eclipses.
    noise = np.sqrt(white**2 / max(int(inside.sum()), 1) + red / max(events, 1))
    return float(depth / noise) if noise > 0 else 0.0


def _shortest_alias(
    bls: BoxLeastSquares, period: float, duration: float, epoch: float, power: float
) -> tuple[float, float, float]:
    """The shortest of the period and its fractions that fits the dips about as well."""
    best = (period, duration, epoch)
    for divisor in ALIAS_DIVISORS:
        trial = period / divisor
        if trial < ECLIPSE_MIN_PERIOD:
            continue
        grid = trial * (1.0 + np.linspace(-ALIAS_WINDOW, ALIAS_WINDOW, ALIAS_STEPS))
        result = bls.power(grid, ECLIPSE_DURATIONS, objective="snr")
        index = int(np.nanargmax(result.power))
        if result.power[index] >= ALIAS_TOLERANCE * power:
            best = (
                float(result.period[index]),
                float(result.duration[index]),
                float(result.transit_time[index]),
            )
    return best


def _in_eclipse(time: np.ndarray, eclipse: Eclipse, phase: float = 0.0) -> np.ndarray:
    cycles = (time - eclipse.epoch) / eclipse.period - phase
    distance = np.abs((cycles + 0.5) % 1.0 - 0.5) * eclipse.period
    return (distance < eclipse.duration / 2).astype(float)


def _bin(time: np.ndarray, flux: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    index = np.floor((time - time[0]) / width).astype(int)
    counts = np.bincount(index)
    filled = counts > 0
    return (
        np.bincount(index, time)[filled] / counts[filled],
        np.bincount(index, flux)[filled] / counts[filled],
    )
