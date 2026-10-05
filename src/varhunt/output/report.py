"""Per-star results, their CSV form and the summary of a run written for a human."""

import csv
import math
from collections import Counter
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from types import NoneType, UnionType
from typing import get_args, get_type_hints

import numpy as np
from astropy.coordinates import SkyCoord

from varhunt.checks.evidence import MAX_RUWE
from varhunt.config import Run

CANDIDATE = "candidate"
KNOWN = "known"
KNOWN_NEARBY = "known_nearby"
SHARED = "shared"
SYSTEMATIC = "systematic"
INCONSISTENT = "inconsistent"
SLOW = "slow"
QUIET = "quiet"
STATUSES = (CANDIDATE, KNOWN, KNOWN_NEARBY, SHARED, SYSTEMATIC, INCONSISTENT, SLOW, QUIET)
REJECTING_VERDICTS = ("off_target", "motion", "no_signal")
# Too little signal in another sector proves nothing, a signal elsewhere does.
MISPLACED_VERDICTS = ("off_target", "motion")
BTJD_OFFSET = 2457000.0
PERIOD_MATCH = 0.01
SURE_MIN_STRENGTH = 10.0
# A weaker signal is just as sure when two more sectors, other years and cameras, repeat it.
SURE_REPEATED_STRENGTH = 7.0
SURE_REPEATS = 2
# A thousandth of a magnitude, below that a discovery is hard to show to anyone.
SURE_MIN_RANGE_PPT = 1.0
SURE_MAX_OFFSET_PX = 0.3
SURE_MAX_CENTROID_ERROR_PX = 0.15
# A star that would need this semi-amplitude of its own cannot have faked the signal.
IMPOSSIBLE_BLEND_PPT = 1000.0
# Measurement errors allow a little over one before rotation is ruled out.
MAX_SIN_I = 1.15


@dataclass
class Result:
    tic: int
    ra: float
    dec: float
    tmag: float
    teff: float | None
    file: str
    n_points: int
    rms_ppt: float
    period: float | None = None
    period_error: float | None = None
    epoch: float | None = None
    epoch_kind: str = ""
    light_max_ppt: float | None = None
    light_min_ppt: float | None = None
    amplitude_ppt: float | None = None
    snr: float | None = None
    periods: str = ""
    orbit_ratio: float | None = None
    orbit_coherence: float | None = None
    eclipse_period: float | None = None
    eclipse_duration: float | None = None
    eclipse_depth_ppt: float | None = None
    eclipse_secondary_ppt: float | None = None
    eclipse_snr: float | None = None
    eclipse_epoch: float | None = None
    type_hint: str = ""
    status: str = QUIET
    shared_with: int | None = None
    echoes: int = 0
    known_name: str = ""
    known_type: str = ""
    known_period: float | None = None
    known_separation: float | None = None
    gaia_id: str = ""
    blend_neighbour: str = ""
    blend_required_ppt: float | None = None
    vet_verdict: str = ""
    vet_offset_px: float | None = None
    vet_centroid_error_px: float | None = None
    vet_source: str = ""
    vet_companions: str = ""
    vet_companion_needs_ppt: float | None = None
    other_verdicts: str = ""
    nearby_catalog_hits: str = ""
    survives_wider_detrending: bool | None = None
    other_sectors: str = ""
    confirmed_sectors: str = ""
    catalog_hits: str = ""
    simbad_name: str = ""
    simbad_type: str = ""
    gaia_ruwe: float | None = None
    gaia_scatter: float | None = None
    gaia_scatter_reliable: bool | None = None
    joint_period: float | None = None
    joint_period_error: float | None = None
    joint_span: float | None = None
    joint_aliases: bool | None = None
    equatorial_velocity: float | None = None
    breakup_velocity: float | None = None
    vsini: float | None = None

    @property
    def strength(self) -> float:
        return max(self.snr or 0.0, self.eclipse_snr or 0.0)

    @property
    def could_be_sure(self) -> bool:
        """Whether vetting and confirming could still make it sure."""
        return (
            self.status == CANDIDATE
            and self.strength >= SURE_REPEATED_STRENGTH
            and self.light_max_ppt - self.light_min_ppt >= SURE_MIN_RANGE_PPT
        )

    @property
    def sure(self) -> bool:
        """Strong, on the target, repeated in another sector, in no catalogue, and no unresolved
        star at the signal position could have faked it."""
        return (
            self.status == CANDIDATE
            and self.vet_verdict == "on_target"
            and self.vet_offset_px is not None
            and self.vet_offset_px <= SURE_MAX_OFFSET_PX
            and self.vet_centroid_error_px is not None
            and self.vet_centroid_error_px <= SURE_MAX_CENTROID_ERROR_PX
            and self._strong_enough
            and self.light_max_ppt - self.light_min_ppt >= SURE_MIN_RANGE_PPT
            and not self.catalog_hits
            and self._blend_ruled_out
            and self._located_elsewhere
            and self.survives_wider_detrending is True
        )

    @property
    def _strong_enough(self) -> bool:
        repeats = len(self.confirmed_sectors.split(";")) if self.confirmed_sectors else 0
        if self.strength >= SURE_MIN_STRENGTH:
            return repeats >= 1
        return self.strength >= SURE_REPEATED_STRENGTH and repeats >= SURE_REPEATS

    @property
    def _located_elsewhere(self) -> bool:
        verdicts = [item.split(":")[1] for item in self.other_verdicts.split(";") if item]
        return "on_target" in verdicts and not any(v in MISPLACED_VERDICTS for v in verdicts)

    @property
    def _blend_ruled_out(self) -> bool:
        needed = self.vet_companion_needs_ppt
        return not self.vet_companions or (needed is not None and needed >= IMPOSSIBLE_BLEND_PPT)


def write_results(path: Path, results: list[Result]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(Result)])
        writer.writeheader()
        for result in results:
            writer.writerow({key: _format(value) for key, value in asdict(result).items()})


def read_results(path: Path) -> list[Result]:
    kinds = get_type_hints(Result)
    with path.open(newline="") as handle:
        return [
            Result(**{key: _parse(text, kinds[key]) for key, text in row.items()})
            for row in csv.DictReader(handle)
        ]


def write_summary(run: Run, results: list[Result]) -> None:
    counts = Counter(r.status for r in results)
    candidates = sorted(
        (r for r in results if r.status == CANDIDATE), key=lambda r: r.strength, reverse=True
    )
    lines = [
        f"# varhunt, TESS sector {run.sector}",
        "",
        f"Field RA {run.field.ra:.4f}, Dec {run.field.dec:+.4f}, "
        f"radius {run.field.radius_deg} deg.",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from {len(results)} QLP light curves.",
        "",
        "| Status | Stars |",
        "| --- | ---: |",
        *(f"| {status} | {counts.get(status, 0)} |" for status in STATUSES),
        "",
        _recovery(results, run),
        "",
        "## Candidates",
        "",
    ]
    standing = [r for r in candidates if r.vet_verdict not in REJECTING_VERDICTS]
    rejected = [r for r in candidates if r.vet_verdict in REJECTING_VERDICTS]
    lines += candidate_table(standing) if standing else ["No candidates left in this field."]
    if rejected:
        lines += [
            "",
            "## Rejected by the pixel check",
            "",
            "Their signal is not a brightness change of the target itself, see the verdict.",
            "",
            *candidate_table(rejected),
        ]
    for result in standing:
        lines += ["", *candidate_details(result)]
    run.summary_md.write_text("\n".join(lines) + "\n")


def write_survey(path: Path, runs: list[tuple[Run, list[Result]]]) -> None:
    standing = [
        (run, r)
        for run, results in runs
        for r in results
        if r.status == CANDIDATE and r.vet_verdict not in REJECTING_VERDICTS
    ]
    standing.sort(key=lambda item: (not item[1].sure, -item[1].strength))
    sure = [item for item in standing if item[1].sure]
    stars = sum(len(results) for _, results in runs)
    lines = [
        "# varhunt survey",
        "",
        f"Updated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. {len(runs)} fields, {stars} stars, "
        f"{len(standing)} candidates past the pixel check, {len(sure)} sure.",
        "",
        "## Candidates",
        "",
        *candidate_table([r for _, r in standing]),
    ]
    for run, result in standing:
        lines += ["", *candidate_details(result), f"- Run: {run.root.name}"]
    path.write_text("\n".join(lines) + "\n")


def write_sure(path: Path, found: list[tuple[Run, Result]], runs: int) -> None:
    lines = [
        "# Sure candidates",
        "",
        f"Updated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC. {len(found)} sure candidates in {runs} "
        "runs, strongest first.",
        "",
        *candidate_table([r for _, r in found]),
    ]
    for run, result in found:
        lines += ["", *candidate_details(result), f"- Run: {run.root}"]
    path.write_text("\n".join(lines) + "\n")


def period_matches(found: float | None, known: float | None) -> bool:
    if found is None or known is None:
        return False
    return any(abs(found * factor - known) <= PERIOD_MATCH * known for factor in (0.5, 1.0, 2.0))


def candidate_table(results: list[Result]) -> list[str]:
    return [
        "| TIC | Tmag | Period (d) | Range (ppt) | S/N | Type | Blend needs | Pixel check "
        "| Other sectors | Sure |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |",
        *(_table_row(r) for r in results),
    ]


def candidate_details(result: Result) -> list[str]:
    coord = SkyCoord(result.ra, result.dec, unit="deg")
    lines = [
        f"### TIC {result.tic}",
        "",
        *([f"- Known as: {result.simbad_name}"] if result.simbad_name else []),
        f"- Gaia DR3: {result.gaia_id or 'no counterpart within 3 arcsec'}",
        f"- RA, Dec (J2000): {coord.to_string('hmsdms', sep=':', precision=2)}",
        f"- Type hint: {result.type_hint}, needs a human check",
        f"- Light range in the TESS band: {_light_range(result)}",
        f"- Period: {_period(result)}",
        f"- Epoch of {result.epoch_kind}: BJD {result.epoch + BTJD_OFFSET:.5f}",
    ]
    if result.eclipse_depth_ppt is not None:
        lines.append(
            f"- Eclipse depth: {result.eclipse_depth_ppt:.2f} ppt, S/N {result.eclipse_snr:.1f}"
        )
        if result.eclipse_secondary_ppt:
            faint = result.tmag - 2.5 * np.log10(1 - result.eclipse_secondary_ppt / 1e3)
            lines.append(
                f"- Secondary eclipse at phase 0.5: {result.eclipse_secondary_ppt:.2f} ppt, "
                f"{faint:.3f} mag"
            )
    else:
        lines += [
            f"- Strongest sinusoid: {result.amplitude_ppt:.2f} ppt, S/N {result.snr:.1f}",
            f"- All detected periods (d): {result.periods.replace(';', ', ')}",
        ]
    if result.blend_neighbour:
        lines.append(
            f"- Blend: {result.blend_neighbour} would need {result.blend_required_ppt:.0f} ppt "
            "of its own to fake this"
        )
    else:
        lines.append("- Blend: no Gaia neighbour within 3 TESS pixels")
    if result.echoes:
        lines.append(f"- Also seen in {result.echoes} neighbouring light curves, attributed here")
    if result.vet_verdict:
        error = result.vet_centroid_error_px
        spread = f" +- {error:.2f}" if error is not None else ""
        lines.append(
            f"- Pixel check: {result.vet_verdict}, offset {result.vet_offset_px:.2f}{spread} px, "
            f"likely source {result.vet_source or 'unknown'}"
        )
        if result.vet_companions:
            names = result.vet_companions.replace(";", ", ")
            lines.append(
                f"- Unresolved Gaia sources at the signal position: {names}, the easiest would "
                f"need {result.vet_companion_needs_ppt:.0f} ppt"
            )
    else:
        lines.append("- Pixel check: not run yet (varhunt vet)")
    if result.other_sectors:
        lines.append(f"- Other sectors: {_sectors(result)} repeat the signal")
    if result.other_verdicts:
        located = result.other_verdicts.replace(":", " ").replace(";", ", ")
        lines.append(f"- Pixel check in the repeating sectors: {located}")
    if result.survives_wider_detrending is not None:
        kept = (
            "kept"
            if result.survives_wider_detrending
            else "lost, probably left behind by a slower variation"
        )
        lines.append(f"- Signal with three times wider detrending: {kept}")
    if result.joint_period is not None:
        aliases = ", the cycle count across the gaps is not certain" if result.joint_aliases else ""
        if result.type_hint == "ROT":
            aliases += ", spots drift over the years so prefer the period of one sector"
        lines.append(
            f"- Period from all sectors over {result.joint_span:.0f} d: "
            f"{result.joint_period:.8f} +- {result.joint_period_error:.8f} d{aliases}"
        )
        lines.append(f"- Sectors folded together: plots/TIC{result.tic}_sectors.png")
    if result.other_sectors or result.catalog_hits or result.simbad_type:
        hits = result.catalog_hits.replace(";", ", ") or "none"
        lines.append(f"- Variability catalogues listing it: {hits}")
        if result.nearby_catalog_hits:
            nearby = result.nearby_catalog_hits.replace(";", ", ")
            lines.append(
                f"- Variability catalogues listing another star within 30 arcsec: {nearby}, "
                "check whether it is this star's light"
            )
        lines.append(f"- SIMBAD type: {result.simbad_type or 'no SIMBAD entry'}")
    if result.equatorial_velocity is not None:
        lines.append(f"- Rotation check: {_rotation(result)}")
    if result.gaia_ruwe is not None and result.gaia_ruwe > MAX_RUWE:
        lines.append(
            f"- Gaia RUWE {result.gaia_ruwe:.2f}: probably an unresolved binary, either star "
            "may be the variable"
        )
    if result.gaia_scatter is not None:
        reliability = "" if result.gaia_scatter_reliable else ", unreliable (close companion)"
        lines.append(f"- Gaia G scatter: {result.gaia_scatter:.1f} times the field{reliability}")
    lines.append(f"- Plot: plots/TIC{result.tic}.png")
    return lines


def _recovery(results: list[Result], run: Run) -> str:
    settings = run.settings
    comparable = [
        r
        for r in results
        if r.known_period is not None
        and r.known_separation is not None
        and r.known_separation <= settings.match_radius_arcsec
        and settings.min_period <= r.known_period <= settings.max_period
    ]
    if not comparable:
        return "No catalogued variables with a period in the searched range to compare against."
    recovered = sum(period_matches(r.period, r.known_period) for r in comparable)
    return (
        f"Catalogued variables with a period between {settings.min_period} and "
        f"{settings.max_period} d: {recovered} of {len(comparable)} recovered at the catalogued "
        "period (or half or double of it)."
    )


def _table_row(result: Result) -> str:
    blend = f"{result.blend_required_ppt:.0f} ppt" if result.blend_required_ppt else "none"
    check = result.vet_verdict or "not run"
    spread = result.light_max_ppt - result.light_min_ppt
    return (
        f"| {result.tic} | {result.tmag:.2f} | {result.period:.6f} | {spread:.2f} "
        f"| {result.strength:.1f} | {result.type_hint} | {blend} | {check} "
        f"| {_sectors(result)} | {'yes' if result.sure else ''} |"
    )


def _sectors(result: Result) -> str:
    if not result.other_sectors:
        return ""
    checked = result.other_sectors.split(";")
    confirmed = result.confirmed_sectors.split(";") if result.confirmed_sectors else []
    return f"{len(confirmed)}/{len(checked)}"


def _rotation(result: Result) -> str:
    speed = result.equatorial_velocity
    text = f"as rotation the equator would move at {speed:.0f} km/s"
    if result.breakup_velocity is not None:
        text += f" (breakup {result.breakup_velocity:.0f} km/s)"
        if speed > result.breakup_velocity:
            return text + ", faster than breakup so not rotation"
    if result.vsini is None:
        return text + ", no v sin i measured"
    sin_i = result.vsini / speed
    verdict = (
        "consistent with rotation"
        if sin_i <= MAX_SIN_I
        else "the star spins faster than this period allows"
    )
    return text + f", measured v sin i {result.vsini:.0f} km/s gives sin i {sin_i:.2f}, {verdict}"


def _period(result: Result) -> str:
    error = f" +- {result.period_error:.7f}" if result.period_error is not None else ""
    strongest = float(result.periods.split(";")[0]) if result.periods else None
    doubled = strongest is not None and math.isclose(result.period, 2 * strongest, rel_tol=1e-3)
    note = ", orbital, twice the strongest photometric period" if doubled else ""
    return f"{result.period:.7f}{error} d{note}"


def _light_range(result: Result) -> str:
    # A relative flux change d moves the magnitude by -2.5 log10(1 + d).
    bright = result.tmag - 2.5 * np.log10(1 + result.light_max_ppt / 1e3)
    faint = result.tmag - 2.5 * np.log10(1 + result.light_min_ppt / 1e3)
    return f"{bright:.3f} to {faint:.3f} mag"


def _format(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _parse(text: str, kind: type | UnionType) -> object:
    options = get_args(kind)
    optional = NoneType in options
    base = next(option for option in options if option is not NoneType) if optional else kind
    if text == "":
        return None if optional else base()
    if base is bool:
        return text == "True"
    return base(text)
