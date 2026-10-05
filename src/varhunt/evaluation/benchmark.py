"""Recall on catalogued variables, whether the reported period and type match VSX."""

from dataclasses import dataclass
from pathlib import Path

from varhunt.analysis.classify import ephemeris, type_hint
from varhunt.analysis.detect import detect
from varhunt.config import Settings
from varhunt.output.report import PERIOD_MATCH, period_matches
from varhunt.sources.catalogs import KnownVariable
from varhunt.sources.lightcurve import load_qlp

# VSX type, its period range in days and the hint varhunt should give it.
TYPES = {
    "EW": ((0.25, 0.6), "EW"),
    "DSCT": ((0.02, 0.2), "DSCT"),
    "RRAB": ((0.45, 0.8), "RR"),
    "EA": ((0.5, 2.0), "EA"),
}
MAGNITUDES = (9.0, 12.5)
MATCH_ARCSEC = 15.0


@dataclass(frozen=True)
class Outcome:
    variable: KnownVariable
    found: float | None
    hint: str

    @property
    def exact(self) -> bool:
        return self.found is not None and abs(self.found - self.variable.period) <= (
            PERIOD_MATCH * self.variable.period
        )

    @property
    def lenient(self) -> bool:
        return period_matches(self.found, self.variable.period)

    @property
    def typed(self) -> bool:
        return self.hint == TYPES[self.variable.kind][1]


def evaluate(variable: KnownVariable, path: Path, settings: Settings) -> Outcome:
    lc = load_qlp(path)
    detection = detect(lc, settings)
    orbit = ephemeris(lc, detection)
    return Outcome(variable, orbit.period if orbit else None, type_hint(lc, detection))


def report(outcomes: list[Outcome]) -> str:
    lines = [
        "| VSX type | Stars | Period within 1 % | Also half or double | Type hint right |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for kind in TYPES:
        group = [o for o in outcomes if o.variable.kind == kind]
        lines.append(
            f"| {kind} | {len(group)} | {sum(o.exact for o in group)} "
            f"| {sum(o.lenient for o in group)} | {sum(o.typed for o in group)} |"
        )
    lines += [
        "",
        "| Star | VSX type | VSX period (d) | Reported period (d) | Hint |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for o in outcomes:
        found = f"{o.found:.6f}" if o.found is not None else "nothing"
        lines.append(
            f"| {o.variable.name} | {o.variable.kind} | {o.variable.period:.6f} | {found} "
            f"| {o.hint} |"
        )
    return "\n".join(lines)
