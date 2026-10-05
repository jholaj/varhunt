"""Field, thresholds and run layout, resolved once from the command line and saved with the run."""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

from astropy.coordinates import SkyCoord

RUN_FILE = "run.json"
# The detrending window spans this many of the longest searched periods so those survive it.
DETREND_FACTOR = 1.5


@dataclass(frozen=True)
class Field:
    ra: float
    dec: float
    radius_deg: float

    @property
    def center(self) -> SkyCoord:
        return SkyCoord(self.ra, self.dec, unit="deg")


@dataclass(frozen=True)
class Settings:
    min_snr: float = 5.0
    min_period: float = 0.005
    max_period: float = 2.0
    max_frequencies: int = 5
    oversample: int = 10
    eclipse_min_snr: float = 8.0
    min_orbit_ratio: float = 0.4
    min_orbit_coherence: float = 0.7
    link_radius_arcsec: float = 180.0
    systematic_min_stars: int = 5
    match_radius_arcsec: float = 21.0

    @property
    def detrend_window(self) -> float:
        return DETREND_FACTOR * self.max_period


@dataclass(frozen=True)
class Resources:
    cache_dir: Path
    workers: int


@dataclass(frozen=True)
class Run:
    root: Path
    field: Field
    sector: int
    settings: Settings

    @property
    def stars_csv(self) -> Path:
        return self.root / "stars.csv"

    @property
    def candidates_csv(self) -> Path:
        return self.root / "candidates.csv"

    @property
    def summary_md(self) -> Path:
        return self.root / "summary.md"

    @property
    def completeness_md(self) -> Path:
        return self.root / "completeness.md"

    @property
    def plots_dir(self) -> Path:
        return self.root / "plots"

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "field": asdict(self.field),
            "sector": self.sector,
            "settings": asdict(self.settings),
        }
        (self.root / RUN_FILE).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, root: Path) -> Self:
        payload = json.loads((root / RUN_FILE).read_text())
        return cls(
            root=root,
            field=Field(**payload["field"]),
            sector=payload["sector"],
            settings=Settings(**payload["settings"]),
        )


def default_cache_dir() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "varhunt"
