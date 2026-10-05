"""Known variables, Gaia neighbours and the names and magnitudes a VSX entry needs."""

from dataclasses import dataclass
from typing import Self

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord, get_constellation
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier

from varhunt.config import Field
from varhunt.sources.archive import retried

VSX = "B/vsx/vsx"
GAIA_CLASSES = "I/358/vclassre"
GAIA_SOURCES = "I/355/gaiadr3"
NEIGHBOUR_MAX_GMAG = 19.0
SAMPLE_POOL = 400
COUNTERPART_ARCSEC = 3.0
APASS = "II/336/apass9"
# APASS V saturates for brighter stars, VSX trusts it within this range.
APASS_RANGE = (10.0, 16.5)
# Catalogue designations VSX uses, the first one a star has becomes its primary name.
NAME_PREFIXES = ("HD ", "BD", "CD", "CPD", "TYC ", "GSC ", "UCAC4 ", "2MASS J", "TIC ", "Gaia DR3 ")
# Gaia G and BP-RP to Johnson V, Riello et al. (2021), G - V as a polynomial in BP-RP.
G_MINUS_V = (-0.02704, 0.01424, -0.2156, 0.01426)


@dataclass(frozen=True)
class KnownVariable:
    name: str
    kind: str
    period: float | None
    ra: float
    dec: float


@dataclass(frozen=True)
class Neighbour:
    source_id: int
    ra: float
    dec: float
    gmag: float

    @property
    def name(self) -> str:
        return f"Gaia DR3 {self.source_id}"

    @property
    def flux(self) -> float:
        return 10 ** (-0.4 * self.gmag)


@dataclass(frozen=True)
class Identity:
    """How a VSX entry names, places and measures a star, from Gaia DR3, APASS and SIMBAD."""

    name: str
    aliases: tuple[str, ...]
    position: SkyCoord
    constellation: str
    v_mag: float
    v_source: str


class KnownCatalog:
    def __init__(self, variables: list[KnownVariable]):
        self.variables = variables
        self._coords = (
            SkyCoord([v.ra for v in variables], [v.dec for v in variables], unit="deg")
            if variables
            else None
        )

    @classmethod
    def query(cls, field: Field, margin_arcsec: float) -> Self:
        radius = field.radius_deg * u.deg + margin_arcsec * u.arcsec
        return cls(_vsx(field.center, radius) + _gaia_classes(field.center, radius))

    def nearest(self, ra: float, dec: float) -> tuple[KnownVariable, float] | None:
        if self._coords is None:
            return None
        separation = SkyCoord(ra, dec, unit="deg").separation(self._coords).arcsec
        index = int(np.argmin(separation))
        return self.variables[index], float(separation[index])


@retried
def identity(tic: int, gaia_id: int, ra: float, dec: float) -> Identity:
    """Gaia DR3 position at J2000, catalogue names and a V magnitude for a VSX submission."""
    where = SkyCoord(ra, dec, unit="deg")
    radius = COUNTERPART_ARCSEC * u.arcsec
    sources = Vizier(
        columns=["Source", "RAJ2000", "DEJ2000", "Gmag", "BP-RP"], row_limit=10
    ).query_region(where, radius=radius, catalog=GAIA_SOURCES)[0]
    source = sources[np.asarray(sources["Source"]) == gaia_id][0]
    position = SkyCoord(float(source["RAJ2000"]), float(source["DEJ2000"]), unit="deg")
    names = _names(tic, gaia_id)
    apass = Vizier(columns=["Vmag"], row_limit=1).query_region(where, radius=radius, catalog=APASS)
    v_apass = _optional_float(apass[0]["Vmag"][0]) if apass else None
    if v_apass is not None and APASS_RANGE[0] <= v_apass <= APASS_RANGE[1]:
        v_mag, v_source = v_apass, "APASS DR9"
    else:
        v_mag = johnson_v(float(source["Gmag"]), float(source["BP-RP"]))
        v_source = "Gaia DR3 G and BP-RP (Riello et al. 2021)"
    return Identity(
        name=names[0],
        aliases=tuple(names[1:]),
        position=position,
        constellation=get_constellation(position),
        v_mag=v_mag,
        v_source=v_source,
    )


def johnson_v(gmag: float, bp_rp: float) -> float:
    """Johnson V from Gaia G and BP-RP, for stars that APASS does not measure well."""
    return gmag - float(np.polyval(G_MINUS_V[::-1], bp_rp))


@retried
def neighbours(ra: float, dec: float, radius_arcsec: float) -> list[Neighbour]:
    vizier = Vizier(
        columns=["Source", "RA_ICRS", "DE_ICRS", "Gmag"],
        column_filters={"Gmag": f"<{NEIGHBOUR_MAX_GMAG}"},
        row_limit=-1,
    )
    tables = vizier.query_region(
        SkyCoord(ra, dec, unit="deg"), radius=radius_arcsec * u.arcsec, catalog=GAIA_SOURCES
    )
    if not tables:
        return []
    return [
        Neighbour(
            int(row["Source"]), float(row["RA_ICRS"]), float(row["DE_ICRS"]), float(row["Gmag"])
        )
        for row in tables[0]
    ]


@retried
def catalogued_sample(
    kind: str,
    periods: tuple[float, float],
    magnitudes: tuple[float, float],
    size: int,
    seed: int,
) -> list[KnownVariable]:
    """Random VSX variables of one type within period and magnitude ranges."""
    vizier = Vizier(
        columns=["Name", "Type", "Period", "RAJ2000", "DEJ2000"],
        column_filters={
            "Type": f"={kind}",
            "Period": f"{periods[0]}..{periods[1]}",
            "max": f"{magnitudes[0]}..{magnitudes[1]}",
        },
        row_limit=SAMPLE_POOL,
    )
    tables = vizier.query_constraints(catalog=VSX)
    if not tables:
        return []
    pool = tables[0]
    chosen = np.random.default_rng(seed).choice(len(pool), size=min(size, len(pool)), replace=False)
    return [
        KnownVariable(
            name=str(pool["Name"][i]),
            kind=kind,
            period=float(pool["Period"][i]),
            ra=float(pool["RAJ2000"][i]),
            dec=float(pool["DEJ2000"][i]),
        )
        for i in sorted(chosen)
    ]


def counterpart(ra: float, dec: float, candidates: list[Neighbour]) -> Neighbour | None:
    """The Gaia source that is the target itself, if one lies within a few arcseconds."""
    if not candidates:
        return None
    separation = separations_arcsec(ra, dec, candidates)
    index = int(np.argmin(separation))
    return candidates[index] if separation[index] <= COUNTERPART_ARCSEC else None


def separations_arcsec(ra: float, dec: float, candidates: list[Neighbour]) -> np.ndarray:
    coords = SkyCoord([n.ra for n in candidates], [n.dec for n in candidates], unit="deg")
    return SkyCoord(ra, dec, unit="deg").separation(coords).arcsec


@retried
def _vsx(center: SkyCoord, radius: u.Quantity) -> list[KnownVariable]:
    vizier = Vizier(columns=["Name", "Type", "Period", "RAJ2000", "DEJ2000"], row_limit=-1)
    tables = vizier.query_region(center, radius=radius, catalog=VSX)
    if not tables:
        return []
    return [
        KnownVariable(
            name=str(row["Name"]),
            kind=str(row["Type"]),
            period=_optional_float(row["Period"]),
            ra=float(row["RAJ2000"]),
            dec=float(row["DEJ2000"]),
        )
        for row in tables[0]
    ]


@retried
def _gaia_classes(center: SkyCoord, radius: u.Quantity) -> list[KnownVariable]:
    vizier = Vizier(columns=["Source", "Class", "RA_ICRS", "DE_ICRS"], row_limit=-1)
    tables = vizier.query_region(center, radius=radius, catalog=GAIA_CLASSES)
    if not tables:
        return []
    return [
        KnownVariable(
            name=f"Gaia DR3 {row['Source']}",
            kind=str(row["Class"]),
            period=None,
            ra=float(row["RA_ICRS"]),
            dec=float(row["DE_ICRS"]),
        )
        for row in tables[0]
    ]


def _names(tic: int, gaia_id: int) -> list[str]:
    found = Simbad.query_objectids(f"Gaia DR3 {gaia_id}")
    known = [" ".join(str(name).split()) for name in found["id"]] if found is not None else []
    names = {name for name in known if name.startswith(NAME_PREFIXES)}
    names |= {f"TIC {tic}", f"Gaia DR3 {gaia_id}"}
    return sorted(names, key=_name_rank)


def _name_rank(name: str) -> int:
    return next(i for i, prefix in enumerate(NAME_PREFIXES) if name.startswith(prefix))


def _optional_float(value: object) -> float | None:
    if np.ma.is_masked(value):
        return None
    number = float(value)
    return number if np.isfinite(number) else None
