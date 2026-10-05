"""Finding and downloading QLP and TESS-SPOC light curves on MAST."""

import logging
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astroquery.mast import Observations

from varhunt.config import Field
from varhunt.sources.archive import retried

log = logging.getLogger(__name__)

PROVENANCE = "QLP"
SPOC_PROVENANCE = "TESS-SPOC"
# TESS-SPOC light curves share the cache directory with the QLP ones.
QLP_DIR = "qlp"
DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri="
PRODUCT_CHUNK = 200
RETRIES = 3
MAX_DOWNLOAD_THREADS = 8
# MAST also lists a few synthetic sequence numbers in the thousands.
MAX_SECTOR = 999
FILENAME = re.compile(r"_s(\d{4})-(\d{16})_tess_v\d+_llc\.fits$")
SPOC_FILENAME = re.compile(r"_(\d{16})-s(\d{4})_tess_v\d+_lc\.fits$")
SECTOR_TRIES = 3
MIN_FIELD_STARS = 50


@dataclass(frozen=True)
class Product:
    tic: int
    sector: int
    uri: str

    @property
    def filename(self) -> str:
        return self.uri.rsplit("/", 1)[-1]


@retried
def available_sectors(field: Field) -> list[int]:
    observations = Observations.query_criteria(
        coordinates=field.center, radius=field.radius_deg * u.deg, provenance_name=PROVENANCE
    )
    sectors = np.ma.filled(observations["sequence_number"], -1).astype(int)
    return sorted({int(s) for s in sectors if 0 < s <= MAX_SECTOR})


@retried
def find_light_curves(field: Field, sector: int) -> list[Product]:
    observations = Observations.query_criteria(
        coordinates=field.center,
        radius=field.radius_deg * u.deg,
        provenance_name=PROVENANCE,
        sequence_number=sector,
    )
    products: dict[int, Product] = {}
    for start in range(0, len(observations), PRODUCT_CHUNK):
        listing = Observations.get_product_list(observations[start : start + PRODUCT_CHUNK])
        for uri in listing["dataURI"]:
            product = _parse(str(uri))
            if product is not None and product.sector == sector:
                products[product.tic] = product
    return sorted(products.values(), key=lambda p: p.tic)


def best_sector(field: Field, sectors: list[int]) -> int:
    """Newest sector that covers the field well, a field on a detector edge has only a few stars."""
    for sector in reversed(sectors[-SECTOR_TRIES:]):
        if len(find_light_curves(field, sector)) >= MIN_FIELD_STARS:
            return sector
    return sectors[-1]


@retried
def light_curves_of(tic: int) -> list[Product]:
    """QLP light curves of one star in every sector it was observed."""
    observations = Observations.query_criteria(provenance_name=PROVENANCE, target_name=str(tic))
    if not len(observations):
        return []
    products = [_parse(str(uri)) for uri in Observations.get_product_list(observations)["dataURI"]]
    found = {p.sector: p for p in products if p is not None and p.tic == tic}
    return [found[sector] for sector in sorted(found) if sector <= MAX_SECTOR]


@retried
def newest_light_curve(ra: float, dec: float, radius_arcsec: float) -> Product | None:
    """QLP light curve of the star nearest to a position, from its most recent sector."""
    observations = Observations.query_criteria(
        coordinates=SkyCoord(ra, dec, unit="deg"),
        radius=radius_arcsec * u.arcsec,
        provenance_name=PROVENANCE,
    )
    if not len(observations):
        return None
    observations.sort("distance")
    own = observations[observations["target_name"] == observations["target_name"][0]]
    sectors = np.ma.filled(own["sequence_number"], -1).astype(int)
    usable = np.flatnonzero((sectors > 0) & (sectors <= MAX_SECTOR))
    if not len(usable):
        return None
    index = usable[np.argmax(sectors[usable])]
    for uri in Observations.get_product_list(own[index : index + 1])["dataURI"]:
        product = _parse(str(uri))
        if product is not None:
            return product
    return None


@retried
def spoc_light_curves(tic: int) -> list[Product]:
    """TESS-SPOC full-frame-image light curves of one star, corrected by the mission pipeline."""
    observations = Observations.query_criteria(
        target_name=str(tic), obs_collection="HLSP", provenance_name=SPOC_PROVENANCE
    )
    if not len(observations):
        return []
    uris = Observations.get_product_list(observations)["dataURI"]
    found = (_parse_spoc(str(uri)) for uri in uris)
    products = {p for p in found if p is not None and p.sector <= MAX_SECTOR}
    return sorted(products, key=lambda p: p.sector)


def download(products: list[Product], cache_dir: Path, workers: int) -> dict[Product, Path]:
    (cache_dir / QLP_DIR).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(min(workers, MAX_DOWNLOAD_THREADS)) as pool:
        paths = list(pool.map(lambda product: _fetch(product, cache_dir), products))
    return {p: path for p, path in zip(products, paths, strict=True) if path is not None}


def local_path(cache_dir: Path, filename: str) -> Path:
    return cache_dir / QLP_DIR / filename


def _parse(uri: str) -> Product | None:
    match = FILENAME.search(uri)
    if match is None:
        return None
    return Product(tic=int(match[2]), sector=int(match[1]), uri=uri)


def _parse_spoc(uri: str) -> Product | None:
    match = SPOC_FILENAME.search(uri)
    if match is None:
        return None
    return Product(tic=int(match[1]), sector=int(match[2]), uri=uri)


def _fetch(product: Product, cache_dir: Path) -> Path | None:
    path = local_path(cache_dir, product.filename)
    if path.exists():
        return path
    partial = path.with_name(path.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            urllib.request.urlretrieve(DOWNLOAD_URL + product.uri, partial)
        except OSError as exc:
            log.debug("download of %s failed (attempt %d): %s", product.filename, attempt, exc)
            time.sleep(attempt)
            continue
        partial.replace(path)
        return path
    log.warning("giving up on %s", product.filename)
    return None
