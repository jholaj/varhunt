from pathlib import Path

from astropy.coordinates import SkyCoord

from varhunt.output.vsx import Attachment, Submission, write_draft
from varhunt.sources.catalogs import Identity

IDENTITY = Identity(
    name="TYC 6611-772-1",
    aliases=("2MASS J09522416-2625058", "TIC 22080156", "Gaia DR3 5657571753994979072"),
    position=SkyCoord(148.1006958, -26.4183072, unit="deg"),
    constellation="Hydra",
    v_mag=11.30,
    v_source="APASS DR9",
)


def submission(tmp_path: Path) -> Submission:
    return Submission(
        identity=IDENTITY,
        variable_type="EA",
        amplitude=0.151,
        period=6.0150244,
        period_error=4e-7,
        epoch_hjd=2461069.7424,
        epoch_bjd=2461069.74315,
        epoch_kind="primary minimum",
        duration_percent=2.85,
        discoverer="Jaroslav Holaj",
        remarks="Min II = 0.111 TESS (amplitude) at phase 0.4994.",
        attachments=(Attachment(tmp_path / "phase.png", "Phase plot", "Folded light curve."),),
        evidence=("25 primary minima timed.",),
    )


def test_the_draft_follows_the_vsx_form(tmp_path):
    path = tmp_path / "draft.md"
    write_draft(path, submission(tmp_path))
    text = path.read_text()
    assert "| Position (J2000.0) | 09 52 24.167 -26 25 05.906, then click Spot check |" in text
    assert "| Primary name | TYC 6611-772-1, then click Check name |" in text
    assert "| Maximum magnitude | 11.30, passband V (APASS DR9) |" in text
    assert "| Minimum magnitude | 0.1510, passband TESS, Minimum is amplitude |" in text
    assert "| Epoch | 2461069.7424 HJD, time of primary minimum |" in text
    assert "| Rise/eclipse duration | 2.85 |" in text
    assert "1. `phase.png`" in text
    assert "- 25 primary minima timed." in text
