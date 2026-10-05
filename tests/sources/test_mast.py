from varhunt.config import Field
from varhunt.sources import mast

FIELD = Field(62.613, 7.721, 0.5)


def test_a_sector_that_only_grazes_the_field_is_skipped(monkeypatch):
    counts = {32: 132, 70: 1, 71: 4}
    monkeypatch.setattr(mast, "find_light_curves", lambda field, sector: [0] * counts[sector])
    assert mast.best_sector(FIELD, [32, 70, 71]) == 32


def test_the_newest_sector_wins_when_it_covers_the_field(monkeypatch):
    monkeypatch.setattr(mast, "find_light_curves", lambda field, sector: [0] * 300)
    assert mast.best_sector(FIELD, [5, 32, 71]) == 71


def test_the_newest_sector_is_used_when_none_covers_the_field_well(monkeypatch):
    monkeypatch.setattr(mast, "find_light_curves", lambda field, sector: [0] * 3)
    assert mast.best_sector(FIELD, [5, 32, 71]) == 71
