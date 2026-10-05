import pytest

from varhunt.sources import archive
from varhunt.sources.archive import retried


def flaky(failures: int):
    calls = []

    @retried
    def query() -> str:
        calls.append(1)
        if len(calls) <= failures:
            raise TimeoutError("timed out")
        return "rows"

    return query, calls


def test_a_query_that_times_out_is_repeated(monkeypatch):
    monkeypatch.setattr(archive.time, "sleep", lambda seconds: None)
    query, calls = flaky(failures=2)
    assert query() == "rows"
    assert len(calls) == 3


def test_a_dead_archive_raises_in_the_end(monkeypatch):
    monkeypatch.setattr(archive.time, "sleep", lambda seconds: None)
    query, calls = flaky(failures=10)
    with pytest.raises(TimeoutError):
        query()
    assert len(calls) == archive.ATTEMPTS


def test_missing_data_is_not_asked_for_again(monkeypatch):
    monkeypatch.setattr(archive.time, "sleep", lambda seconds: None)
    calls = []

    @retried
    def query() -> str:
        calls.append(1)
        raise FileNotFoundError("no such sector")

    with pytest.raises(FileNotFoundError):
        query()
    assert len(calls) == 1
