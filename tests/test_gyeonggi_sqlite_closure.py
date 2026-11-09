"""Ensure temporary database connections close on success and input failure."""

import csv
import sqlite3

import pytest

from pickplace_ai import gyeonggi_candidates


class TrackedConnection(sqlite3.Connection):
    closed = False

    def close(self):
        self.closed = True
        super().close()


@pytest.fixture
def track_connections(monkeypatch):
    real_connect = sqlite3.connect
    opened = []

    def tracked_connect(*args, **kwargs):
        db = real_connect(*args, factory=TrackedConnection, **kwargs)
        opened.append(db)
        return db

    monkeypatch.setattr(sqlite3, "connect", tracked_connect)
    return opened


@pytest.mark.parametrize("bad_row", [False, True])
def test_gyeonggi_index_connection_is_closed(tmp_path, monkeypatch, track_connections, bad_row):
    raw, book, regions = [tmp_path / name for name in ("source.csv", "codes.xlsx", "regions.csv")]
    book.write_text("placeholder")
    regions.write_text("placeholder")
    monkeypatch.setattr(gyeonggi_candidates, "load_industries", lambda path: {"Q13": "카페"})
    monkeypatch.setattr(
        gyeonggi_candidates, "load_current_regions", lambda path, month: {"1111051500"}
    )
    with raw.open("w", encoding="cp949", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["bad"] if bad_row else gyeonggi_candidates.HEADERS)
        if not bad_row:
            writer.writerow(["202402", "1111051500", "Q13", "30"])
    kwargs = dict(
        source=raw,
        industries=book,
        current_regions=regions,
        region_reference_month="2026-10",
        output_dir=tmp_path / "output",
    )
    if bad_row:
        with pytest.raises(ValueError):
            gyeonggi_candidates.prepare(**kwargs)
    else:
        gyeonggi_candidates.prepare(**kwargs)
    assert len(track_connections) == 1
    assert track_connections[0].closed
