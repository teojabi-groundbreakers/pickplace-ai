"""Ensure temporary database connections close on success and input failure."""

import csv
import sqlite3

import pytest

from pickplace_ai import nationwide_ingest


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
def test_store_index_connection_is_closed(tmp_path, track_connections, bad_row):
    path = tmp_path / "stores.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "상가업소번호",
                "행정동코드",
                "행정동명",
                "시도명",
                "시군구명",
                "상권업종소분류코드",
                "상권업종소분류명",
            ]
        )
        writer.writerow(
            ["1", "bad" if bad_row else "11110515", "동", "서울", "종로구", "I21201", "카페"]
        )
    if bad_row:
        with pytest.raises(ValueError):
            nationwide_ingest._aggregate_stores([path])
    else:
        nationwide_ingest._aggregate_stores([path])
    assert len(track_connections) == 1
    assert track_connections[0].closed
