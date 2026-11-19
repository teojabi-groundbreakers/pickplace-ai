"""Synthetic request generator checks; never publish fake values as real."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from scripts import make_unified_requests as maker


def test_examples_require_supported_complete_scope_and_preserve_no_overwrite(tmp_path, monkeypatch):
    seoul, gyeonggi = tmp_path / "seoul.csv", tmp_path / "gyeonggi.csv"
    seoul.write_text("not a real file")
    gyeonggi.write_text("not a real file")
    scopes = {"seoul": {("11110515", "CS100010")}, "gyeonggi": {("4111111111", "Q13")}}
    registry = SimpleNamespace(
        metadata={"source_hashes": {"seoul": "s", "gyeonggi": "g"}},
        source_known_by={"seoul": date(2026, 11, 4), "gyeonggi": date(2026, 11, 9)},
        created_at=date(2026, 11, 9),
        scopes=scopes,
    )
    monkeypatch.setattr(maker, "Registry", lambda _: registry)
    monkeypatch.setattr(maker.train_gyeonggi_research, "sha", lambda p: "s" if p == seoul else "g")
    periods = pd.period_range("2025Q3", "2026Q2", freq="Q")

    def frame(scope):
        return pd.DataFrame(
            [[1.0, 2.0, 3.0, 4.0]], index=pd.MultiIndex.from_tuples([scope]), columns=periods
        )

    monkeypatch.setattr(
        maker.train_unified_growth, "seoul_quarters", lambda _: frame(("11110515", "CS100010"))
    )
    monkeypatch.setattr(maker.pd, "read_csv", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        maker.train_gyeonggi_research, "make_quarters", lambda _: frame(("4111111111", "Q13"))
    )
    out = tmp_path / "examples"
    values = maker.make_requests(seoul, gyeonggi, tmp_path, out)
    assert values["seoul"]["history"][-1]["sales_amount"] == 4.0
    assert values["gyeonggi"]["data_available_by"] == "2026-11-09"
    assert (out / "seoul-request.json").exists() and (out / "gyeonggi-request.json").exists()
    with pytest.raises(FileExistsError):
        maker.make_requests(seoul, gyeonggi, tmp_path, out)

    calls = 0

    def changed(path):
        nonlocal calls
        calls += 1
        if calls == 3:
            return "changed"
        return "s" if path == seoul else "g"

    monkeypatch.setattr(maker.train_gyeonggi_research, "sha", changed)
    changed_out = tmp_path / "changed"
    with pytest.raises(ValueError, match="changed while reading"):
        maker.make_requests(seoul, gyeonggi, tmp_path, changed_out)
    assert not changed_out.exists()
