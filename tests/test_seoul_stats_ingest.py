import csv
import hashlib
import json
import sqlite3

import pytest

from pickplace_ai.ingest import ImportDataError
from pickplace_ai.seoul_stats_ingest import prepare_snapshot


def _write(path, fields, rows, encoding="utf-8-sig"):
    with path.open("w", encoding=encoding, newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    return path


@pytest.fixture
def sources(tmp_path):
    sales = _write(
        tmp_path / "sales.csv",
        ["기준_년분기_코드", "행정동_코드", "서비스_업종_코드", "당월_매출_금액"],
        [["20262", "11110515", "CS100001", "1500"], ["20261", "11110515", "CS100001", ""]],
        "cp949",
    )
    stores = _write(
        tmp_path / "stores.csv",
        [
            "STDR_YYQU_CD",
            "ADSTRD_CD",
            "SVC_INDUTY_CD",
            "SIMILR_INDUTY_STOR_CO",
            "OPBIZ_STOR_CO",
            "CLSBIZ_STOR_CO",
        ],
        [
            ["20262", "11110515", "CS100001", "3", "0", "1"],
            ["20262", "11110515", "CS200001", "5", "", "2"],
        ],
    )
    resident = _write(
        tmp_path / "resident.csv",
        ["기준_년분기_코드", "행정동_코드", "총_상주인구_수"],
        [["20262", "11110515", "200"]],
        "cp949",
    )
    floating = _write(
        tmp_path / "floating.csv",
        ["기준_년분기_코드", "행정동_코드", "총_유동인구_수"],
        [["20262", "11110515", "0"]],
        "cp949",
    )
    return [sales], [stores], resident, floating


def test_independent_tables_preserve_null_zero_and_store_only(sources, tmp_path):
    out = tmp_path / "snapshot"
    manifest = prepare_snapshot(*sources, out)
    with sqlite3.connect(out / "snapshot.sqlite3") as con:
        assert con.execute("select * from sales order by period").fetchall() == [
            ("2026Q1", "11110515", "CS100001", None),
            ("2026Q2", "11110515", "CS100001", 1500),
        ]
        assert con.execute("select * from stores order by industry_code").fetchall() == [
            ("2026Q2", "11110515", "CS100001", 3, 0, 1),
            ("2026Q2", "11110515", "CS200001", 5, None, 2),
        ]
        assert con.execute("select total from floating").fetchone() == (0,)
    assert manifest == json.loads((out / "manifest.json").read_text())
    assert manifest["datasets"]["sales"]["periods"] == ["2026Q1", "2026Q2"]
    assert manifest["datasets"]["stores"]["row_count"] == 2
    assert manifest["datasets"]["resident"]["publication_date_verified"] is False
    assert (
        manifest["database_sha256"]
        == hashlib.sha256((out / "snapshot.sqlite3").read_bytes()).hexdigest()
    )


def test_duplicate_key_fails_without_output(sources, tmp_path):
    sales = sources[0][0]
    with sales.open("a", encoding="cp949") as f:
        f.write("20262,11110515,CS100001,100\n")
    out = tmp_path / "snapshot"
    with pytest.raises(ImportDataError, match="duplicate"):
        prepare_snapshot(*sources, out)
    assert not out.exists()


def test_existing_output_not_overwritten(sources, tmp_path):
    out = tmp_path / "snapshot"
    out.mkdir()
    (out / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        prepare_snapshot(*sources, out)
    assert (out / "keep").read_text() == "keep"


@pytest.mark.parametrize("value", ["-1", "1.5", "NaN", "9999999999999999999999999"])
def test_invalid_values_rejected(value, sources, tmp_path):
    sales = sources[0][0]
    _write(
        sales,
        ["STDR_YYQU_CD", "ADSTRD_CD", "SVC_INDUTY_CD", "THSMON_SELNG_AMT"],
        [["20262", "11110515", "CS100001", value]],
    )
    with pytest.raises(ImportDataError):
        prepare_snapshot(*sources, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_missing_column_rejected(sources, tmp_path):
    _write(sources[2], ["STDR_YYQU_CD", "ADSTRD_CD"], [["20262", "11110515"]])
    with pytest.raises(ImportDataError):
        prepare_snapshot(*sources, tmp_path / "snapshot")


def test_inputs_must_have_distinct_paths(sources, tmp_path):
    with pytest.raises(ImportDataError, match="more than once"):
        prepare_snapshot(sources[0] * 2, *sources[1:], tmp_path / "snapshot")


def test_database_connection_is_closed_before_return(sources, tmp_path, monkeypatch):
    from pickplace_ai import seoul_stats_ingest

    original_connect = sqlite3.connect
    instances = []

    class TrackingConnection(sqlite3.Connection):
        is_closed = False

        def close(self):
            self.is_closed = True
            return super().close()

    def tracked_connect(*args, **kwargs):
        connection = original_connect(*args, factory=TrackingConnection, **kwargs)
        instances.append(connection)
        return connection

    monkeypatch.setattr(seoul_stats_ingest.sqlite3, "connect", tracked_connect)
    prepare_snapshot(*sources, tmp_path / "snapshot")
    assert len(instances) == 1
    assert instances[0].is_closed


def test_first_source_change_during_later_files_is_detected(sources, tmp_path, monkeypatch):
    from pickplace_ai import seoul_stats_ingest

    original = seoul_stats_ingest._digest
    sales = sources[0][0]
    calls = 0

    def changed_digest(path):
        nonlocal calls
        if path == sales:
            calls += 1
            if calls == 5:
                with path.open("ab") as stream:
                    stream.write(b"\n")
        return original(path)

    monkeypatch.setattr(seoul_stats_ingest, "_digest", changed_digest)
    with pytest.raises(ImportDataError, match="source changed"):
        prepare_snapshot(*sources, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
