import csv
import json

import pytest

from pickplace_ai.nationwide_ingest import prepare_nationwide, sha256_file

STORE_FIELDS = [
    "상가업소번호",
    "행정동코드",
    "행정동명",
    "시도명",
    "시군구명",
    "상권업종소분류코드",
    "상권업종소분류명",
]
POP_FIELDS = ["행정구역", "2026년10월_계_총인구수"]


def write_csv(path, fields, rows, encoding="utf-8-sig"):
    with path.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def setup_data(tmp_path):
    stores = tmp_path / "stores.csv"
    population = tmp_path / "population.csv"
    write_csv(
        stores,
        STORE_FIELDS,
        [
            dict(
                zip(
                    STORE_FIELDS,
                    [
                        "one",
                        "11110515",
                        "청운효자동",
                        "서울",
                        "종로구",
                        "I21201",
                        "카페",
                    ],
                    strict=True,
                )
            ),
            dict(
                zip(
                    STORE_FIELDS,
                    [
                        "two",
                        "11110515",
                        "청운효자동",
                        "서울",
                        "종로구",
                        "I21201",
                        "카페",
                    ],
                    strict=True,
                )
            ),
        ],
    )
    write_csv(
        population,
        POP_FIELDS,
        [
            {"행정구역": "서울특별시 (1100000000)", POP_FIELDS[1]: "9,000"},
            {"행정구역": "서울특별시 종로구 (1111000000)", POP_FIELDS[1]: "5,000"},
            {
                "행정구역": "서울특별시 종로구 청운효자동(1111051500)",
                POP_FIELDS[1]: "1,500",
            },
            {"행정구역": "서울특별시 종로구 사직동(1111053000)", POP_FIELDS[1]: ""},
        ],
        encoding="cp949",
    )
    return stores, population


def run(stores, population, output):
    return prepare_nationwide(
        stores=stores,
        population=population,
        output_dir=output,
        store_reference_label="20260930",
        population_reference_month="2026-10",
    )


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_separate_original_codes_and_metadata(tmp_path):
    stores, population = setup_data(tmp_path)
    output = tmp_path / "out"
    summary = run([stores], population, output)
    a = read_csv(output / "stores.csv")
    b = read_csv(output / "population.csv")
    assert len(a) == 1 and a[0]["store_count"] == "2"
    assert a[0]["region_code"] == "11110515"
    assert [x["region_code"] for x in b] == ["1111051500", "1111053000"]
    assert [x["resident_population"] for x in b] == ["1500", ""]
    assert summary["crosswalk_verified"] is False
    assert summary["stores"]["observed_date"] is None
    assert summary["stores"]["available_at"] is None
    assert summary["population"]["observed_date"] == "2026-10-31"
    assert summary["stores"]["source_files"][0]["sha256"] == sha256_file(stores)
    assert summary["stores"]["output_sha256"] == sha256_file(output / "stores.csv")
    assert summary["population"]["output_sha256"] == sha256_file(output / "population.csv")
    assert json.loads((output / "manifest.json").read_text())["population"]["row_count"] == 2
    with pytest.raises(FileExistsError):
        run([stores], population, output)


def test_duplicate_files_and_store_identifiers_are_rejected(tmp_path):
    stores, population = setup_data(tmp_path)
    with pytest.raises(ValueError, match="중복"):
        run([stores, stores], population, tmp_path / "out")
    duplicate = tmp_path / "different.csv"
    write_csv(
        duplicate,
        STORE_FIELDS,
        [
            dict(
                zip(
                    STORE_FIELDS,
                    [
                        "one",
                        "11110515",
                        "청운효자동",
                        "서울",
                        "종로구",
                        "I21201",
                        "카페",
                    ],
                    strict=True,
                )
            )
        ],
    )
    with pytest.raises(ValueError, match="중복 상가업소번호"):
        run([stores, duplicate], population, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_code_and_invalid_population_are_not_treated_as_zero(tmp_path):
    stores, population = setup_data(tmp_path)
    write_csv(
        stores,
        STORE_FIELDS,
        [
            dict(
                zip(
                    STORE_FIELDS,
                    ["one", "", "청운효자동", "서울", "종로구", "I21201", "카페"],
                    strict=True,
                )
            )
        ],
    )
    with pytest.raises(ValueError, match="행정동코드"):
        run([stores], population, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_dates_and_snapshot_version_are_restricted(tmp_path):
    stores, population = setup_data(tmp_path)
    with pytest.raises(ValueError, match="20260930"):
        prepare_nationwide(
            stores=[stores],
            population=population,
            output_dir=tmp_path / "a",
            store_reference_label="20261399",
            population_reference_month="2026-10",
        )
    with pytest.raises(ValueError, match="YYYY-MM"):
        prepare_nationwide(
            stores=[stores],
            population=population,
            output_dir=tmp_path / "b",
            store_reference_label="20260930",
            population_reference_month="2026-13",
        )
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_changed_source_is_rejected_before_writing(tmp_path, monkeypatch):
    from pickplace_ai import nationwide_ingest

    stores, population = setup_data(tmp_path)
    original = nationwide_ingest._source
    calls = 0

    def changed(path):
        nonlocal calls
        result = original(path)
        if path == stores:
            calls += 1
            if calls > 1:
                result["sha256"] = "changed"
        return result

    monkeypatch.setattr(nationwide_ingest, "_source", changed)
    with pytest.raises(ValueError, match="처리 도중 변경"):
        run([stores], population, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_lowercase_industry_code_is_rejected(tmp_path):
    stores, population = setup_data(tmp_path)
    write_csv(
        stores,
        STORE_FIELDS,
        [
            dict(
                zip(
                    STORE_FIELDS,
                    [
                        "one",
                        "11110515",
                        "청운효자동",
                        "서울",
                        "종로구",
                        "i21201",
                        "카페",
                    ],
                    strict=True,
                )
            )
        ],
    )
    with pytest.raises(ValueError, match="업종 코드"):
        run([stores], population, tmp_path / "out")


def test_short_csv_row_is_rejected(tmp_path):
    stores, population = setup_data(tmp_path)
    population.write_text(
        "행정구역,2026년10월_계_총인구수,별도항목\n서울특별시 종로구 청운효자동(1111051500),100\n",
        encoding="cp949",
    )
    with pytest.raises(ValueError, match="열 수가 일치하지"):
        run([stores], population, tmp_path / "out")
    assert not (tmp_path / "out").exists()
