"""허구의 SQLite 자료로 점수 연구의 시점·결측·면적 경계를 검증한다."""

import sqlite3
from datetime import date

import pandas as pd
import pytest

from scripts import audit_seoul_scores as audit

ACQUIRED_AT = date(2026, 11, 9)
REGION = "11110515"
OTHER_REGION = "11110516"
INDUSTRY = "CS100008"


@pytest.fixture
def connection():
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(
        """
        CREATE TABLE sales (
            period TEXT, region_code TEXT, industry_code TEXT, sales_amount INTEGER,
            PRIMARY KEY(period,region_code,industry_code)
        );
        CREATE TABLE stores (
            period TEXT, region_code TEXT, industry_code TEXT,
            store_count INTEGER, opening_count INTEGER, closing_count INTEGER,
            PRIMARY KEY(period,region_code,industry_code)
        );
        CREATE TABLE resident (
            period TEXT, region_code TEXT, total INTEGER,
            PRIMARY KEY(period,region_code)
        );
        CREATE TABLE floating (
            period TEXT, region_code TEXT, total INTEGER,
            PRIMARY KEY(period,region_code)
        );
        """
    )
    try:
        yield db
    finally:
        db.close()


def add_store(db, period, *, stores=20, closings=0, region=REGION):
    db.execute(
        "INSERT INTO stores VALUES(?,?,?,?,?,?)",
        (period, region, INDUSTRY, stores, 0, closings),
    )


def add_sales(db, period, amount, *, region=REGION):
    db.execute("INSERT INTO sales VALUES(?,?,?,?)", (period, region, INDUSTRY, amount))


def add_population(db, period, total=10000, *, region=REGION):
    db.execute("INSERT INTO resident VALUES(?,?,?)", (period, region, total))


def example(db):
    add_sales(db, "2024Q1", 1000000)
    add_sales(db, "2024Q2", 900000)
    # 전년 동기 조회가 가장 가까운 관측값으로 바뀌지 않아야 한다.
    add_sales(db, "2024Q4", 1)
    add_store(db, "2025Q1", closings=3)
    add_sales(db, "2025Q1", 750000)
    add_population(db, "2025Q1")
    add_store(db, "2025Q2")
    add_sales(db, "2025Q2", 1125000)
    add_population(db, "2025Q2")


def score(db, periods=("2025Q1",), area=None, snapshot_period="2025Q2"):
    return audit.scored_rows(
        db,
        periods,
        {REGION: 2.0} if area is None else area,
        snapshot_period,
        acquired_at=ACQUIRED_AT,
    )


@pytest.mark.parametrize("encoding", ["cp949", "utf-8-sig"])
def test_area_snapshot_converts_declared_square_metres_to_km2(tmp_path, encoding):
    path = tmp_path / "fictional-area.csv"
    path.write_text(
        "행정동_코드,영역_면적\n11110515,2000000\n11110516,500000.5\n",
        encoding=encoding,
    )
    result = audit.read_area(path)
    assert result == {REGION: 2.0, OTHER_REGION: pytest.approx(0.5000005)}


def test_duplicate_area_codes_are_rejected_even_when_values_match(tmp_path):
    path = tmp_path / "duplicate.csv"
    path.write_text(
        "행정동_코드,영역_면적\n11110515,2000000\n11110515,2000000\n",
        encoding="utf-8-sig",
    )
    with pytest.raises(ValueError):
        audit.read_area(path)


@pytest.mark.parametrize("value", ["0", "-1", "NaN", "inf", ""])
def test_area_snapshot_requires_finite_positive_area(tmp_path, value):
    path = tmp_path / "invalid-area.csv"
    path.write_text(f"행정동_코드,영역_면적\n11110515,{value}\n", encoding="utf-8-sig")
    with pytest.raises(ValueError):
        audit.read_area(path)


@pytest.mark.parametrize("code", ["1111051500", "1111051", "21110515", "11110515.0", ""])
def test_area_codes_are_preserved_without_guessing_or_truncation(tmp_path, code):
    path = tmp_path / "invalid-code.csv"
    path.write_text(f"행정동_코드,영역_면적\n{code},2000000\n", encoding="utf-8-sig")
    with pytest.raises(ValueError):
        audit.read_area(path)


def test_scores_match_disclosed_components_and_exact_calendar_joins(connection):
    example(connection)
    rows = score(connection, periods=("2025Q1", "2025Q2"))
    current = rows.loc[rows.period == "2025Q1"].iloc[0]
    assert current.risk_score == 100.0
    assert current.sales_risk_points == 50.0
    assert current.closure_risk_points == 50.0
    assert current.sales_yoy == pytest.approx(-0.25)
    assert current.closure_ratio == pytest.approx(0.15)
    assert current.sales_per_store == pytest.approx(37500)
    assert current.next_sales_yoy == pytest.approx(0.25)
    assert current.next_closure_ratio == 0.0
    assert pd.isna(current.fit_score)
    latest = rows.loc[rows.period == "2025Q2"].iloc[0]
    assert latest.residents_per_store == 500.0
    assert latest.stores_per_km2 == 10.0
    assert latest.demand_points == 50.0
    assert latest.competition_points == 40.0
    assert latest.fit_score == 90.0


def test_future_outcomes_never_change_current_score(connection):
    example(connection)
    before = score(connection).iloc[0]
    connection.execute("UPDATE sales SET sales_amount=0 WHERE period='2025Q2'")
    connection.execute("UPDATE stores SET closing_count=20 WHERE period='2025Q2'")
    after = score(connection).iloc[0]
    for name in ("risk_score", "sales_risk_points", "closure_risk_points", "sales_yoy"):
        assert before[name] == after[name]
    assert before.next_sales_yoy == pytest.approx(0.25)
    assert after.next_sales_yoy == -1.0
    assert after.next_closure_ratio == 1.0


def test_latest_row_does_not_expose_future_beyond_snapshot(connection):
    example(connection)
    current = score(connection, snapshot_period="2025Q1").iloc[0]
    assert current.risk_score == 100.0
    assert current.fit_score == 90.0
    assert pd.isna(current.next_sales_yoy)
    assert pd.isna(current.next_closure_ratio)


def test_missing_next_quarter_is_not_replaced_by_next_available_row(connection):
    example(connection)
    connection.execute("DELETE FROM sales WHERE period='2025Q2'")
    connection.execute("DELETE FROM stores WHERE period='2025Q2'")
    add_store(connection, "2025Q3", closings=20)
    add_sales(connection, "2025Q3", 50000000)
    current = score(connection, snapshot_period="2025Q3").iloc[0]
    assert current.risk_score == 100.0
    assert pd.isna(current.next_sales_yoy)
    assert pd.isna(current.next_closure_ratio)


def test_missing_next_year_ago_sales_does_not_become_zero(connection):
    example(connection)
    connection.execute("DELETE FROM sales WHERE period='2024Q2'")
    current = score(connection).iloc[0]
    assert current.risk_score == 100.0
    assert pd.isna(current.next_sales_yoy)
    assert current.next_closure_ratio == 0.0


def test_missing_exact_previous_year_makes_only_risk_unavailable(connection):
    example(connection)
    connection.execute("DELETE FROM sales WHERE period='2024Q1'")
    current = score(connection, snapshot_period="2025Q1").iloc[0]
    assert pd.isna(current.risk_score)
    assert current.fit_score == 90.0


def test_observed_zero_sales_is_retained_as_a_valid_observation(connection):
    example(connection)
    connection.execute("UPDATE sales SET sales_amount=0 WHERE period='2025Q1'")
    connection.execute("UPDATE stores SET closing_count=0 WHERE period='2025Q1'")
    current = score(connection).iloc[0]
    assert bool(current.has_sales)
    assert current.sales_yoy == -1.0
    assert current.sales_risk_points == 50.0
    assert current.closure_risk_points == 0.0
    assert current.risk_score == 50.0
    assert current.sales_per_store == 0.0


def test_store_population_anchor_preserves_rows_without_sales(connection):
    add_store(connection, "2025Q1")
    add_population(connection, "2025Q1")
    current = score(connection, snapshot_period="2025Q1").iloc[0]
    assert not bool(current.has_sales)
    assert pd.isna(current.risk_score)
    assert current.fit_score == 90.0
    assert pd.isna(current.sales_per_store)


def test_area_is_not_backdated_into_earlier_population_rows(connection):
    for period in ("2024Q4", "2025Q1"):
        add_store(connection, period)
        add_population(connection, period)
    rows = score(connection, periods=("2024Q4", "2025Q1"), snapshot_period="2025Q1")
    assert pd.isna(rows.loc[rows.period == "2024Q4", "fit_score"].iloc[0])
    assert rows.loc[rows.period == "2025Q1", "fit_score"].iloc[0] == 90.0


def test_missing_current_population_is_not_forward_filled(connection):
    add_store(connection, "2025Q1")
    add_population(connection, "2024Q4")
    current = score(connection, snapshot_period="2025Q1").iloc[0]
    assert pd.isna(current.fit_score)
    assert pd.isna(current.residents_per_store)


def test_missing_area_is_not_filled_from_another_region(connection):
    add_store(connection, "2025Q1")
    add_population(connection, "2025Q1")
    current = score(
        connection,
        area={OTHER_REGION: 2.0},
        snapshot_period="2025Q1",
    ).iloc[0]
    assert pd.isna(current.fit_score)
    assert pd.isna(current.stores_per_km2)
