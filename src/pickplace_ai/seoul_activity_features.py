"""Source-preserving activity covariates for Seoul retrospective research.

No legal land-use designation or customer count is inferred here. The input
time-band values remain the official published aggregate, not people/hour.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from pickplace_ai import features
from pickplace_ai.ingest import _period, _read_csv

TIME_COLUMNS = (
    "시간대_00_06_유동인구_수",
    "시간대_06_11_유동인구_수",
    "시간대_11_14_유동인구_수",
    "시간대_14_17_유동인구_수",
    "시간대_17_21_유동인구_수",
    "시간대_21_24_유동인구_수",
)
POPULATION_FEATURES = (
    "resident_log1p",
    "floating_log1p",
    "day_overnight_hourly_log_ratio",
    "evening_day_hourly_log_ratio",
)
STORE_FEATURES = ("store_log1p",)
SALES_FEATURES = features.GYEONGGI_RESEARCH_FEATURES
FEATURE_SETS = {
    "sales_only": SALES_FEATURES,
    "sales_population": (*SALES_FEATURES, *POPULATION_FEATURES),
    "sales_stores": (*SALES_FEATURES, *STORE_FEATURES),
    "sales_population_stores": (*SALES_FEATURES, *POPULATION_FEATURES, *STORE_FEATURES),
}
SCOPE = ("행정동코드", "중분류업종코드")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def activity_covariates(
    resident: np.ndarray,
    floating: np.ndarray,
    stores: np.ndarray,
    bands: np.ndarray,
) -> pd.DataFrame:
    """Use duration-adjusted relative time patterns; never divide floating by residents."""
    r = np.asarray(resident, dtype=float)
    f = np.asarray(floating, dtype=float)
    s = np.asarray(stores, dtype=float)
    b = np.asarray(bands, dtype=float)
    if r.ndim != 1 or f.shape != r.shape or s.shape != r.shape or b.shape != (len(r), 6):
        raise ValueError("Invalid Seoul activity feature shape")
    if not np.isfinite(r).all() or not np.isfinite(f).all() or not np.isfinite(s).all():
        raise ValueError("Nonfinite Seoul population or store count")
    if not np.isfinite(b).all() or (r < 0).any() or (f < 0).any() or (s < 0).any():
        raise ValueError("Invalid Seoul population or store count")
    if (b < 0).any() or (np.abs(b.sum(axis=1) - f) > 6).any():
        raise ValueError("Time bands do not reconcile to the source total")
    # The six time windows have different lengths. These are relative indices,
    # not a validated daily average or a count of customers.
    daytime_per_hour = (b[:, 2] + b[:, 3]) / 6.0
    overnight_per_hour = (b[:, 5] + b[:, 0]) / 9.0
    evening_per_hour = b[:, 4] / 4.0
    return pd.DataFrame(
        {
            "resident_log1p": np.log1p(r),
            "floating_log1p": np.log1p(f),
            "day_overnight_hourly_log_ratio": np.log1p(daytime_per_hour)
            - np.log1p(overnight_per_hour),
            "evening_day_hourly_log_ratio": np.log1p(evening_per_hour) - np.log1p(daytime_per_hour),
            "store_log1p": np.log1p(s),
        }
    )


def load_time_bands(path: Path) -> pd.DataFrame:
    """Read exact eight-digit Seoul codes and quarterly time-band rows."""
    rows = []
    with _read_csv(Path(path)) as (headers, source_rows):
        required = {"기준_년분기_코드", "행정동_코드", "총_유동인구_수", *TIME_COLUMNS}
        if not required.issubset(headers):
            raise ValueError("Missing required Seoul time-band columns")
        for line, row in source_rows:
            period = _period(row["기준_년분기_코드"])
            code = row["행정동_코드"]
            if not re.fullmatch(r"11\d{6}", code):
                raise ValueError(f"Invalid Seoul region code at line {line}")
            try:
                values = [int(row[name]) for name in ("총_유동인구_수", *TIME_COLUMNS)]
            except ValueError as exc:
                raise ValueError(f"Invalid Seoul time-band value at line {line}") from exc
            if min(values) < 0 or abs(sum(values[1:]) - values[0]) > 6:
                raise ValueError(f"Invalid Seoul time-band total at line {line}")
            rows.append((period, code, *values))
    if not rows:
        raise ValueError("Empty Seoul time-band source")
    frame = pd.DataFrame(rows, columns=["period", "region_code", "floating_total", *TIME_COLUMNS])
    if frame.duplicated(["period", "region_code"]).any():
        raise ValueError("Duplicate Seoul time-band key")
    return frame


def load_snapshot(
    snapshot_dir: Path, floating_csv: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Verify source fingerprints and exact original-key joins before learning."""
    directory = Path(snapshot_dir)
    database = directory / "snapshot.sqlite3"
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("mode") != "seoul_descriptive_research_only":
        raise ValueError("Unexpected Seoul snapshot mode")
    if manifest.get("database_sha256") != sha256(database):
        raise ValueError("Seoul snapshot checksum mismatch")
    floating_sources = manifest.get("datasets", {}).get("floating", {}).get("source_files", [])
    if len(floating_sources) != 1 or floating_sources[0].get("sha256") != sha256(floating_csv):
        raise ValueError("Seoul floating CSV checksum mismatch")
    query = """
        SELECT s.period, s.region_code, s.industry_code, s.sales_amount,
               st.store_count, r.total AS resident_total, f.total AS floating_total
        FROM sales AS s
        INNER JOIN stores AS st
          ON st.period=s.period AND st.region_code=s.region_code
         AND st.industry_code=s.industry_code
        INNER JOIN resident AS r ON r.period=s.period AND r.region_code=s.region_code
        INNER JOIN floating AS f ON f.period=s.period AND f.region_code=s.region_code
        ORDER BY s.period, s.region_code, s.industry_code
    """
    with closing(sqlite3.connect(f"file:{database}?mode=ro", uri=True)) as connection:
        joined = pd.read_sql_query(query, connection)
        sales_count = connection.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    if sales_count != manifest["datasets"]["sales"]["row_count"] or len(joined) != sales_count:
        raise ValueError("Unmatched or unexpected Seoul sales rows")
    if (
        joined.isna().any().any()
        or joined.duplicated(["period", "region_code", "industry_code"]).any()
    ):
        raise ValueError("Missing or duplicate Seoul source values")
    if not joined.region_code.str.fullmatch(r"11\d{6}").all():
        raise ValueError("Invalid Seoul source region code")
    if not joined.industry_code.str.fullmatch(r"CS\d{6}").all():
        raise ValueError("Invalid Seoul source industry code")
    numeric = joined[["sales_amount", "store_count", "resident_total", "floating_total"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all() or (numeric < 0).any().any():
        raise ValueError("Invalid Seoul source numbers")
    bands = load_time_bands(floating_csv)
    joined = joined.merge(
        bands,
        on=["period", "region_code"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_raw"),
        indicator=True,
    )
    if not joined._merge.eq("both").all():
        raise ValueError("Seoul time-band row missing")
    if not joined.floating_total.eq(joined.floating_total_raw).all():
        raise ValueError("Seoul time-band total does not match snapshot")
    joined = joined.drop(columns=["floating_total_raw", "_merge"])
    if (
        sha256(database) != manifest["database_sha256"]
        or sha256(floating_csv) != floating_sources[0]["sha256"]
    ):
        raise ValueError("Seoul source changed during read")
    quarters = joined.pivot(
        index=["region_code", "industry_code"], columns="period", values="sales_amount"
    )
    quarters.columns = pd.PeriodIndex(quarters.columns, freq="Q")
    quarters.index.names = list(SCOPE)
    return quarters, joined, manifest
