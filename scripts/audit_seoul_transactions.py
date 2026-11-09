"""Describe original Seoul sales fields for the largest existing validation errors.

This is retrospective observation only; the raw provider calls the fields
"monthly" despite the quarter key. Never scale them or infer an error cause.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pickplace_ai import seoul_guard_policy as guard
from scripts.train_gyeonggi_research import sha

SOURCE_MAP = {
    "STDR_YYQU_CD": "기준_년분기_코드",
    "ADSTRD_CD": "행정동_코드",
    "SVC_INDUTY_CD": "서비스_업종_코드",
    "THSMON_SELNG_AMT": "당월_매출_금액",
    "THSMON_SELNG_CO": "당월_매출_건수",
    "MDWK_SELNG_AMT": "주중_매출_금액",
    "WKEND_SELNG_AMT": "주말_매출_금액",
    "TMZON_00_06_SELNG_AMT": "시간대_00~06_매출_금액",
    "TMZON_06_11_SELNG_AMT": "시간대_06~11_매출_금액",
    "TMZON_11_14_SELNG_AMT": "시간대_11~14_매출_금액",
    "TMZON_14_17_SELNG_AMT": "시간대_14~17_매출_금액",
    "TMZON_17_21_SELNG_AMT": "시간대_17~21_매출_금액",
    "TMZON_21_24_SELNG_AMT": "시간대_21~24_매출_금액",
}
KEYS = ["행정동_코드", "서비스_업종_코드"]
FIELD_PERIOD = "기준_년분기_코드"
AMOUNT = "당월_매출_금액"
COUNT = "당월_매출_건수"
TIME = [v for k, v in SOURCE_MAP.items() if k.startswith("TMZON_")]
FIELDS = list(SOURCE_MAP.values())


def positive_ratio(numerator: object, denominator: object) -> float | None:
    if pd.isna(numerator) or pd.isna(denominator) or float(denominator) <= 0:
        return None
    value = float(numerator) / float(denominator)
    return value if np.isfinite(value) else None


def read_source(path: Path, *, api: bool, selected: pd.DataFrame) -> pd.DataFrame:
    cols = list(SOURCE_MAP) if api else FIELDS
    identifier_columns = list(SOURCE_MAP)[:3] if api else FIELDS[:3]
    data = pd.read_csv(
        path,
        encoding="utf-8-sig" if api else "cp949",
        usecols=cols,
        dtype={col: str for col in identifier_columns},
    )
    if api:
        data = data.rename(columns=SOURCE_MAP)
    data = data.merge(selected, on=KEYS, how="inner", validate="many_to_one")
    if data.duplicated([*KEYS, FIELD_PERIOD]).any():
        raise ValueError("Duplicate raw region, industry and quarter")
    return data


def record(row: pd.Series) -> dict:
    amount, count = row[AMOUNT], row[COUNT]
    times = pd.to_numeric(row[TIME], errors="coerce")
    if times.isna().any() or pd.isna(amount) or float(amount) <= 0:
        top_time, top_share, time_total_ratio = None, None, None
    else:
        top_time = str(times.idxmax())
        top_share = positive_ratio(float(times.max()), amount)
        time_total_ratio = positive_ratio(float(times.sum()), amount)
    return {
        "period_source": str(row[FIELD_PERIOD]),
        "region_code": str(row[KEYS[0]]),
        "industry_code": str(row[KEYS[1]]),
        "region_name_2025": str(row["region_name_2025_reference"]),
        "industry_name_2025": str(row["industry_name_2025_reference"]),
        "amount_raw": None if pd.isna(amount) else float(amount),
        "count_raw": None if pd.isna(count) else float(count),
        "amount_divided_by_count": positive_ratio(amount, count),
        "weekday_amount_fraction": positive_ratio(row["주중_매출_금액"], amount),
        "time_amount_sum_divided_by_total": time_total_ratio,
        "top_time_bucket": top_time,
        "top_time_amount_fraction": top_share,
    }


def guard_diagnostics(values: np.ndarray, error: np.ndarray) -> dict:
    """Describe evaluation rows, with overlapping signals kept separate."""
    if values.shape != (len(error), 5) or not np.isfinite(values).all():
        raise ValueError("Expected five complete finite source periods")
    if (values[:, :3] <= 0).any() or not np.isfinite(error).all():
        raise ValueError("Invalid denominator or error")
    history = values[:, :4]
    growth = history[:, 1:] / history[:, :-1] - 1
    signals = {
        "reference_low": history[:, 0] < guard.LOW_REFERENCE_SALES_KRW,
        "latest_low": history[:, 3] < guard.LOW_LATEST_SALES_KRW,
        "prior_surge": growth.max(axis=1) > guard.MAX_QOQ_INCREASE,
        "prior_drop": growth.min(axis=1) < guard.MIN_QOQ_DECREASE,
    }
    flags = np.stack(list(signals.values()), axis=1)
    total = float(error.sum())
    by_signal = {}
    for name, mask in signals.items():
        by_signal[name] = {
            "rows": int(mask.sum()),
            "overlap_rows": int((mask & (flags.sum(axis=1) > 1)).sum()),
            "error_share_pct": float(error[mask].sum() / total * 100) if total else None,
        }
    high_error_count = max(1, int(np.ceil(len(error) * 0.01)))
    high_error = np.argsort(error)[-high_error_count:]
    surge = values[:, 1:4] / values[:, :3] > 1 + guard.MAX_QOQ_INCREASE
    # The last comparison uses the validation target. Retrospective description only.
    subsequent_ratio = values[:, 2:5] / values[:, :3]
    returned = (surge & (subsequent_ratio >= 0.75) & (subsequent_ratio <= 1.25)).any(axis=1)
    return {
        "rows": len(error),
        "flagged_rows": int(flags.any(axis=1).sum()),
        "flagged_error_share_pct": float(error[flags.any(axis=1)].sum() / total * 100)
        if total
        else None,
        "top_1pct_error_rows": high_error_count,
        "top_1pct_errors_flagged": int(flags.any(axis=1)[high_error].sum()),
        "overlapping_signals": by_signal,
        "prior_surge_rows": int(surge.any(axis=1).sum()),
        "return_after_surge_rows": int(returned.sum()),
        "return_definition": "Subsequent source value within 75%-125% of pre-surge value; "
        "may use target; retrospective only and not seasonality adjusted.",
    }


def run(
    *,
    validation: Path,
    summary: Path,
    y2024: Path,
    y2025: Path,
    q1: Path,
    q2: Path,
    output: Path,
) -> dict:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a new output directory")
    sources = {
        "validation": validation,
        "validation_summary": summary,
        "sales_2024": y2024,
        "sales_2025": y2025,
        "api_2026q1": q1,
        "api_2026q2": q2,
    }
    hashes = {name: sha(path) for name, path in sources.items()}
    prior = json.loads(summary.read_text(encoding="utf-8"))
    if prior.get("output_sha256", {}).get(validation.name) != hashes["validation"]:
        raise ValueError("Validation row file checksum mismatch")
    rows = pd.read_csv(validation, dtype={"행정동코드": str, "중분류업종코드": str})
    if rows.duplicated(["행정동코드", "중분류업종코드"]).any():
        raise ValueError("Duplicate validation scope")
    all_scopes = rows[
        [
            "행정동코드",
            "중분류업종코드",
            "region_name_2025_reference",
            "industry_name_2025_reference",
        ]
    ].rename(columns={"행정동코드": KEYS[0], "중분류업종코드": KEYS[1]})
    all_source = pd.concat(
        [
            read_source(y2024, api=False, selected=all_scopes),
            read_source(y2025, api=False, selected=all_scopes),
        ],
        ignore_index=True,
    )
    source_periods = ["20244", "20251", "20252", "20253", "20254"]
    source_pivot = all_source.pivot(index=KEYS, columns=FIELD_PERIOD, values=AMOUNT)
    source_index = pd.MultiIndex.from_frame(all_scopes[KEYS])
    source_values = source_pivot.loc[source_index, source_periods].to_numpy(dtype=float)
    if not np.array_equal(source_values[:, 0], rows["comparison_sales_krw"].to_numpy(float)):
        raise ValueError("Validation reference and raw source disagree")
    if not np.array_equal(source_values[:, 4], rows["target_sales_krw"].to_numpy(float)):
        raise ValueError("Validation target and raw source disagree")
    diagnostics = guard_diagnostics(source_values, rows["absolute_error"].to_numpy(float))
    selected = rows.nlargest(12, "absolute_error")[
        [
            "행정동코드",
            "중분류업종코드",
            "region_name_2025_reference",
            "industry_name_2025_reference",
        ]
    ].rename(columns={"행정동코드": KEYS[0], "중분류업종코드": KEYS[1]})
    if len(selected) < 12 or selected.isna().any().any():
        raise ValueError("Incomplete validation scopes or source names")
    frames = [
        read_source(y2024, api=False, selected=selected),
        read_source(y2025, api=False, selected=selected),
        read_source(q1, api=True, selected=selected),
        read_source(q2, api=True, selected=selected),
    ]
    combined = pd.concat(frames, ignore_index=True)
    if combined.duplicated([*KEYS, FIELD_PERIOD]).any():
        raise ValueError("Duplicate scope-period across input files")
    combined = combined[(combined[FIELD_PERIOD] >= "20242") & (combined[FIELD_PERIOD] <= "20262")]
    records = [record(row) for _, row in combined.sort_values([*KEYS, FIELD_PERIOD]).iterrows()]
    result = {
        "purpose": "retrospective_descriptive_not_cause_detection",
        "source_sha256": hashes,
        "script_sha256": sha(Path(__file__)),
        "guard_policy_sha256": sha(Path(guard.__file__)),
        "guard_diagnostics_validation_2025q4": diagnostics,
        "selection": "12 largest absolute errors in already inspected 2025Q4 validation",
        "data_field_semantics": (
            "Provider calls values monthly although quarter key is present; "
            "source period and average-price meaning unverified. "
            "Values not multiplied or assumed to be quarterly totals."
        ),
        "official_sources": {
            "method_description": "https://golmok.seoul.go.kr/introduce.do",
            "sales_dataset_and_spatial_change_notice": "https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do",
        },
        "warnings": [
            "Estimated card-based sales, not individual business actuals.",
            "Other breakouts are derived in the same source, not independent validation.",
            "Ratio of source amount/count is descriptive; field interpretation unverified.",
            "No change cause, data error, region risk or verified change-point is inferred.",
            "A later-known revised snapshot is not a point-in-time backtest.",
        ],
        "rows": len(records),
    }
    if hashes != {name: sha(path) for name, path in sources.items()}:
        raise ValueError("Input changed during read")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".seoul-transaction-audit-", dir=output.parent) as tmp:
        dest = Path(tmp) / "result"
        dest.mkdir()
        csv = dest / "selected_timelines.csv"
        pd.DataFrame.from_records(records).to_csv(csv, index=False)
        result["output_sha256"] = {csv.name: sha(csv)}
        (dest / "summary.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )
        if output.exists() or output.is_symlink():
            raise FileExistsError("Choose a new output directory")
        os.rename(dest, output)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("validation", "summary", "y2024", "y2025", "q1", "q2", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = run(
        validation=args.validation,
        summary=args.summary,
        y2024=args.y2024,
        y2025=args.y2025,
        q1=args.q1,
        q2=args.q2,
        output=args.output,
    )
    print(json.dumps({"purpose": result["purpose"], "rows": result["rows"]}))


if __name__ == "__main__":
    main()
