"""Explicit synthetic fixtures: fictional codes, deterministic data, never real evidence."""

from datetime import timedelta
from math import sin

import numpy as np

from pickplace_ai.periods import quarter_end, shift_quarter
from pickplace_ai.schemas import Observation, PredictionRequest


def synthetic_observations(
    regions: int = 32,
    quarters: int = 24,
    seed: int = 42,
) -> list[Observation]:
    if not 1 <= regions <= 1000 or not 10 <= quarters <= 80:
        raise ValueError("Use 1..1000 regions and 10..80 quarters for synthetic fixtures")
    rng = np.random.default_rng(seed)
    result = []
    for index in range(regions):
        level = 50_000_000 + index * 5_000_000
        annual_growth = float(rng.uniform(-0.15, 0.20))
        for step in range(quarters):
            period = shift_quarter("2020Q1", step)
            stores = 15 + index % 20 + step // 8
            residents = max(1000, 10000 + index * 600 + int(step * annual_growth * 300))
            seasonal = 1 + 0.08 * sin((step % 4) * np.pi / 2)
            sales = level * ((1 + annual_growth) ** (step / 4)) * seasonal
            sales *= float(rng.uniform(0.985, 1.015))
            result.append(
                Observation(
                    region_code=f"{99000000 + index:08d}",
                    region_code_system="synthetic-only-v1",
                    industry_code="DEMO_COFFEE",
                    period=period,
                    available_at=quarter_end(period) + timedelta(days=14),
                    source_kind="synthetic",
                    source_dataset=f"generated-seed-{seed}",
                    sales_amount=round(sales, 2),
                    store_count=stores,
                    opening_count=1 + (index + step) % 3,
                    closing_count=(index + step) % 3,
                    resident_population=float(residents),
                    working_population=float(residents * 0.6),
                    floating_population=float(residents * 2.5),
                    area_km2=1.2 + index * 0.05,
                )
            )
    return result


def example_request(rows: list[Observation]) -> PredictionRequest:
    first = rows[0]
    history = [
        row
        for row in rows
        if (row.region_code_system, row.region_code, row.industry_code)
        == (first.region_code_system, first.region_code, first.industry_code)
    ]
    current = max(history, key=lambda r: r.period)
    return PredictionRequest(
        region_code=first.region_code,
        region_code_system=first.region_code_system,
        industry_code=first.industry_code,
        base_period=current.period,
        as_of=current.available_at,
        source_kind="synthetic",
        history=history,
    )
