"""Create local test requests using trusted research artifacts and real snapshots."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pickplace_ai.unified_api import SYSTEMS, UNIT, Registry
from scripts import train_gyeonggi_research, train_unified_growth


def make_requests(seoul: Path, gyeonggi: Path, model_root: Path, output_dir: Path) -> dict:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError("Never overwrite existing examples")
    registry = Registry(model_root)
    sources = {"seoul": seoul, "gyeonggi": gyeonggi}
    for source, path in sources.items():
        if train_gyeonggi_research.sha(path) != registry.metadata["source_hashes"][source]:
            raise ValueError(f"Source snapshot differs from model metadata: {source}")
    quarters = {
        "seoul": train_unified_growth.seoul_quarters(seoul),
        "gyeonggi": train_gyeonggi_research.make_quarters(
            pd.read_csv(
                gyeonggi,
                dtype={"기준년월": str, "행정동코드": str, "중분류업종코드": str},
            )
        ),
    }
    periods = pd.period_range("2025Q3", "2026Q2", freq="Q")
    as_of = max(registry.created_at, *registry.source_known_by.values())
    results = {}
    for source, data in quarters.items():
        if not set(periods).issubset(data.columns):
            raise ValueError(f"Missing quarterly input: {source}")
        candidates = data[list(periods)].dropna().sort_index()
        for (region, industry), row in candidates.iterrows():
            values = row.to_numpy(dtype=float)
            if (region, industry) not in registry.scopes[source] or not np.isfinite(values).all():
                continue
            if (values[:3] <= 0).any() or values[3] < 0:
                continue
            results[source] = {
                "source": source,
                "region_code_system": SYSTEMS[source],
                "region_code": region,
                "industry_code": industry,
                "sales_unit": UNIT[source],
                "base_period": "2026Q2",
                "target_period": "2026Q3",
                "as_of": as_of.isoformat(),
                "data_available_by": registry.source_known_by[source].isoformat(),
                "history": [
                    {"period": str(period), "sales_amount": float(value)}
                    for period, value in zip(periods, values, strict=True)
                ],
            }
            break
        if source not in results:
            raise ValueError(f"No supported scope with complete input: {source}")
    for source, path in sources.items():
        if train_gyeonggi_research.sha(path) != registry.metadata["source_hashes"][source]:
            raise ValueError(f"Source snapshot changed while reading: {source}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".unified-examples-", dir=output_dir.parent) as temp:
        staged = Path(temp) / "output"
        staged.mkdir()
        for source, request in results.items():
            (staged / f"{source}-request.json").write_text(
                json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        (staged / "README.txt").write_text(
            "Real aggregate data; local research/testing only. Not synthetic. Do not publish.\n"
            "Source values, scopes, dates, and geography remain subject to original caveats.\n",
            encoding="utf-8",
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError("Never overwrite existing examples")
        os.rename(staged, output_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seoul", type=Path, required=True)
    parser.add_argument("--gyeonggi", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    requests = make_requests(args.seoul, args.gyeonggi, args.model_root, args.output_dir)
    print(json.dumps({key: value["region_code"] for key, value in requests.items()}))


if __name__ == "__main__":
    main()
