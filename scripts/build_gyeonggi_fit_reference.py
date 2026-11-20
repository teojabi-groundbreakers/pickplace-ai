"""같은 기준일의 경기 행정동 경계·주민등록인구·영업 업소로 비교표를 만든다."""

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from pickplace_ai.artifacts import file_sha256
from pickplace_ai.fit_reference import (
    GYEONGGI_CODE_SYSTEM,
    GYEONGGI_KIND,
    GYEONGGI_STORE_DEFINITION,
    ROW_FIELDS,
    FitReference,
)
from scripts.gyeonggi_fit_geometry import inspect_boundaries, locate_stores
from scripts.gyeonggi_fit_population import read_mois_gyeonggi_population

QUARTER = "2026Q2"
REFERENCE_DATE = date(2026, 6, 30)
# 인구 원천은 화도읍 본체와 두 출장소를 따로 집계하지만 도형은 화도읍 한 개다.
UNVERIFIED_BRANCH_SCOPE = "4136025600"
# 7월 1일 명칭 변경이 반영된 원천 업소는 다른 동으로 재배정하지 않는다.
POST_QUARTER_STORE_CODES = frozenset({"41171582", "41171583"})
# 같은 원본 행정동의 점포가 여러 도형으로 크게 나뉘면 관련 도형 전체를 보류한다.
MIN_STORE_GROUP_CONCENTRATION = 0.95


def _verify_inputs(paths, sources):
    if set(paths) != set(sources):
        raise ValueError("경계·인구·점포의 원천 증빙이 모두 필요합니다.")
    for name, path in paths.items():
        source = sources[name]
        url = urlsplit(source["url"])
        if (
            date.fromisoformat(source["reference_date"]) != REFERENCE_DATE
            or date.fromisoformat(source["acquired_at"]) < REFERENCE_DATE
            or url.scheme != "https"
            or not (url.hostname or "").endswith(".go.kr")
            or url.username is not None
            or file_sha256(path) != source["sha256"]
        ):
            raise ValueError(f"{name} 원천의 기준일 또는 지문이 일치하지 않습니다.")


def build(
    archive: Path, population_path: Path, stores_path: Path, evidence_path: Path, output: Path
) -> dict:
    if output.exists():
        raise FileExistsError("경기 비교표 출력 경로가 이미 존재합니다.")
    content = evidence_path.read_bytes()
    sources = json.loads(content)["sources"]
    paths = {"area": archive, "population": population_path, "stores": stores_path}
    _verify_inputs(paths, sources)

    boundaries, boundary_audit = inspect_boundaries(archive)
    population = read_mois_gyeonggi_population(population_path, REFERENCE_DATE)
    shape_codes, population_codes = set(boundaries), set(population)
    common = shape_codes & population_codes
    invalid_geometry = {
        code
        for code in common
        if not boundaries[code].is_valid or boundaries[code].is_empty or boundaries[code].area <= 0
    }
    eligible = common - invalid_geometry - {UNVERIFIED_BRANCH_SCOPE}
    counts, spatial_audit = locate_stores(
        stores_path, boundaries, eligible, excluded_source_codes=POST_QUARTER_STORE_CODES
    )
    split_source_codes = {
        group["source_code"]
        for group in spatial_audit["source_code_groups"]
        if group["dominant_share"] < MIN_STORE_GROUP_CONCENTRATION
    }
    unresolved_scopes = eligible & {
        pair["geometry_h10"]
        for pair in spatial_audit["source_to_geometry_pairs"]
        if pair["source_code"] in split_source_codes
    }
    incomplete_neighbor_scopes = eligible & spatial_audit["invalid_bbox_valid_hit_counts"].keys()
    withheld_scopes = unresolved_scopes | incomplete_neighbor_scopes
    eligible -= withheld_scopes
    rows = [
        (code, industry, count, population[code], boundaries[code].area)
        for (code, industry), count in sorted(counts.items())
        if code in eligible
    ]
    _verify_inputs(paths, sources)
    if evidence_path.read_bytes() != content:
        raise ValueError("처리 중 원천 증빙이 바뀌었습니다.")

    audit = {
        "shape_codes": len(shape_codes),
        "population_codes": len(population_codes),
        "common_codes": len(common),
        "shape_only": sorted(shape_codes - population_codes),
        "population_only": sorted(population_codes - shape_codes),
        "invalid_geometry": sorted(invalid_geometry),
        "branch_scope_excluded": sorted(common & {UNVERIFIED_BRANCH_SCOPE}),
        "store_group_concentration_threshold": MIN_STORE_GROUP_CONCENTRATION,
        "split_source_codes": sorted(split_source_codes),
        "unresolved_store_scopes": sorted(unresolved_scopes),
        "incomplete_neighbor_scopes": sorted(incomplete_neighbor_scopes),
        "stores_in_reference": sum(row[2] for row in rows),
        "stores_withheld_in_unresolved_scopes": sum(
            count for (code, _industry), count in counts.items() if code in withheld_scopes
        ),
        "eligible_boundary_codes": len(eligible),
        "boundary_geometry": boundary_audit,
        "spatial_stores": spatial_audit,
    }
    metadata = {
        "kind": GYEONGGI_KIND,
        "source_kind": "official",
        "region_code_system": GYEONGGI_CODE_SYSTEM,
        "store_count_definition": GYEONGGI_STORE_DEFINITION,
        "base_period": QUARTER,
        "stores_period": QUARTER,
        "reference_date": REFERENCE_DATE.isoformat(),
        "available_at": max(
            date.fromisoformat(source["acquired_at"]) for source in sources.values()
        ).isoformat(),
        **{f"{key}_reference_date": REFERENCE_DATE.isoformat() for key in (*sources, "mapping")},
        "sources": sources.copy(),
        "row_count": len(rows),
        "cohort_sizes": dict(sorted(Counter(row[1] for row in rows).items())),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory)
        reference = stage / "reference.csv"
        with reference.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(ROW_FIELDS)
            writer.writerows(rows)
        audit_path = stage / "source-audit.json"
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        metadata["sources"]["mapping"] = {
            "url": sources["area"]["url"],
            "sha256": file_sha256(audit_path),
            "method": "gemd_mois_exact_h10_and_store_point_in_polygon",
        }
        metadata["reference_sha256"] = file_sha256(reference)
        (stage / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        FitReference(stage, GYEONGGI_KIND)
        output.mkdir()
        published = []
        try:
            for name in ("source-audit.json", "reference.csv", "manifest.json"):
                target = output / name
                os.link(stage / name, target)
                published.append(target)
        except BaseException:
            for target in published:
                target.unlink(missing_ok=True)
            output.rmdir()
            raise
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "population", "stores", "evidence", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    metadata = build(args.archive, args.population, args.stores, args.evidence, args.output)
    print(f"{metadata['base_period']} 경기 비교표: {metadata['row_count']:,}행")


if __name__ == "__main__":
    main()
