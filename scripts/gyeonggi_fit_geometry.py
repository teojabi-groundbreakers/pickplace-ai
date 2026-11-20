"""경기 원본 GEMD 전수 조회와 SBIZ 좌표 기반 집계용 오프라인 보조 함수.

GEMD의 좌표계는 제공기관 공식 답변으로 확인한 EPSG:5179이다. ZIP에 원본
PRJ가 있었다고 주장하지 않는다. SBIZ 8자리 코드는 변환하지 않고 감사에만 쓴다.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from pickplace_ai.artifacts import file_sha256 as _sha256


def _gis():
    try:
        import pyproj
        import shapefile
        import shapely
        from shapely.geometry import shape
        from shapely.strtree import STRtree
    except ImportError as error:
        raise RuntimeError(
            "Offline geometry processing requires pyshp, pyproj and Shapely"
        ) from error
    return pyproj, shapefile, shapely, shape, STRtree


CRS_BASIS_URL = (
    "https://business.juso.go.kr/jsi/jsiInquiryDetails?qaPstUnqMgno=114351&qaPstRfrncMgno=114351"
)
DEFAULT_COLUMNS = {
    "store_id": "상가업소번호",
    "source_code": "행정동코드",
    "industry": "상권업종소분류코드",
    "lon": "경도",
    "lat": "위도",
}


def inspect_boundaries(archive: Path) -> tuple[dict, dict]:
    """602개 등의 수를 고정하지 않고 모든 행과 삭제 플래그를 검사한다."""
    pyproj, shapefile, shapely, shape, _tree = _gis()
    before = _sha256(archive)
    with zipfile.ZipFile(archive) as zipped:
        members = {}
        for name in zipped.namelist():
            base = Path(name).name
            if base.startswith("TL_SCCO_GEMD."):
                suffix = Path(base).suffix.lower()
                if suffix in members:
                    raise ValueError("Ambiguous GEMD archive components")
                members[suffix] = name
        if any(ext not in members for ext in (".shp", ".dbf", ".shx")):
            raise ValueError("SHP/DBF/SHX components required")
        data = {ext: zipped.read(name) for ext, name in members.items()}
    if ".prj" in data:
        crs = pyproj.CRS.from_wkt(data[".prj"].decode("utf-8-sig"))
        if (
            crs.to_epsg() != 5179
            or not crs.is_projected
            or any(axis.unit_conversion_factor != 1.0 for axis in crs.axis_info)
        ):
            raise ValueError("ZIP PRJ conflicts with documented EPSG:5179 metre coordinates")
    geometries, names, orientation_alerts = {}, {}, []
    with shapefile.Reader(
        shp=io.BytesIO(data[".shp"]),
        dbf=io.BytesIO(data[".dbf"]),
        shx=io.BytesIO(data[".shx"]),
        encoding="cp949",
    ) as reader:
        fields = [f for f in reader.fields[1:] if f[0] == "EMD_CD"]
        if len(fields) != 1 or tuple(fields[0][1:3]) != ("C", 10):
            raise ValueError("Expected original character(10) H10")
        if reader.shapeType not in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM):
            raise ValueError("Expected a polygon source")
        if sum(1 for _ in reader.iterShapes()) != reader.numRecords:
            raise ValueError("SHP/DBF row counts disagree")
        for geometry_record, record in zip(
            reader.iterShapes(),
            reader.iterRecords(fields=["EMD_CD", "EMD_KOR_NM"], deleted_as_None=True),
            strict=True,
        ):
            if record is None:
                raise ValueError("Deleted DBF row cannot be joined safely")
            code = record["EMD_CD"]
            if not re.fullmatch(r"41[0-9]{6}00", code) or code in geometries:
                raise ValueError("Invalid or duplicated Gyeonggi H10")
            parts = list(geometry_record.parts) + [len(geometry_record.points)]
            rings = [geometry_record.points[a:b] for a, b in zip(parts, parts[1:], strict=False)]
            # 원본의 방향 문제도 남긴다. make_valid나 좌표 수정은 하지 않는다.
            signed = [
                sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:], strict=False)) / 2
                for ring in rings
            ]
            if signed and not any(area < 0 for area in signed):
                orientation_alerts.append({"h10_code": code, "part_count": len(rings)})
            geometries[code] = shape(geometry_record.__geo_interface__)
            names[code] = record["EMD_KOR_NM"]
    if _sha256(archive) != before:
        raise ValueError("Boundary source changed while reading")
    invalid = [
        {"h10_code": c, "name": names[c], "reason": shapely.is_valid_reason(g)}
        for c, g in geometries.items()
        if g.is_empty or not g.is_valid or not g.area > 0
    ]
    return geometries, {
        "archive_sha256": before,
        "count": len(geometries),
        "names": names,
        "invalid": invalid,
        "original_prj_present": ".prj" in data,
        "coordinate_reference_system": "EPSG:5179",
        "coordinate_reference_basis": CRS_BASIS_URL,
        "component_sha256": {ext: hashlib.sha256(b).hexdigest() for ext, b in data.items()},
        "ring_orientation_alerts": orientation_alerts,
        "geometry_interpretation": "pyshp GeoJSON; original ring-direction alerts retained",
        "repairs": "No make_valid, coordinate edits or source-code replacements",
    }


def locate_stores(
    stores_csv: Path,
    boundaries: dict,
    eligible_h10: set,
    *,
    batch_size: int = 10000,
    excluded_source_codes: set | None = None,
) -> tuple[Counter, dict]:
    """전체 도형으로 단일 내부 위치를 찾고 감사 통과 범위만 업종별로 센다.

    반환 Counter의 키는 (H10, 원본 SBIZ 소분류)이다. 원천 sbiz-admin-8은
    변환하지 않으며 source_to_geometry_pairs/source_code_groups에서만 대조한다.
    오류 원본 도형의 bbox에 걸린 점, 중첩·경계점은 보수적으로 제외한다.
    """
    pyproj, _shapefile, shapely, _shape, STRtree = _gis()
    stores_csv = Path(stores_csv)
    excluded_source_codes = set(excluded_source_codes or ())
    if any(not re.fullmatch(r"[0-9]{8}", code) for code in excluded_source_codes):
        raise ValueError("Excluded source codes must remain original eight-digit codes")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Expected a positive batch size")
    if not eligible_h10 <= boundaries.keys():
        raise ValueError("Eligible codes must be from the supplied boundary source")
    fields = DEFAULT_COLUMNS
    codes = list(boundaries)
    geometries = [boundaries[c] for c in codes]
    invalid = {
        i
        for i, g in enumerate(geometries)
        if g.is_empty or not g.is_valid or not math.isfinite(g.area) or g.area <= 0
    }
    tree = STRtree(geometries)
    transformer = pyproj.Transformer.from_crs(4326, 5179, always_xy=True, allow_ballpark=False)
    counts, excluded, pairs = Counter(), Counter(), Counter()
    seen_store_ids = set()
    source_exclusions = Counter()
    invalid_bbox_valid_hits = Counter()
    invalid_bbox_valid_hit_rows = 0
    input_rows = 0
    coordinate_located_rows = 0
    industry_seen = set()
    before = _sha256(stores_csv)

    def consume(batch):
        nonlocal coordinate_located_rows, invalid_bbox_valid_hit_rows
        if not batch:
            return
        x, y = transformer.transform([r[2] for r in batch], [r[3] for r in batch], errcheck=False)
        points = shapely.points(x, y)
        transform_failed = {
            i
            for i, (east, north) in enumerate(zip(x, y, strict=True))
            if not math.isfinite(east) or not math.isfinite(north)
        }
        hits = defaultdict(list)
        invalid_bbox = set()
        point_indices, geometry_indices = tree.query(points)
        valid_pairs = []
        for point_index, geometry_index in zip(point_indices, geometry_indices, strict=True):
            p, g = int(point_index), int(geometry_index)
            if g in invalid:
                invalid_bbox.add(p)
            else:
                valid_pairs.append((p, g))
        if valid_pairs:
            mask = shapely.covers(
                [geometries[g] for p, g in valid_pairs],
                [points[p] for p, g in valid_pairs],
            )
            for (p, g), covered in zip(valid_pairs, mask, strict=True):
                if covered:
                    hits[p].append(g)
        for p, (source_code, industry, _lon, _lat) in enumerate(batch):
            if p in transform_failed:
                excluded["coordinate_transform_failed"] += 1
            elif p in invalid_bbox:
                excluded["invalid_source_geometry_bbox"] += 1
                if hits[p]:
                    invalid_bbox_valid_hit_rows += 1
                    for geometry_index in hits[p]:
                        invalid_bbox_valid_hits[codes[geometry_index]] += 1
            elif not hits[p]:
                excluded["outside_supplied_boundaries"] += 1
            elif len(hits[p]) > 1:
                excluded["overlapping_boundaries"] += 1
            elif not geometries[hits[p][0]].contains(points[p]):
                excluded["on_boundary"] += 1
            else:
                code = codes[hits[p][0]]
                coordinate_located_rows += 1
                pairs[source_code, code] += 1
                if code not in eligible_h10:
                    excluded["outside_reporting_scope"] += 1
                else:
                    counts[code, industry] += 1

    with stores_csv.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("Expected unique CSV header fields")
        if not set(fields.values()) <= set(reader.fieldnames):
            raise ValueError("Store coordinate/source/industry columns missing")
        batch = []
        for row in reader:
            input_rows += 1
            if None in row or any(row.get(field) is None for field in fields.values()):
                raise ValueError("Malformed store row")
            store_id = row[fields["store_id"]]
            source_code = row[fields["source_code"]]
            if not re.fullmatch(r"[A-Za-z0-9]+", store_id):
                raise ValueError("Expected an unchanged nonempty ASCII store identifier")
            if store_id in seen_store_ids:
                raise ValueError("Duplicate original store identifier")
            seen_store_ids.add(store_id)
            if not re.fullmatch(r"[0-9]{8}", source_code):
                raise ValueError("Expected unchanged eight-digit sbiz-admin-8 source code")
            if source_code in excluded_source_codes:
                excluded["excluded_source_code"] += 1
                source_exclusions[source_code] += 1
                continue
            industry = row[fields["industry"]]
            if not re.fullmatch(r"[A-Z][0-9]{5}", industry):
                excluded["invalid_native_sbiz_industry"] += 1
                continue
            industry_seen.add(industry)
            try:
                lon, lat = float(row[fields["lon"]]), float(row[fields["lat"]])
            except ValueError:
                excluded["invalid_coordinates"] += 1
                continue
            if not (
                math.isfinite(lon)
                and math.isfinite(lat)
                and -180 <= lon <= 180
                and -90 <= lat <= 90
            ):
                excluded["invalid_coordinates"] += 1
                continue
            batch.append((row[fields["source_code"]], industry, lon, lat))
            if len(batch) >= batch_size:
                consume(batch)
                batch = []
        consume(batch)
    if _sha256(stores_csv) != before:
        raise ValueError("Store CSV changed while processing")
    groups = defaultdict(Counter)
    for (source_code, geometry_code), n in pairs.items():
        groups[source_code][geometry_code] += n
    summary = []
    for source_code, values in sorted(groups.items()):
        total = sum(values.values())
        dominant, largest = sorted(values.items(), key=lambda x: (-x[1], x[0]))[0]
        summary.append(
            {
                "source_code": source_code,
                "located_rows": total,
                "distinct_geometry_h10": len(values),
                "dominant_h10": dominant,
                "dominant_share": largest / total,
            }
        )
    audit = {
        "store_source_sha256": before,
        "input_rows": input_rows,
        "unique_store_ids": len(seen_store_ids),
        "excluded_source_codes": sorted(excluded_source_codes),
        "excluded_source_rows_by_code": dict(sorted(source_exclusions.items())),
        "coordinate_located_rows": coordinate_located_rows,
        "counted_rows": sum(counts.values()),
        "exclusions": dict(sorted(excluded.items())),
        "industry_count_in_source": len(industry_seen),
        "source_code_system": "sbiz-admin-8 (preserved, not converted)",
        "located_code_system": "MOIS H10 from original GEMD",
        "eligible_h10_count": len(eligible_h10),
        "invalid_boundary_h10": [codes[i] for i in sorted(invalid)],
        "invalid_bbox_valid_hit_rows": invalid_bbox_valid_hit_rows,
        "invalid_bbox_valid_hit_counts": dict(sorted(invalid_bbox_valid_hits.items())),
        "source_to_geometry_pairs": [
            {"source_code": source, "geometry_h10": target, "rows": n}
            for (source, target), n in sorted(pairs.items())
        ],
        "source_code_groups": summary,
        "location_policy": "Unique strict interior in full boundary source; invalid bbox withheld",
    }
    if audit["counted_rows"] + sum(excluded.values()) != input_rows:
        raise AssertionError("Store audit accounting does not balance")
    return counts, audit
