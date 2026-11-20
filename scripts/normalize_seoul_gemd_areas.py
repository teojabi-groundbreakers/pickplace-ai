"""검증된 행안부 행정동 도형을 기존 SGIS 면적 입력 형식으로 정규화한다.

압축 해제·다운로드·기준일 추정은 수행하지 않는다. 기준일을 확인한 담당자가
공식 근거와 구성요소/연계표 지문을 메타데이터에 기록한 경우에만 실행한다.
"""

import argparse
import csv
import json
import math
import os
import re
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit

from pickplace_ai.artifacts import file_sha256 as _sha256

LAYER = "TL_SCCO_GEMD"
AREA_FIELDS = ("reference_date", "sgis_code", "area_m2")
BRIDGE_FIELDS = ("reference_date", "sgis_code", "h10_code")
EXCLUDED_FIELDS = ("reference_date", "h10_code", "reason")
REQUIRED_COMPONENTS = (".shp", ".dbf", ".shx", ".prj")


def _official_url(value):
    if not isinstance(value, str):
        raise ValueError("Expected an official HTTPS source URL")
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if (
        parsed.scheme != "https"
        or not host.endswith(".go.kr")
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Expected an official HTTPS source URL")
    return value


def _digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a recorded SHA-256")
    return value


def _iso_date(value):
    if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
        raise ValueError("Expected an ISO reference date")
    return date.fromisoformat(value)


def _components(path):
    if path.stem.upper() != LAYER or path.suffix.lower() != ".shp":
        raise ValueError("Expected an extracted TL_SCCO_GEMD.shp")
    result = {}
    for candidate in path.parent.iterdir():
        suffix = candidate.suffix.lower()
        if candidate.stem.lower() == path.stem.lower() and suffix in (
            *REQUIRED_COMPONENTS,
            ".cpg",
        ):
            if suffix in result or not candidate.is_file():
                raise ValueError("Ambiguous GEMD sidecar components")
            result[suffix] = candidate
    if not all(suffix in result for suffix in REQUIRED_COMPONENTS):
        raise ValueError("GEMD requires SHP, DBF, SHX and actual PRJ components")
    return result


def _metadata(path, components, bridge):
    value = json.loads(path.read_text(encoding="utf-8"))
    reference = _iso_date(value["reference_date"])
    acquired = _iso_date(value["acquired_at"])
    basis = value["reference_date_basis"]
    if value.get("reference_date_verified") is not True or acquired < reference:
        raise ValueError("Expected a verified reference date and actual acquisition date")
    _official_url(value["source_url"])
    _official_url(basis["url"])
    if not isinstance(basis.get("description"), str) or not basis["description"].strip():
        raise ValueError("Expected official reference-date basis")
    if value.get("layer", LAYER) != LAYER:
        raise ValueError("Metadata refers to a different layer")
    if "publication_date" in value:
        if _iso_date(value["publication_date"]) > acquired:
            raise ValueError("Publication date is after acquisition")
    recorded = value["components"]
    if set(recorded) != set(components):
        raise ValueError("Metadata must bind every supplied GEMD component")
    for suffix, component in components.items():
        if _sha256(component) != _digest(recorded[suffix]):
            raise ValueError(f"GEMD {suffix} does not match recorded SHA-256")
    identity = value["bridge"]
    _official_url(identity["url"])
    if _iso_date(identity["reference_date"]) != reference:
        raise ValueError("Bridge identity uses a different reference date")
    if _sha256(bridge) != _digest(identity["sha256"]):
        raise ValueError("Bridge does not match recorded SHA-256")
    return value


def _read_bridge(path, reference_date):
    mapping, sgis_seen = {}, set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != BRIDGE_FIELDS:
            raise ValueError("Unexpected canonical bridge columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("Malformed bridge row")
            h10, sgis = row["h10_code"], row["sgis_code"]
            if (
                row["reference_date"] != reference_date
                or not re.fullmatch(r"11[0-9]{8}", h10)
                or not re.fullmatch(r"11[0-9]{6}", sgis)
            ):
                raise ValueError("Invalid bridge code or reference date")
            if h10 in mapping or sgis in sgis_seen:
                raise ValueError("Expected a one-to-one official bridge")
            mapping[h10] = sgis
            sgis_seen.add(sgis)
    if not mapping:
        raise ValueError("Empty official bridge")
    return mapping


def _write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)


def _check_unchanged(paths, hashes):
    if any(_sha256(path) != hashes[key] for key, path in paths.items()):
        raise ValueError("An input changed while normalizing; no outputs published")


def normalize(shapefile_path: Path, metadata_path: Path, bridge_path: Path, output: Path):
    """모든 원본 행을 검사한 뒤 새 경로에만 면적·근거·제외 내역을 발행한다."""
    if output.exists() or output.is_symlink():
        raise FileExistsError("Area output already exists")
    components = _components(shapefile_path)
    paths = {**components, "metadata": metadata_path, "bridge": bridge_path}
    hashes = {key: _sha256(path) for key, path in paths.items()}
    metadata = _metadata(metadata_path, components, bridge_path)
    reference = metadata["reference_date"]
    bridge = _read_bridge(bridge_path, reference)
    try:
        import pyproj
        import shapefile
        import shapely
        from shapely.geometry import shape
    except ImportError as error:
        raise RuntimeError("Offline normalization requires pyshp, pyproj and Shapely") from error

    # 실제 PRJ를 읽는다. 좌표계가 없을 때 EPSG를 임의 지정하지 않는다.
    source_crs = pyproj.CRS.from_wkt(components[".prj"].read_text(encoding="utf-8-sig"))
    if not (source_crs.is_projected or source_crs.is_geographic):
        raise ValueError("Expected a projected or geographic horizontal PRJ")
    target_crs = pyproj.CRS.from_epsg(5179)
    if any(axis.unit_name != "metre" for axis in target_crs.axis_info):
        raise ValueError("Area projection must measure metres")
    converter = pyproj.Transformer.from_crs(
        source_crs, target_crs, always_xy=True, allow_ballpark=False
    )

    def project(x, y):
        return converter.transform(x, y, errcheck=True)

    selected, excluded, seen = [], [], set()
    seoul_count = 0
    # 코드만 읽으므로 이름의 인코딩이나 표기를 매칭 근거로 사용하지 않는다.
    with shapefile.Reader(
        shp=str(components[".shp"]),
        shx=str(components[".shx"]),
        dbf=str(components[".dbf"]),
        encoding="ascii",
    ) as reader:
        fields = [field for field in reader.fields[1:] if field[0] == "EMD_CD"]
        if len(fields) != 1 or tuple(fields[0][1:3]) != ("C", 10):
            raise ValueError("Expected original character(10) EMD_CD")
        if reader.shapeType not in (shapefile.POLYGON, shapefile.POLYGONZ, shapefile.POLYGONM):
            raise ValueError("Expected a GEMD polygon layer")
        shape_count = sum(1 for _ in reader.iterShapes())
        if shape_count != reader.numRecords:
            raise ValueError("SHP and DBF record counts differ")
        # 삭제된 DBF 행도 위치를 유지한다. 건너뛰면 다음 코드와 이전 도형이
        # 잘못 결합할 수 있으므로, 발견 즉시 전체 입력을 거부한다.
        for geometry_record, attribute_record in zip(
            reader.iterShapes(),
            reader.iterRecords(fields=["EMD_CD"], deleted_as_None=True),
            strict=True,
        ):
            if attribute_record is None:
                raise ValueError("Deleted DBF rows cannot be joined safely")
            h10 = attribute_record["EMD_CD"]
            if not isinstance(h10, str) or not re.fullmatch(r"[0-9]{10}", h10):
                raise ValueError("Expected an unchanged ten-digit H10")
            if h10 in seen:
                raise ValueError("Duplicated H10 geometry is ambiguous")
            seen.add(h10)
            reason = None
            if not h10.startswith("11"):
                reason = "outside_seoul"
            else:
                seoul_count += 1
                if h10 not in bridge:
                    reason = "official_bridge_missing"
                elif geometry_record.shapeType == shapefile.NULL:
                    reason = "empty_geometry"
                else:
                    geometry = shape(geometry_record.__geo_interface__)
                    if geometry.is_empty or not geometry.is_valid:
                        reason = "invalid_geometry"
                    else:
                        projected = shapely.transform(geometry, project, interleaved=False)
                        area = projected.area
                        if projected.is_empty or not projected.is_valid:
                            reason = "invalid_projected_geometry"
                        elif not math.isfinite(area) or area <= 0:
                            reason = "nonpositive_or_nonfinite_area"
                        else:
                            selected.append((reference, bridge[h10], area))
            if reason:
                excluded.append((reference, h10, reason))
    for h10 in sorted(bridge.keys() - seen):
        excluded.append((reference, h10, "geometry_missing"))
    if not selected:
        raise ValueError("No verified Seoul polygon areas remain")
    selected.sort(key=lambda row: row[1])
    excluded.sort(key=lambda row: (row[1], row[2]))
    manifest = {
        "kind": "seoul_gemd_areas_v1",
        "layer": LAYER,
        "source_kind": "official",
        "reference_date": reference,
        "reference_date_verified": True,
        "reference_date_basis": metadata["reference_date_basis"],
        "source_url": metadata["source_url"],
        "acquired_at": metadata["acquired_at"],
        "publication_date": metadata.get("publication_date"),
        "source_projection_wkt": source_crs.to_wkt(),
        "area_projection": "EPSG:5179",
        "area_unit": "m2",
        "transformation": "always_xy=True, allow_ballpark=False, errcheck=True; no repair",
        "input_sha256": hashes,
        "component_files": {key: path.name for key, path in components.items()},
        "bridge_identity": {**metadata["bridge"], "row_count": len(bridge)},
        "input_record_count": shape_count,
        "seoul_record_count": seoul_count,
        "row_count": len(selected),
        "exclusions": dict(sorted(Counter(row[2] for row in excluded).items())),
        "versions": {
            "pyshp": shapefile.__version__,
            "pyproj": pyproj.__version__,
            "PROJ": pyproj.proj_version_str,
            "Shapely": shapely.__version__,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        stage = Path(directory)
        _write_csv(stage / "areas.csv", AREA_FIELDS, selected)
        _write_csv(stage / "excluded.csv", EXCLUDED_FIELDS, excluded)
        manifest["area_source"] = {
            "url": metadata["source_url"],
            "sha256": _sha256(stage / "areas.csv"),
            "reference_date": reference,
            "acquired_at": metadata["acquired_at"],
        }
        manifest["excluded_sha256"] = _sha256(stage / "excluded.csv")
        (stage / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _check_unchanged(paths, hashes)
        output.mkdir()
        published = []
        try:
            # 근거 파일은 마지막에 발행한다. 기존 경로나 파일을 덮어쓰지 않는다.
            for name in ("areas.csv", "excluded.csv", "manifest.json"):
                os.link(stage / name, output / name)
                published.append(output / name)
        except BaseException:
            for path in published:
                path.unlink(missing_ok=True)
            output.rmdir()
            raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("shapefile", "metadata", "bridge", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = normalize(args.shapefile, args.metadata, args.bridge, args.output)
    print(f"{result['reference_date']} verified Seoul areas: {result['row_count']}")
    print(f"Excluded: {result['exclusions']}")


if __name__ == "__main__":
    main()
