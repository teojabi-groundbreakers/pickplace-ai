"""가상 원천으로 경기 비교표의 시점·범위 검증을 검사한다."""

import json
from collections import Counter
from types import SimpleNamespace

import pytest

from pickplace_ai.artifacts import file_sha256
from pickplace_ai.fit_reference import GYEONGGI_KIND, FitReference
from scripts import build_gyeonggi_fit_reference as builder


@pytest.fixture
def inputs(tmp_path):
    names = ("area", "population", "stores")
    paths = [tmp_path / name for name in names]
    sources = {}
    for name, path in zip(names, paths, strict=True):
        path.write_text(name, encoding="utf-8")
        sources[name] = {
            "reference_date": "2026-06-30",
            "acquired_at": "2026-11-20",
            "url": f"https://fixture.go.kr/{name}",
            "sha256": file_sha256(path),
        }
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps({"sources": sources}), encoding="utf-8")
    return (*paths, evidence, tmp_path / "output")


def test_only_exact_h10_with_usable_boundaries_enter_the_reference(inputs, monkeypatch):
    included = "4111111100"
    branch = builder.UNVERIFIED_BRANCH_SCOPE
    invalid = "4157057000"
    new_code = "4117158200"
    old_code = "4117158000"
    split_a, split_b = "4111111200", "4111111300"
    neighbor = "4157057500"
    valid_geometry = SimpleNamespace(is_valid=True, is_empty=False, area=2_000_000)
    invalid_geometry = SimpleNamespace(is_valid=False, is_empty=False, area=0)
    boundaries = {
        included: valid_geometry,
        branch: valid_geometry,
        invalid: invalid_geometry,
        new_code: valid_geometry,
        split_a: valid_geometry,
        split_b: valid_geometry,
        neighbor: valid_geometry,
    }
    populations = {
        included: 800,
        branch: 20_000,
        invalid: 25_000,
        old_code: 20_000,
        split_a: 4000,
        split_b: 2000,
        neighbor: 1000,
    }

    def locate(_stores, all_boundaries, eligible, *, excluded_source_codes):
        assert eligible == {included, split_a, split_b, neighbor}
        assert all_boundaries == boundaries
        assert excluded_source_codes == {"41171582", "41171583"}
        counts = Counter(
            {(included, "I21201"): 20, (split_a, "I21201"): 80, (split_b, "I21201"): 20}
        )
        counts[neighbor, "I21201"] = 10
        return counts, {
            "source_code_groups": [{"source_code": "41111112", "dominant_share": 0.8}],
            "source_to_geometry_pairs": [
                {"source_code": "41111112", "geometry_h10": code} for code in (split_a, split_b)
            ],
            "invalid_bbox_valid_hit_counts": {neighbor: 1},
        }

    monkeypatch.setattr(builder, "inspect_boundaries", lambda _archive: (boundaries, {}))
    monkeypatch.setattr(builder, "read_mois_gyeonggi_population", lambda *_args: populations)
    monkeypatch.setattr(builder, "locate_stores", locate)
    metadata = builder.build(*inputs)
    output = inputs[-1]
    assert metadata["row_count"] == 1
    reference = FitReference(output, GYEONGGI_KIND)
    assert reference.scopes[included, "I21201"] == (20, 800, 2.0, (40.0, 10.0))
    audit = json.loads((output / "source-audit.json").read_text(encoding="utf-8"))
    assert audit["shape_only"] == [new_code]
    assert audit["population_only"] == [old_code]
    assert audit["invalid_geometry"] == [invalid]
    assert audit["branch_scope_excluded"] == [branch]
    assert audit["unresolved_store_scopes"] == [split_a, split_b]
    assert audit["incomplete_neighbor_scopes"] == [neighbor]
    assert audit["stores_in_reference"] == 20
    assert audit["stores_withheld_in_unresolved_scopes"] == 110


@pytest.mark.parametrize("replacement", [{"reference_date": "2026-09-30"}, {"sha256": "0" * 64}])
def test_mismatched_store_input_is_rejected_before_spatial_processing(inputs, replacement):
    evidence = inputs[-2]
    content = json.loads(evidence.read_text(encoding="utf-8"))
    content["sources"]["stores"].update(replacement)
    evidence.write_text(json.dumps(content), encoding="utf-8")
    with pytest.raises(ValueError, match="stores"):
        builder.build(*inputs)
    assert not inputs[-1].exists()
