"""Offline-only contract tests; the VWorld layer and live response remain unverified."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "probe_vworld_zoning", Path(__file__).parents[1] / "scripts/probe_vworld_zoning.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


RAW = json.dumps(
    {
        "response": {
            "status": "OK",
            "result": {
                "featureCollection": {
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {"source_label": "unverified example"},
                            "geometry": {"type": "Polygon", "coordinates": []},
                        }
                    ]
                }
            },
        }
    }
).encode()


def test_writes_raw_sample_without_pretending_it_is_training_data(tmp_path):
    destination = tmp_path / "probe.json"
    calls = []

    def fake_request(params):
        calls.append(params)
        return RAW

    saved, count, digest = mod.probe_layer(
        "VERIFIED_LAYER_ID", 127.0, 37.5, 10, "private-key", destination, request=fake_request
    )
    assert saved == destination
    assert count == 1
    assert destination.read_bytes() == RAW
    assert digest == hashlib.sha256(RAW).hexdigest()
    metadata = json.loads(destination.with_suffix(".json.metadata.json").read_text())
    assert metadata["raw_sha256"] == digest
    assert metadata["training_eligible"] is False
    assert metadata["layer_schema_verified"] is False
    assert metadata["historical_validity_verified"] is False
    assert metadata["coverage_verified"] is False
    assert metadata["geometry_scope"] == "small_bbox_probe"
    assert metadata["effective_at"] is None
    assert "private-key" not in str(metadata)
    assert calls[0]["service"] == "data"
    assert calls[0]["request"] == "GetFeature"
    assert calls[0]["data"] == "VERIFIED_LAYER_ID"
    assert calls[0]["crs"] == "EPSG:3857"
    assert calls[0]["geomFilter"].startswith("BOX(")
    assert calls[0]["size"] == "100"
    assert calls[0]["page"] == "1"


@pytest.mark.parametrize(
    "raw",
    [
        b"",
        b"not-json",
        b' {"response":{"status":"ERROR","message":"private-key"}}',
        b'{"response":{"status":"ERROR"}}',
        b'{"response":{"status":"OK","result":null}}',
        b'{"response":{"status":"OK","result":{"featureCollection":{"features":{}}}}}',
        json.dumps(
            {"response": {"status": "OK", "result": {"featureCollection": {"features": [1]}}}}
        ).encode(),
        b"x" * (mod.MAX_RESPONSE_BYTES + 1),
    ],
)
def test_bad_response_is_never_saved(tmp_path, raw):
    destination = tmp_path / "probe.json"
    with pytest.raises(mod.ProbeError):
        mod.probe_layer("LAYER", 127.0, 37.5, 10, "private-key", destination, request=lambda _: raw)
    assert not destination.exists()
    assert not destination.with_suffix(".json.metadata.json").exists()


def test_request_failure_does_not_echo_secret(tmp_path):
    def leak(_):
        raise RuntimeError("private-key appeared in a URL")

    with pytest.raises(mod.ProbeError) as caught:
        mod.probe_layer("LAYER", 127.0, 37.5, 10, "private-key", tmp_path / "x", request=leak)
    assert "private-key" not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_existing_output_is_preserved_without_request(tmp_path):
    destination = tmp_path / "probe.json"
    destination.write_bytes(b"original")

    def unexpected(_):
        raise AssertionError("A request should not have been made")

    with pytest.raises(FileExistsError):
        mod.probe_layer("LAYER", 127.0, 37.5, 10, "private-key", destination, request=unexpected)
    assert destination.read_bytes() == b"original"


@pytest.mark.parametrize(
    ("layer", "lon", "lat", "radius", "key", "domain"),
    [
        ("bad/layer", 127, 37.5, 10, "private-key", None),
        ("LAYER", 127, 37.5, 10, None, None),
        ("LAYER", 127, 37.5, 10, "bad key", None),
        ("LAYER", float("nan"), 37.5, 10, "private-key", None),
        ("LAYER", 127, 90, 10, "private-key", None),
        ("LAYER", 127, 37.5, 101, "private-key", None),
        ("LAYER", 127, 37.5, 10, "private-key", "file:///tmp/key"),
        ("LAYER", 127, 37.5, 10, "private-key", "https://user:pass@example.com"),
    ],
)
def test_invalid_inputs_never_request(tmp_path, layer, lon, lat, radius, key, domain):
    def unexpected(_):
        raise AssertionError("A request should not have been made")

    with pytest.raises(mod.ProbeError):
        mod.probe_layer(
            layer, lon, lat, radius, key, tmp_path / "probe.json", domain=domain, request=unexpected
        )
    assert list(tmp_path.iterdir()) == []


def test_empty_success_is_kept_as_a_probe_not_a_zone_classification(tmp_path):
    raw = b'{"response":{"status":"OK","result":{"featureCollection":{"features":[]}}}}'
    destination = tmp_path / "empty.json"
    _, count, _ = mod.probe_layer(
        "LAYER", 127.0, 37.5, 10, "private-key", destination, request=lambda _: raw
    )
    assert count == 0
    metadata = json.loads(destination.with_suffix(".json.metadata.json").read_text())
    assert metadata["feature_count"] == 0
    assert metadata["training_eligible"] is False
