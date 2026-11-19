"""Fetch one small, unclassified VWorld feature sample for contract inspection.

The supplied layer ID and returned attributes are unverified. This probe must not be
used to classify a whole administrative dong or as a historical model feature.
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus, urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

API_URL = "https://api.vworld.kr/req/data"
CRS = "EPSG:3857"
PAGE_SIZE = 100
MAX_RESPONSE_BYTES = 2_000_000
MERCATOR_RADIUS_M = 6_378_137
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data/raw/nationwide/zoning/vworld"


class ProbeError(ValueError):
    """The request or response is unsuitable for a raw VWorld probe."""


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _validate_inputs(layer_id, lon, lat, radius_m, key, domain):
    if not isinstance(layer_id, str) or not re.fullmatch(r"[A-Za-z0-9_]+", layer_id):
        raise ProbeError("Provide a layer ID confirmed in the official VWorld catalog")
    if not isinstance(key, str) or not key or any(c.isspace() for c in key):
        raise ProbeError("Set VWORLD_API_KEY in the environment")
    if layer_id == key:
        raise ProbeError("Layer ID must differ from the API key")
    if not all(
        isinstance(value, (int, float)) and math.isfinite(value) for value in (lon, lat, radius_m)
    ):
        raise ProbeError("Coordinates and radius must be finite numbers")
    if not (-180 <= lon <= 180 and -85 < lat < 85 and 0 < radius_m <= 100):
        raise ProbeError("Coordinates or radius are outside the probe range")
    if domain is not None:
        parsed = urlsplit(domain)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or any(c.isspace() for c in domain)
        ):
            raise ProbeError("VWORLD_API_DOMAIN must be a registered HTTP(S) URL")


def _bbox(lon, lat, radius_m):
    lon_radians = math.radians(lon)
    lat_radians = math.radians(lat)
    x = MERCATOR_RADIUS_M * lon_radians
    y = MERCATOR_RADIUS_M * math.log(math.tan(math.pi / 4 + lat_radians / 2))
    # EPSG:3857 distances are enlarged by 1/cos(latitude). A small box remains
    # only a location probe; its results cannot be extrapolated to a district.
    projected_radius = radius_m / math.cos(lat_radians)
    return [
        round(v, 3)
        for v in (
            x - projected_radius,
            y - projected_radius,
            x + projected_radius,
            y + projected_radius,
        )
    ]


def _params(layer_id, bbox, key, domain):
    parameters = {
        "service": "data",
        "version": "2.0",
        "request": "GetFeature",
        "format": "json",
        "data": layer_id,
        "geomFilter": "BOX(" + ",".join(str(v) for v in bbox) + ")",
        "crs": CRS,
        "size": str(PAGE_SIZE),
        "page": "1",
        "key": key,
    }
    if domain:
        parameters["domain"] = domain
    return parameters


def request_features(parameters):
    """Return bounded raw bytes; never expose the key-bearing URL in an error."""
    request = Request(
        API_URL + "?" + urlencode(parameters),
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with build_opener(NoRedirect).open(request, timeout=30) as response:
            if response.status != 200:
                raise ProbeError("VWorld did not return HTTP 200")
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except (HTTPError, URLError, TimeoutError, OSError):
        raise ProbeError(
            "VWorld request failed; check key, registered domain, layer and limits"
        ) from None
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ProbeError("VWorld response exceeds the probe size limit")
    return raw


def _check_response(raw, key):
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise ProbeError("VWorld returned an empty or oversized response")
    # Even successful APIs can echo credentials. Never persist a response that does.
    if key.encode() in raw or quote_plus(key).encode() in raw:
        raise ProbeError("VWorld response contains a credential and was not saved")
    try:
        data = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeError, ValueError):
        raise ProbeError("VWorld returned invalid JSON") from None
    if key in json.dumps(data, ensure_ascii=False):
        raise ProbeError("VWorld response contains a credential and was not saved")
    if not isinstance(data, dict) or not isinstance(data.get("response"), dict):
        raise ProbeError("VWorld response envelope is unrecognized")
    envelope = data["response"]
    if envelope.get("status") != "OK":
        raise ProbeError("VWorld did not return a successful feature collection")
    result = envelope.get("result")
    collection = result.get("featureCollection") if isinstance(result, dict) else None
    features = collection.get("features") if isinstance(collection, dict) else None
    if not isinstance(features, list) or any(not isinstance(feature, dict) for feature in features):
        raise ProbeError("VWorld feature collection is unrecognized")
    if len(features) > PAGE_SIZE:
        raise ProbeError("VWorld returned more features than requested")
    return len(features)


def _default_output():
    now = datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    return RAW_ROOT / now.strftime("%Y%m%d") / f"probe-{stamp}.json"


def probe_layer(
    layer_id, lon, lat, radius_m, key, output=None, *, domain=None, request=request_features
):
    """Save a single raw sample and a sidecar; never overwrite an existing pair."""
    _validate_inputs(layer_id, lon, lat, radius_m, key, domain)
    destination = Path(output) if output is not None else _default_output()
    metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
    if destination.exists() or metadata_path.exists():
        raise FileExistsError("Refusing to overwrite an existing VWorld probe")
    bbox = _bbox(lon, lat, radius_m)
    started_at = datetime.now(UTC).isoformat()
    try:
        raw = request(_params(layer_id, bbox, key, domain))
    except Exception:
        raise ProbeError("VWorld request failed; no probe was saved") from None
    count = _check_response(raw, key)
    digest = hashlib.sha256(raw).hexdigest()
    metadata = {
        "provider": "VWorld",
        "api_endpoint": API_URL,
        "request_kind": "GetFeature",
        "layer_id": layer_id,
        "geometry_scope": "small_bbox_probe",
        "center_wgs84": {"lon": lon, "lat": lat},
        "radius_m_approx": radius_m,
        "bbox_crs": CRS,
        "bbox": bbox,
        "page": 1,
        "page_size": PAGE_SIZE,
        "feature_count": count,
        "possible_truncation": count == PAGE_SIZE,
        "request_started_at_utc": started_at,
        "retrieved_at_utc": datetime.now(UTC).isoformat(),
        "raw_sha256": digest,
        "layer_schema_verified": False,
        "historical_validity_verified": False,
        "coverage_verified": False,
        "training_eligible": False,
        "effective_at": None,
        "publication_date": None,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw_temp = metadata_temp = None
    linked_metadata = False
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            raw_temp = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=destination.parent, delete=False
        ) as stream:
            metadata_temp = Path(stream.name)
            json.dump(metadata, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(metadata_temp, metadata_path)
        linked_metadata = True
        os.link(raw_temp, destination)
    except BaseException:
        if linked_metadata:
            metadata_path.unlink(missing_ok=True)
        raise
    finally:
        if raw_temp is not None:
            raw_temp.unlink(missing_ok=True)
        if metadata_temp is not None:
            metadata_temp.unlink(missing_ok=True)
    return destination, count, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer-id", required=True, help="Officially verified VWorld layer ID")
    parser.add_argument("--lon", type=float, required=True, help="WGS84 longitude of sample center")
    parser.add_argument("--lat", type=float, required=True, help="WGS84 latitude of sample center")
    parser.add_argument(
        "--radius-m", type=float, default=10.0, help="Approximate ground radius, <= 100 m"
    )
    parser.add_argument(
        "--output", type=Path, help="New raw JSON path under the ignored data folder"
    )
    args = parser.parse_args()
    if args.output is not None and not args.output.resolve().is_relative_to(RAW_ROOT.resolve()):
        parser.error("--output must be under the ignored data/raw/nationwide/zoning/vworld folder")
    try:
        destination, count, digest = probe_layer(
            args.layer_id,
            args.lon,
            args.lat,
            args.radius_m,
            os.environ.get("VWORLD_API_KEY"),
            args.output,
            domain=os.environ.get("VWORLD_API_DOMAIN"),
        )
    except (ProbeError, OSError):
        print(
            "VWorld probe failed; check inputs, credentials and response. No model was changed.",
            file=sys.stderr,
        )
        return 1
    print(f"Saved {count} unclassified features to {destination}; SHA-256: {digest}")
    print("Layer schema, area coverage and historical validity remain unverified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
