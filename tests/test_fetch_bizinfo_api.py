import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "fetch_bizinfo_api.py"
spec = importlib.util.spec_from_file_location("fetch_bizinfo_api", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def response(ids, total):
    items = [{"seq": name, "title": "공고 " + name, "totCnt": str(total)} for name in ids]
    return json.dumps({"jsonArray": {"item": items}}, ensure_ascii=False).encode()


def test_pagination_snapshot_and_no_overwrite(tmp_path):
    payloads = [response(["1", "2"], 3), response(["3"], 3)]
    rows, pages = module.collect(
        "test-credential-123456", 2, 5, None, lambda key, page, size, tags: payloads[page - 1]
    )
    dest = tmp_path / "snapshot"
    module.save_snapshot(dest, rows, pages, 2, None)
    with (dest / "announcements.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert [row["seq"] for row in csv.DictReader(stream)] == ["1", "2", "3"]
    metadata = json.loads((dest / "metadata.json").read_text())
    assert metadata["row_count"] == 3 and metadata["page_count"] == 2
    assert "test-credential-123456" not in json.dumps(metadata)
    with pytest.raises(module.CollectionError):
        module.save_snapshot(dest, rows, pages, 2, None)


@pytest.mark.parametrize(
    "payloads",
    [
        [response(["1", "2"], 3), response(["2"], 3)],
        [response(["1", "2"], 3), response(["3"], 4)],
        [response(["1"], 3)],
        [b'{"RESULT":{"CODE":"ERROR"}}'],
        [b"not-json"],
    ],
)
def test_reject_incomplete_or_bad_responses(payloads):
    with pytest.raises(module.CollectionError):
        module.collect(
            "test-credential-123456", 2, 5, None, lambda key, page, size, tags: payloads[page - 1]
        )


def test_page_cap_missing_key_and_terms():
    with pytest.raises(module.CollectionError):
        module.collect("test-credential-123456", 2, 1, None, lambda *args: response(["1", "2"], 3))
    with pytest.raises(module.CollectionError):
        module.collect("", 2, 1, None)
    assert module.main(["--output-dir", "/tmp/not-used-bizinfo-output"]) == 1


def test_key_in_response_aborts():
    with pytest.raises(module.CollectionError):
        module.collect(
            "test-credential-123456", 2, 1, None, lambda *args: b"test-credential-123456"
        )
