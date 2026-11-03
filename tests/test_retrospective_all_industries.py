import csv
import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from pickplace_ai.artifacts import ArtifactError, load_bundle
from pickplace_ai.data import write_observations
from pickplace_ai.demo import synthetic_observations
from scripts import retrospective_all_industries as all_experiments

SNAPSHOT = date(2026, 11, 3)


def _rows():
    rows = synthetic_observations(3, 24, 42)
    return [
        row.model_copy(update={"industry_code": code, "available_at": SNAPSHOT})
        for code in ("DEMO_A", "DEMO_B")
        for row in rows
    ]


def _write_input(path):
    write_observations(_rows(), path)
    return path


def _metadata(code):
    return {
        "eligible_rows": 40,
        "rows_by_partition": {"train": 100, "validation": 29, "test": 9},
        "quarter_splits": {
            "train": ["2022Q2"],
            "validation": ["2024Q4", "2025Q1"],
            "test": ["2025Q3", "2025Q4"] if code == "DEMO_A" else ["2024Q3", "2024Q4"],
        },
        "embargo_quarters": ["2024Q3", "2025Q2"],
        "selection": "recent_growth",
        "learned_model_selection": "hist",
    }


def _evaluation():
    return {
        "validation": {"recent_growth": {"mae_pp": 10}, "hist": {"mae_pp": 12}},
        "test": {"recent_growth": {"mae_pp": 11}, "hist": {"mae_pp": 13}},
    }


def _fake_artifacts(output_path):
    output_path.mkdir()
    (output_path / "MODEL_CARD.md").write_text("# Test\n")


def test_industries_partitioned_summary_and_warnings(tmp_path, monkeypatch):
    source = _write_input(tmp_path / "source.csv")
    names = tmp_path / "names.json"
    names.write_text(
        json.dumps({"names": {"DEMO_A": "업종 가", "DEMO_B": "업종 나"}, "conflicts": {}}),
        encoding="utf-8",
    )
    calls = []

    def fake_run(input_path, output_path, snapshot_at, **kwargs):
        with input_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        codes = {row["industry_code"] for row in rows}
        assert len(codes) == 1
        code = codes.pop()
        assert output_path.name == code
        assert snapshot_at == SNAPSHOT
        assert kwargs["source_kind"] == "synthetic"
        calls.append(code)
        _fake_artifacts(output_path)
        return _metadata(code), _evaluation()

    monkeypatch.setattr(all_experiments.experiment, "run", fake_run)
    output = tmp_path / "out"
    result = all_experiments.run_all(
        source, output, SNAPSHOT, source_kind="synthetic", industry_names_path=names
    )
    assert calls == ["DEMO_A", "DEMO_B"]
    assert result["completed_count"] == 2
    assert result["failed_count"] == 0
    assert result["input_observations"] == len(_rows())
    assert result["industries"][0]["industry_name"] == "업종 가"
    assert "fewer_than_30_test_samples" in result["industries"][0]["warnings"]
    assert "fewer_than_30_validation_samples" in result["industries"][0]["warnings"]
    assert result["warning_thresholds"]["validation_rows_below"] == 30
    assert "different_test_period_from_most_industries" in result["industries"][1]["warnings"]
    with (output / "summary.csv").open(encoding="utf-8", newline="") as stream:
        saved = list(csv.DictReader(stream))
    assert saved[0]["test_rows"] == "9"
    assert saved[0]["test_quarters"] == "2025Q3,2025Q4"
    assert json.loads((output / "summary.json").read_text())["industry_count"] == 2
    saved_names = (output / "industry-names.json").read_bytes()
    assert saved_names == names.read_bytes()
    assert hashlib.sha256(saved_names).hexdigest() == result["industry_names_sha256"]
    with pytest.raises(FileExistsError):
        all_experiments.run_all(source, output, SNAPSHOT, source_kind="synthetic")


def test_industry_failure_is_recorded_and_other_industries_continue(tmp_path, monkeypatch):
    source = _write_input(tmp_path / "source.csv")

    def fake_run(input_path, output_path, snapshot_at, **kwargs):
        if input_path.stem == "DEMO_A":
            output_path.mkdir()
            (output_path / "model.joblib").write_bytes(b"partial-test-artifact")
            raise ValueError("No retrospective samples")
        _fake_artifacts(output_path)
        return _metadata(input_path.stem), _evaluation()

    monkeypatch.setattr(all_experiments.experiment, "run", fake_run)
    output = tmp_path / "out"
    result = all_experiments.run_all(source, output, SNAPSHOT, source_kind="synthetic")
    assert result["failed_count"] == 1
    assert result["completed_count"] == 1
    assert result["industries"][0]["error_type"] == "ValueError"
    assert "No retrospective samples" in result["industries"][0]["error"]
    assert result["industries"][0]["artifact_status"] == "quarantined_partial"
    assert result["industries"][0]["artifact_path"] == "failed/DEMO_A"
    assert (output / "failed" / "DEMO_A" / "model.joblib").read_bytes() == b"partial-test-artifact"
    assert not (output / "industries" / "DEMO_A").exists()
    with (output / "summary.csv").open(encoding="utf-8", newline="") as stream:
        recorded = list(csv.DictReader(stream))
    assert recorded[0]["status"] == "failed"
    assert recorded[0]["artifact_path"] == "failed/DEMO_A"
    assert (output / "summary.json").exists()


def test_failure_after_training_is_not_marked_complete(tmp_path, monkeypatch):
    source = _write_input(tmp_path / "source.csv")

    def fake_run(input_path, output_path, snapshot_at, **kwargs):
        output_path.mkdir()
        (output_path / "MODEL_CARD.md").write_text("# Test\n")
        metadata = _metadata(input_path.stem)
        if input_path.stem == "DEMO_B":
            del metadata["rows_by_partition"]["validation"]
        return metadata, _evaluation()

    monkeypatch.setattr(all_experiments.experiment, "run", fake_run)
    output = tmp_path / "out"
    result = all_experiments.run_all(source, output, SNAPSHOT, source_kind="synthetic")
    assert result["completed_count"] == 1
    assert result["failed_count"] == 1
    assert result["industries"][1]["status"] == "failed"
    assert result["industries"][1]["artifact_status"] == "quarantined_partial"
    assert (output / "failed" / "DEMO_B").is_dir()


def test_quarantine_failure_is_explicit_and_partial_artifact_remains(tmp_path, monkeypatch):
    source = _write_input(tmp_path / "source.csv")

    def fake_run(input_path, output_path, snapshot_at, **kwargs):
        output_path.mkdir()
        raise ValueError("partial")

    def fail_rename(self, target):
        raise OSError("rename failed")

    monkeypatch.setattr(all_experiments.experiment, "run", fake_run)
    monkeypatch.setattr(Path, "rename", fail_rename)
    output = tmp_path / "out"
    result = all_experiments.run_all(source, output, SNAPSHOT, source_kind="synthetic")
    assert result["failed_count"] == 2
    row = result["industries"][0]
    assert row["artifact_status"] == "partial_not_quarantined"
    assert row["artifact_path"] == "industries/DEMO_A"
    assert row["quarantine_error"] == "rename failed"
    assert (output / row["artifact_path"]).is_dir()


@pytest.mark.parametrize("change", ["duplicate", "invalid_code", "wrong_date", "bad_fields"])
def test_invalid_input_fails_before_output_creation(tmp_path, change):
    source = _write_input(tmp_path / "source.csv")
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        headers, rows = reader.fieldnames, list(reader)
    if change == "duplicate":
        rows.append(rows[0].copy())
    elif change == "invalid_code":
        rows[0]["industry_code"] = "../escape"
    elif change == "wrong_date":
        rows[0]["available_at"] = "2024-01-01"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
        if change == "bad_fields":
            stream.write("too,few,fields\n")
    with pytest.raises(ValueError):
        all_experiments.run_all(source, tmp_path / "out", SNAPSHOT, source_kind="synthetic")
    assert not (tmp_path / "out").exists()


def test_empty_input_and_conflicting_names_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("industry_code\n", encoding="utf-8")
    with pytest.raises(ValueError):
        all_experiments.run_all(empty, tmp_path / "out", SNAPSHOT, source_kind="synthetic")
    source = _write_input(tmp_path / "source.csv")
    names = tmp_path / "names.json"
    names.write_text(json.dumps({"names": {}, "conflicts": {"DEMO_A": ["a", "b"]}}))
    with pytest.raises(ValueError, match="conflict-free"):
        all_experiments.run_all(
            source, tmp_path / "out", SNAPSHOT, source_kind="synthetic", industry_names_path=names
        )


def test_real_experiment_for_one_synthetic_industry(tmp_path, monkeypatch):
    rows = [row for row in _rows() if row.industry_code == "DEMO_A"]
    source = tmp_path / "source.csv"
    write_observations(rows, source)
    monkeypatch.setattr(
        all_experiments.experiment.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="synthetic-repository-head\n"),
    )
    output = tmp_path / "out"
    summary = all_experiments.run_all(source, output, SNAPSHOT, source_kind="synthetic")
    assert summary["completed_count"] == 1
    folder = output / "industries" / "DEMO_A"
    assert (folder / "model.joblib").is_file()
    assert (folder / "learned-model.joblib").is_file()
    metadata = json.loads((folder / "metadata.json").read_text())
    assert metadata["industry_code"] == "DEMO_A"
    assert metadata["all_industries_input_sha256"] == summary["input_sha256"]
    assert metadata["release_status"] == "retrospective_research_only"
    with pytest.raises(ArtifactError):
        load_bundle(folder)


def test_official_manifest_validation_and_per_industry_scope(tmp_path, monkeypatch):
    source = tmp_path / "source.csv"
    write_observations(
        [row.model_copy(update={"source_kind": "official"}) for row in _rows()], source
    )
    root = tmp_path / "raw"
    entries = []
    for category in ("sales", "stores"):
        (root / category).mkdir(parents=True)
        path = root / category / "fixture.csv"
        path.write_bytes(f"mock-{category}".encode())
        entries.append(
            {
                "category": category,
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "snapshot_available_at": SNAPSHOT.isoformat(),
                "industry_code": "all",
                "source_files": entries,
            }
        )
    )

    def fake_run(input_path, output_path, snapshot_at, **kwargs):
        item = json.loads(kwargs["manifest_path"].read_text())
        assert item["industry_code"] == input_path.stem
        assert item["source_files"] == entries
        _fake_artifacts(output_path)
        return _metadata(input_path.stem), _evaluation()

    monkeypatch.setattr(all_experiments.experiment, "run", fake_run)
    output = tmp_path / "out"
    summary = all_experiments.run_all(source, output, SNAPSHOT, manifest, root)
    assert summary["completed_count"] == 2
    assert json.loads((output / "source-manifest.json").read_text())["industry_code"] == "all"
    (root / "sales" / "fixture.csv").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        all_experiments.run_all(source, tmp_path / "out2", SNAPSHOT, manifest, root)
