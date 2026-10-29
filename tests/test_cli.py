from pickplace_ai.cli import main
from pickplace_ai.data import read_observations


def test_make_demo_inspect_prepare_roundtrip(tmp_path):
    path = tmp_path / "synthetic.csv"
    request = tmp_path / "request.json"
    assert (
        main(
            [
                "make-demo",
                "--output",
                str(path),
                "--request-output",
                str(request),
                "--regions",
                "2",
                "--quarters",
                "16",
            ]
        )
        == 0
    )
    assert len(read_observations(path)) == 32
    assert main(["inspect", "--input", str(path), "--output", str(tmp_path / "quality.json")]) == 0
    assert main(["prepare", "--input", str(path), "--output", str(tmp_path / "features.csv")]) == 0
    assert request.exists()


def test_cli_returns_error_and_does_not_overwrite(tmp_path):
    path = tmp_path / "data.csv"
    assert main(["make-demo", "--output", str(path), "--regions", "1"]) == 0
    original = path.read_bytes()
    assert main(["make-demo", "--output", str(path), "--regions", "2"]) == 2
    assert path.read_bytes() == original
