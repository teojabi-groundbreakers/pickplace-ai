"""Exercise a saved SYNTHETIC bundle with TestClient, not an AWS/BE integration test."""

import argparse
import json
from pathlib import Path

from fastapi.testclient import TestClient

from pickplace_ai.api import Settings, create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument("--request", type=Path, default=Path("examples/demo-request.json"))
    parser.add_argument("--output", type=Path, default=Path("examples/demo-response.json"))
    args = parser.parse_args()
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    if payload["source_kind"] != "synthetic":
        raise ValueError("This smoke script is only for explicitly synthetic test fixtures")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
    app = create_app(
        Settings(
            model_dir=args.model_dir,
            allow_synthetic=True,
            allow_experimental=True,
        )
    )
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["growth_model_ready"] is True, health.json()
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text
        actual = response.json()
        assert actual["source_kind"] == "synthetic"
        assert actual["growth"]["status"] == "available", actual
        assert actual["risk"]["policy_status"] == "draft"
        assert actual["fit"]["policy_status"] == "draft"
    with TestClient(create_app(Settings(model_dir=args.model_dir))) as client:
        assert client.get("/health").json()["growth_model_ready"] is False
        assert client.post("/predict", json=payload).status_code == 403
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source_kind": actual["source_kind"],
                "growth_status": actual["growth"]["status"],
                "target_period": actual["growth"]["target_period"],
                "explanation_count": len(actual["growth"].get("explanations", [])),
                "production_defaults_block_synthetic": True,
                "output": str(args.output),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
