from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
from fastapi.testclient import TestClient
from app.main import app

ROOT = Path(__file__).parent
CLEAN_DIR = ROOT / "data" / "train_clean"
ERR_DIR = ROOT / "data" / "test"

client = TestClient(app)


def _upload_files(file_paths: List[Path]) -> List[Tuple[str, Tuple[str, bytes, str]]]:
    return [("files", (p.name, p.read_bytes(), "text/plain")) for p in file_paths]


def test_full_cycle() -> None:
    # ------------------------------------------------------------------ train
    clean_files = sorted(CLEAN_DIR.glob("*.log"))
    resp = client.post(
        "/train",
        files=_upload_files(clean_files),
        params={"contamination": 0.05, "n_estimators": 200},
    )
    assert resp.status_code == 200, resp.text
    model_used = resp.json()["model_path"]

    # ------------------------------------------------------------------ analyse – clean logs
    for p in clean_files:
        resp = client.post(
            "/analyse?mode=local",
            files={"file": (p.name, p.read_bytes(), "text/plain")},
        )
        body = resp.json()
        assert resp.status_code == 200, body
        assert body["model_used"] == Path(model_used).name
        assert body["anomalies"] == [], f"Unexpected anomaly in clean log {p.name!s}"

    # ------------------------------------------------------------------ analyse – error logs
    err_files = sorted(ERR_DIR.glob("*.log"))
    for p in err_files:
        resp = client.post(
            "/analyse?mode=local&classify=true",
            files={"file": (p.name, p.read_bytes(), "text/plain")},
        )
        body = resp.json()
        assert resp.status_code == 200, body
        assert len(body["anomalies"]) == 1, f"{p.name} should have exactly 1 anomaly"
        assert body["classifications"], (
            f"{p.name} should return at least one classification label"
        )
