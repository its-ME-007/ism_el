"""Promote / rollback / log / monitor round trip for the MVP model registry."""

import json

import pytest

from src.tiering.prediction import registry


def test_promote_rollback_and_monitor(tmp_path):
    first, second = tmp_path / "a.joblib", tmp_path / "b.joblib"
    first.write_bytes(b"model-a")
    second.write_bytes(b"model-b")
    root = tmp_path / "models"

    with pytest.raises(RuntimeError):
        registry.active_model(root)
    a = registry.promote(first, root)
    b = registry.promote(second, root)
    assert registry.active_model(root)[0] == b["id"]
    assert registry.rollback(root)["id"] == a["id"]
    active_id, active_path = registry.active_model(root)
    assert active_id == a["id"] and active_path.read_bytes() == b"model-a"

    log = tmp_path / "predictions.jsonl"
    for tier in ("COLD", "COLD", "HOT", "COLD"):
        registry.log_prediction({"document_id": "d", "predicted_tier": tier,
                                 "model": "C_combined",
                                 "probability_any_access_next_24h": 0.5,
                                 "text": "must not be logged"}, active_id, log)
    assert "must not be logged" not in log.read_text(encoding="utf-8")
    reference = {"HOT": 0.25, "WARM": 0.25, "COLD": 0.5}
    report = registry.monitor(log, reference)
    assert report["predictions"] == 4
    assert report["served_tier_share"]["COLD"] == 0.75
    assert report["drift_alert"] is None  # below min_predictions: no alarm
    report = registry.monitor(log, {"C_combined": reference, "D_combined_retrieval": {}},
                              min_predictions=4)
    assert report["reference_run"] == "C_combined" and report["drift_alert"] is True
    report = registry.monitor(log, reference, min_predictions=4)
    assert report["drift_alert"] is True  # WARM 0 vs 0.25, COLD +0.25
    assert json.loads(log.read_text().splitlines()[0])["model_id"] == active_id
