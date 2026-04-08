"""
Phase 3 — Integration: HTTP API wraps ForensicAuditEnv; reset/step/state serialize cleanly.
Run from repo root: PYTHONPATH=. pytest tests/test_phase3_integration.py -v
"""

import pytest
from fastapi.testclient import TestClient

from server.app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_step_requires_reset_first(client):
    r = client.post(
        "/step",
        json={"action_type": "request_category", "params": {"category": "invoices"}},
    )
    assert r.status_code == 400


def test_reset_all_task_ids(client):
    for task_id in ("easy", "medium", "hard"):
        r = client.post("/reset", params={"task_id": task_id})
        assert r.status_code == 200, r.text
        obs = r.json()
        assert obs["step"] == 0
        assert obs["done"] is False
        assert obs["remaining_steps"] >= 1


def test_reset_unknown_task(client):
    r = client.post("/reset", params={"task_id": "invalid"})
    assert r.status_code == 400


def test_e2e_request_category_then_submit_findings(client):
    """reset → request_category → submit_findings (episode ends)."""
    assert client.post("/reset", params={"task_id": "easy"}).status_code == 200

    r = client.post(
        "/step",
        json={"action_type": "request_category", "params": {"category": "invoices"}},
    )
    assert r.status_code == 200
    body = r.json()
    assert "observation" in body and "reward" in body
    assert "terminated" in body and "truncated" in body and "info" in body
    # reward is now scalar float (openM standard)
    assert isinstance(body["reward"], (int, float))
    # reward breakdown moved to info
    assert "reward_breakdown" in body["info"]
    assert body["terminated"] is False
    assert body["truncated"] is False
    assert isinstance(body["info"], dict)

    r = client.get("/state")
    assert r.status_code == 200
    state = r.json()
    assert state["true_fraud_type"] == "channel_stuffing"
    assert "inv_q3_early" in state["document_index"]

    r = client.post(
        "/step",
        json={
            "action_type": "submit_findings",
            "params": {
                "fraud_type": "channel_stuffing",
                "evidence_chain": ["inv_q3_early"],
            },
        },
    )
    assert r.status_code == 200
    assert r.json()["terminated"] is True


def test_cross_reference_after_two_categories(client):
    client.post("/reset", params={"task_id": "easy"})
    client.post(
        "/step",
        json={"action_type": "request_category", "params": {"category": "invoices"}},
    )
    client.post(
        "/step",
        json={
            "action_type": "request_category",
            "params": {"category": "financial_statements"},
        },
    )
    r = client.post(
        "/step",
        json={
            "action_type": "cross_reference",
            "params": {"doc_a": "inv_q3_early", "doc_b": "fin_q3_statement"},
        },
    )
    assert r.status_code == 200
    # reward is scalar float now
    assert r.json()["reward"] > 0


def test_flag_anomaly(client):
    client.post("/reset", params={"task_id": "easy"})
    client.post(
        "/step",
        json={"action_type": "request_category", "params": {"category": "invoices"}},
    )
    r = client.post(
        "/step",
        json={
            "action_type": "flag_anomaly",
            "params": {"doc_id": "inv_q3_early", "description": "timing"},
        },
    )
    assert r.status_code == 200
