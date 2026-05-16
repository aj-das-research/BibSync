"""End-to-end server tests.

The unit tests in this file run the FastAPI app via ``fastapi.testclient.TestClient``
(in-process, no port binding). They verify each endpoint's contract, not the
correctness of the AI pipeline itself — that's covered by the benchmark.

Pytest markers:
  - ``@pytest.mark.live``: requires an LLM API key and network access.
    Skipped by default. Run with ``pytest -m live``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip the whole module if FastAPI is missing — keeps the test suite green
# for users who install bibsync without the [server] extras.
fastapi = pytest.importorskip("fastapi")
testclient = pytest.importorskip("fastapi.testclient")

from bibsync import patches  # noqa: E402
from bibsync.server import create_app  # noqa: E402

TOKEN = "test-token-pytest"


@pytest.fixture
def client():
    app = create_app(token=TOKEN)
    return testclient.TestClient(app)


@pytest.fixture
def auth():
    return {"Authorization": f"Bearer {TOKEN}"}


# ── auth ────────────────────────────────────────────────────────────────────


def test_health_requires_auth(client):
    assert client.get("/health").status_code == 401


def test_health_rejects_wrong_token(client):
    r = client.get("/health", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 403


def test_health_accepts_correct_token(client, auth):
    r = client.get("/health", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "version" in body
    assert "ts" in body


# ── patches ────────────────────────────────────────────────────────────────


def test_patch_preview_renders_diff(client, auth):
    p = patches.Patch.new(
        type="raw", file="main.tex", start=0, end=5,
        old_text="Hello", new_text="Greetings",
    )
    r = client.post(
        "/patch/preview", headers=auth,
        json={"patches": [p.to_dict()], "files": {"main.tex": "Hello world."}},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["ok"]
    pv = body["preview"]["main.tex"]
    assert pv["before"] == "Hello world."
    assert pv["after"] == "Greetings world."
    assert "-Hello world." in pv["diff_unified"]
    assert "+Greetings world." in pv["diff_unified"]


def test_patch_apply_rejects_unapproved(client, auth):
    p = patches.Patch.new(
        type="raw", file="main.tex", start=0, end=5,
        old_text="Hello", new_text="Greetings",
    )
    r = client.post(
        "/patch/apply", headers=auth,
        json={"patches": [p.to_dict()], "files": {"main.tex": "Hello world."}},
    )
    body = r.json()
    assert body["ok"] is False
    assert "not user-approved" in body["errors"][0]


def test_patch_apply_succeeds_when_approved(client, auth):
    p = patches.Patch.new(
        type="raw", file="main.tex", start=0, end=5,
        old_text="Hello", new_text="Greetings",
    )
    p.user_approved = True
    r = client.post(
        "/patch/apply", headers=auth,
        json={"patches": [p.to_dict()], "files": {"main.tex": "Hello world."}},
    )
    body = r.json()
    assert body["ok"], body.get("errors")
    assert body["files"]["main.tex"] == "Greetings world."
    assert body["applied"] == [p.patch_id]


def test_patch_apply_atomic_under_conflict(client, auth):
    # Two patches; one has a stale old_text. Both should be rejected.
    p1 = patches.Patch.new(
        type="raw", file="main.tex", start=0, end=5,
        old_text="Hello", new_text="Greetings",
    )
    p2 = patches.Patch.new(
        type="raw", file="main.tex", start=6, end=11,
        old_text="WRONG", new_text="UNIVERSE",  # mismatched old_text
    )
    p1.user_approved = True
    p2.user_approved = True
    r = client.post(
        "/patch/apply", headers=auth,
        json={"patches": [p1.to_dict(), p2.to_dict()],
              "files": {"main.tex": "Hello world."}},
    )
    body = r.json()
    assert body["ok"] is False
    assert len(body["conflicts"]) == 1
    # No partial mutation
    assert body["files"]["main.tex"] == "Hello world."
    assert body["applied"] == []


# ── memory / cache / privacy ────────────────────────────────────────────────


def test_memory_listing_works(client, auth, tmp_path):
    r = client.get(
        "/memory",
        headers=auth,
        params={"project_root": str(tmp_path), "scope": "project"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "records" in body
    assert "total" in body
    assert body["total"] == 0  # fresh tmp dir


def test_cache_status_returns_subdirs(client, auth):
    r = client.get("/cache/status", headers=auth)
    assert r.status_code == 200
    body = r.json()
    assert "subdirs" in body
    for sub in ("paper_content", "pdfs", "embeddings", "memory"):
        assert sub in body["subdirs"]


def test_cache_clear_refuses_memory(client, auth):
    r = client.post("/cache/clear", headers=auth, params={"target": "memory"})
    assert r.status_code == 400


def test_memory_remember_forget_roundtrip(client, auth):
    """A record written via /memory/remember is listed by /memory and
    disappears after /memory/forget — the full Sprint-F memory loop."""
    pid = "test-sprint-f-project"
    # Write an override record (the "Ignore warning" action).
    r = client.post(
        "/memory/remember", headers=auth,
        json={
            "project_id": pid,
            "type": "override",
            "claim_text": "GPT-3 achieves 86.5% on MedQA",
            "paper_key": "arxiv-2005.14165",
            "decision": "user_ignored",
            "scope": "project",
        },
    )
    assert r.status_code == 200, r.text
    rec = r.json()["record"]
    assert rec is not None
    rid = rec["id"]

    # It should now appear in /memory for this project.
    r = client.get("/memory", headers=auth,
                    params={"project_id": pid, "scope": "project"})
    ids = [rr["id"] for rr in r.json()["records"]]
    assert rid in ids

    # Forget it.
    r = client.post("/memory/forget", headers=auth,
                    json={"record_id": rid, "scope": "project", "project_id": pid})
    assert r.json()["ok"] is True

    # Gone from /memory.
    r = client.get("/memory", headers=auth,
                    params={"project_id": pid, "scope": "project"})
    ids = [rr["id"] for rr in r.json()["records"]]
    assert rid not in ids

    # Clean up the project file.
    client.request("DELETE", "/memory/project", headers=auth,
                    params={"project_id": pid})


def test_privacy_endpoint(client, auth, tmp_path):
    r = client.get("/privacy", headers=auth, params={"project_root": str(tmp_path)})
    assert r.status_code == 200
    body = r.json()
    assert body["memory"]["project_records"] == 0
    assert body["caches_root"] is not None


# ── OpenAPI ────────────────────────────────────────────────────────────────


def test_openapi_exposes_all_endpoints(client):
    r = client.get("/openapi.json")
    assert r.status_code in (200, 401)
    if r.status_code == 200:
        paths = r.json().get("paths", {})
        for endpoint in (
            "/health", "/audit", "/evidence", "/source-rank",
            "/patch/preview", "/patch/apply",
            "/memory", "/memory/forget", "/memory/project",
            "/cache/status", "/cache/clear", "/privacy",
        ):
            assert endpoint in paths, f"missing {endpoint}"
