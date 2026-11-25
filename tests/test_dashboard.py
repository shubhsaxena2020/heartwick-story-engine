import base64, importlib, json, os, sys
import pytest
from conftest import ROOT

pytest.importorskip("httpx")
from fastapi.testclient import TestClient


def _load(monkeypatch, tmp_path, password=""):
    monkeypatch.setenv("HEARTWICK_DASH_PASSWORD", password)
    sys.path.insert(0, os.path.join(ROOT, "dashboard"))
    import app as dash
    dash = importlib.reload(dash)
    monkeypatch.setattr(dash, "DATA", str(tmp_path))
    monkeypatch.setattr(dash, "APPROVALS", str(tmp_path / "approvals.json"))
    monkeypatch.setattr(dash, "_kaggle_live", lambda: {})
    return dash


def test_state_and_page(monkeypatch, tmp_path):
    dash = _load(monkeypatch, tmp_path)
    c = TestClient(dash.app)
    s = c.get("/api/state").json()
    assert s["stages"] == ["script", "narration", "images", "render", "review", "approved"]
    assert {x["slug"] for x in s["stories"]} >= {"sample"}
    assert "quota_minutes" in s["kaggle"]
    for v in s["videos"]:
        assert v["stage"] in s["stages"]
    assert "Heartwick" in c.get("/").text


def test_approve_roundtrip_and_validation(monkeypatch, tmp_path):
    dash = _load(monkeypatch, tmp_path)
    c = TestClient(dash.app)
    vids = [n for n in os.listdir(os.path.join(ROOT, "out")) if os.path.exists(os.path.join(ROOT, "out", n, "plan.json"))]
    if not vids:
        pytest.skip("no videos yet")
    r = c.post("/api/approve", json={"name": vids[0], "status": "needs_changes", "note": "slower captions please"})
    assert r.status_code == 200
    saved = json.load(open(tmp_path / "approvals.json"))
    assert saved[vids[0]]["status"] == "needs_changes" and saved[vids[0]]["note"] == "slower captions please"
    assert c.post("/api/approve", json={"name": vids[0], "status": "ship it"}).status_code == 400
    assert c.post("/api/approve", json={"name": "../../etc", "status": "approved"}).status_code == 400


def test_no_path_traversal(monkeypatch, tmp_path):
    dash = _load(monkeypatch, tmp_path)
    c = TestClient(dash.app)
    assert c.get("/files/pip_pilot/..%2F..%2Fheartwick.py").status_code == 404
    assert c.get("/files/..%2F..%2Fstories/sample.md").status_code == 404
    assert c.get("/fonts/..%2Fbrand.py").status_code == 404


def test_password_required_when_set(monkeypatch, tmp_path):
    dash = _load(monkeypatch, tmp_path, password="s3cret")
    c = TestClient(dash.app)
    assert c.get("/api/state").status_code == 401
    assert c.get("/files/pip_pilot/plan.json").status_code == 401
    bad = base64.b64encode(b"x:nope").decode()
    assert c.get("/api/state", headers={"Authorization": "Basic " + bad}).status_code == 401
    good = base64.b64encode(b"dad:s3cret").decode()
    assert c.get("/api/state", headers={"Authorization": "Basic " + good}).status_code == 200


def test_page_script_has_no_raw_newlines_in_string_literals():
    """Regression: an escaped \n turned into a literal newline inside a JS string and broke the whole page."""
    import re
    html = open(os.path.join(ROOT, "dashboard", "index.html"), encoding="utf-8").read()
    js = html.split("<script>")[1].split("</script>")[0]
    for line in js.splitlines():
        code = re.sub(r"`[^`]*`", "", line)  # template literals may span lines
        assert code.count('"') % 2 == 0 or "`" in line, line
