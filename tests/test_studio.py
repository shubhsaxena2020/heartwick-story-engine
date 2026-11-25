"""Heartwick Studio laptop side: snapshot builder, self-check, actions, live channel, queue runner, media sync."""
import hashlib, hmac, importlib, json, os, sys, threading, time, urllib.error, urllib.request

import pytest
from conftest import ROOT

sys.path.insert(0, os.path.join(ROOT, "dashboard"))
sys.path.insert(0, os.path.join(ROOT, "tools"))


@pytest.fixture
def studio(tmp_path, monkeypatch):
    """studio_common paths pointed at a temp data dir (the real stories/out/cache stay read-only inputs)."""
    import studio_common as C
    for name in ("APPROVALS", "APPLIED", "KAGGLE_LIVE", "MEDIA_INDEX", "SNAPSHOT_LOCAL", "QUEUE"):
        monkeypatch.setattr(C, name, str(tmp_path / os.path.basename(getattr(C, name))))
    monkeypatch.setattr(C, "DATA", str(tmp_path))
    import actions as A, media as M, studio_state as S
    A, M, S = importlib.reload(A), importlib.reload(M), importlib.reload(S)
    monkeypatch.setattr(S, "TRUTH_CACHE", str(tmp_path / "truth_cache.json"))
    monkeypatch.setattr(M, "PREVIEWS", str(tmp_path / "previews"))
    return C, A, M, S, tmp_path


def test_snapshot_from_real_files(studio):
    C, A, M, S, tmp = studio
    snap, previews = S.build(M.Index())
    assert snap["schema"] == 1 and snap["stages"][-1] == "published"
    assert {s["slug"] for s in snap["stories"]} >= {"sample"}
    names = {v["name"] for v in snap["videos"]}
    assert names == {d for d in os.listdir(os.path.join(ROOT, "out")) if os.path.isfile(os.path.join(ROOT, "out", d, "plan.json"))}
    for v in snap["videos"]:
        assert v["stage"] in snap["stages"]
        p = v["progress"]
        assert 0 <= p["lines_done"] <= p["lines_total"] and 0 <= p["images_done"] <= p["images_total"]
        order = [v["reached"][s] for s in ("narration", "images", "render", "review") if s in v["reached"]]
        assert order == sorted(order)                       # stages are entered in order
        if os.path.isfile(os.path.join(ROOT, "out", v["name"], v["name"] + ".mp4")):
            assert v["stage"] in ("render", "review", "approved", "published")
            assert v["reached"]["render"] >= v["reached"].get("narration", "")
    assert sum(snap["counts"].values()) == len(snap["videos"])
    json.dumps(snap)  # serialisable


def test_self_check_recomputes_and_flags(studio):
    C, A, M, S, tmp = studio
    snap, _ = S.build(M.Index())
    t = S.self_check(snap)
    assert t["n_checks"] >= 4 and {"stories", "rendered", "reviewable", "approved"} <= {c["key"] for c in t["checks"]}
    assert all(c["ok"] is not False for c in t["checks"] if c["key"] in ("stories", "rendered", "reviewable", "approved"))
    rendered = [v for v in snap["videos"] if v["stage"] in ("render", "review", "approved", "published")]
    if rendered:                                           # a shown number that no longer matches the files is flagged
        rendered[0]["stage"] = "script"
        t2 = S.self_check(snap)
        assert next(c for c in t2["checks"] if c["key"] == "rendered")["ok"] is False and t2["n_fail"] >= 1


def test_kaggle_jobs_waits_and_transitions(studio, monkeypatch):
    C, A, M, S, tmp = studio
    usage = tmp / "usage.jsonl"
    rows = [{"slug": "u/heartwick-gpu-x", "gpu": True, "pushed": "2026-11-21T10:00:00", "state": "pushed"},
            {"slug": "u/heartwick-gpu-x", "gpu": True, "pushed": "2026-11-21T10:00:00", "state": "complete", "wall_min": 40.0, "run_min_est": 15.0},
            {"slug": "u/heartwick-gpu-y", "gpu": True, "pushed": "2026-11-21T11:00:00", "state": "pushed"}]
    usage.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    monkeypatch.setattr(S, "USAGE", str(usage))
    qlog = [{"at": C.local_iso_to_epoch("2026-11-21T11:20:00"), "text": "heartwick-gpu-y -> RUNNING", "body": []}]
    jobs = {j["kernel"]: j for j in S._jobs(qlog, {}, [], [], C.local_iso_to_epoch("2026-11-21T11:30:00"))}
    assert jobs["heartwick-gpu-x"]["state"] == "complete" and jobs["heartwick-gpu-x"]["wait_min"] == 25.0
    assert jobs["heartwick-gpu-y"]["state"] == "running" and jobs["heartwick-gpu-y"]["wait_min"] == 20.0 and jobs["heartwick-gpu-y"]["run_min"] == 10.0


def test_actions_validate_apply_once(studio):
    C, A, M, S, tmp = studio
    vid = next((d for d in os.listdir(os.path.join(ROOT, "out")) if os.path.isfile(os.path.join(ROOT, "out", d, "plan.json"))), None)
    if vid is None:
        pytest.skip("no rendered videos in out/ (removed from the public snapshot)")
    a = {"id": "abcdef12-0000", "kind": "needs_changes", "video": vid, "note": "slower captions", "at": "2026-11-21T12:00:00Z"}
    assert A.validate(a) is None
    assert A.validate({**a, "video": "../../etc"}) == "unknown video"
    assert A.validate({**a, "kind": "ship it"}) == "unknown kind"
    assert A.validate({**a, "id": "x"}) == "bad id"
    r1 = A.apply([a], "tunnel")
    r2 = A.apply([a], "vercel inbox")                       # same id from the other path: applied once
    assert r1[0]["result"] == "ok" and r2 == []
    ap = json.load(open(C.APPROVALS, encoding="utf-8"))
    assert ap[vid]["status"] == "needs_changes" and ap[vid]["note"] == "slower captions"
    bad = A.apply([{**a, "id": "abcdef12-0001", "video": "nope_nope"}], "tunnel")
    assert bad[0]["result"] == "rejected"


def test_queue_story_and_rerender_args(studio, monkeypatch):
    C, A, M, S, tmp = studio
    monkeypatch.setattr(A, "start_runner_if_needed", lambda log=None: None)
    args, name = A.queue_story_args("sample", "short")
    assert args[:3] == ["make", "stories/sample.md", "--short"] and name == "sample_short"
    res = A.apply([{"id": "q1234567-1", "kind": "queue_story", "story": "sample", "mode": "short", "at": "x"}], "tunnel")
    assert res[0]["result"] == "ok"
    q = json.load(open(C.QUEUE, encoding="utf-8"))
    assert q["jobs"][0]["status"] == "pending" and "--auto-pick" in q["jobs"][0]["args"]
    if os.path.isfile(os.path.join(ROOT, "out", "pip_pilot_v2", "plan.json")):
        assert A.make_args_for("pip_pilot_v2") == ["make", "stories/pip.md", "--name", "pip_pilot_v2", "--short", "--sections", "1-2"]


def test_queue_runner_never_repushes_interrupted(studio, monkeypatch):
    C, A, M, S, tmp = studio
    import run_queue as R
    R = importlib.reload(R)
    monkeypatch.setattr(R, "QUEUE", C.QUEUE)
    monkeypatch.setattr(R, "LOG", str(tmp / "queue_runner.log"))
    C.write_json(C.QUEUE, {"jobs": [{"id": "a", "status": "running", "runner_pid": 999999, "args": []},
                                    {"id": "b", "status": "pending", "args": ["status"]}]})
    job = R.take()
    q = json.load(open(C.QUEUE, encoding="utf-8"))
    assert q["jobs"][0]["status"] == "interrupted" and job["id"] == "b" and q["jobs"][1]["status"] == "running"
    assert R.take() is None


def _token(secret, exp):
    msg = f"hw1.{int(exp)}"
    return msg + "." + hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def test_live_token_rules():
    import live
    s = "sekrit"
    now = time.time()
    assert live.verify_token(_token(s, now + 300), s, now)
    assert not live.verify_token(_token(s, now - 1), s, now)            # expired
    assert not live.verify_token(_token(s, now + 3600), s, now)         # too far ahead
    assert not live.verify_token(_token("other", now + 300), s, now)
    assert not live.verify_token("hw1.123.zz", s, now)


def test_live_server_auth_origin_traversal(studio, monkeypatch):
    C, A, M, S, tmp = studio
    import live
    live = importlib.reload(live)
    from http.server import ThreadingHTTPServer
    st = live.State()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), live.make_handler(st, "sekrit"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    tok = _token("sekrit", time.time() + 300)

    def get(path, headers=None):
        try:
            with urllib.request.urlopen(urllib.request.Request(base + path, headers=headers or {}), timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
    try:
        assert get("/health")[0] == 200
        assert get("/state?t=nope")[0] == 401
        assert get(f"/state?t={tok}", {"Origin": "https://evil.example"})[0] == 403
        code, body = get(f"/state?t={tok}", {"Origin": "https://heartwick-studio.vercel.app"})
        j = json.loads(body)
        assert code == 200 and j["snapshot"]["schema"] == 1 and j["same"] is False
        code, body = get(f"/state?t={tok}&v={j['rev']}")
        assert json.loads(body)["same"] is True and "snapshot" not in json.loads(body)
        assert get(f"/media/..%2F..%2Fheartwick.py?t={tok}")[0] == 404
        assert get(f"/media/0123456789ab/nothing.jpg?t={tok}")[0] == 404
    finally:
        srv.shutdown()


def test_media_sync_uploads_only_new_and_deletes_unused(studio, monkeypatch):
    C, A, M, S, tmp = studio
    f1, f2 = tmp / "a.jpg", tmp / "b.jpg"
    f1.write_bytes(b"one")
    f2.write_bytes(b"two")
    idx = M.Index()
    k1, k2 = idx.ref(str(f1))["key"], idx.ref(str(f2))["key"]
    calls = []
    monkeypatch.setattr(M, "blob_call", lambda ops, token, log=None: calls.append(ops) or [{"ok": True} for _ in ops])
    r = M.sync(idx, [k1, k2], "tok")
    assert r["uploaded"] == 2 and len(calls[0]) == 2
    r = M.sync(idx, [k1, k2], "tok")                          # nothing changed: no Blob call at all
    assert r["uploaded"] == 0 and len(calls) == 1
    f1.write_bytes(b"one, edited")
    k1b = idx.ref(str(f1))["key"]
    r = M.sync(idx, [k1b, k2], "tok")                         # changed file: one upload + delete of the old key
    ops = calls[-1]
    assert [o["op"] for o in ops] == ["put", "del"] and ops[1]["pathnames"] == [f"m/{k1}"]
    assert k1b != k1


def test_exporter_fingerprint_ignores_timers(studio):
    C, A, M, S, tmp = studio
    import exporter as E
    snap, _ = S.build(M.Index())
    snap["truth"] = {"n_fail": 0}
    imp1, minor1 = E.Exporter.fingerprints(snap)
    snap["generated_at"] = "later"
    assert E.Exporter.fingerprints(snap) == (imp1, minor1)
    if snap["videos"]:
        snap["videos"][0]["stage"] = "published"
        assert E.Exporter.fingerprints(snap)[0] != imp1


def test_quota_and_status_parsing():
    import exporter as E
    q = E.parse_quota("resource  used   remaining  total   refreshAt\n--------  -----\nGPU       0.87h  29.13h     30.00h  2026-11-28T00:00:00+00:00\n")
    assert q["used_h"] == 0.87 and q["total_h"] == 30.0 and q["refresh_at"].startswith("2026-11-28")
    assert E.parse_status('shubh/heartwick-gpu-pip has status "KernelWorkerStatus.RUNNING"') == "RUNNING"
