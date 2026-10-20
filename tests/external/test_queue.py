import importlib.util
import json
from pathlib import Path
import sys
import threading
import time

import psutil

spec = importlib.util.spec_from_file_location("queue_impl", Path(__file__).resolve().parents[2] / "tools/run_external.py")
Q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(Q)


def job(jid, command, **kwargs):
    return dict(id=jid, command=[sys.executable, "-c", command], memory_bytes=1048576, **kwargs)


def test_parallel_dependency_and_identity_resume(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    jobs = {"jobs": [
        job("a", f"import time; from pathlib import Path; time.sleep(.4); Path({str(a)!r}).write_text('A')", outputs=[str(a)]),
        job("b", f"import time; from pathlib import Path; time.sleep(.4); Path({str(b)!r}).write_text('B')", outputs=[str(b)]),
        job("c", f"from pathlib import Path; assert Path({str(a)!r}).read_text()=='A'",
            inputs=[str(a)], dependencies=["a"]),
    ]}
    root = tmp_path / "run"
    assert Q.run_queue(jobs, root, time.time()+15, max_workers=2, min_free=0) == dict(a="done", b="done", c="done")
    rows = [json.loads(x) for x in (root / "RECEIPTS_ALL.jsonl").read_text().splitlines()]
    events = {(r["job"], r["event"]): r for r in rows}
    assert events["a", "start"]["utc"] < events["b", "end"]["utc"]
    assert events["b", "start"]["utc"] < events["a", "end"]["utc"]
    assert events["a", "end"]["utc"] <= events["c", "start"]["utc"]
    before = (root / "RECEIPTS_ALL.jsonl").read_bytes()
    assert Q.run_queue(jobs, root, time.time()+15, max_workers=2, min_free=0) == dict(a="done", b="done", c="done")
    assert (root / "RECEIPTS_ALL.jsonl").read_bytes() == before


def test_stop_ends_owned_child_and_admits_no_more(tmp_path):
    root = tmp_path / "run"
    def stop_when_started():
        for _ in range(100):
            p = root / "RECEIPTS_ALL.jsonl"
            if p.exists() and '"event": "start"' in p.read_text():
                (root / "STOP_REQUESTED").write_text("synthetic STOP test")
                return
            time.sleep(.02)
    t = threading.Thread(target=stop_when_started)
    t.start()
    result = Q.run_queue({"jobs": [job("sleep", "import time; time.sleep(30)"),
                                 job("never", "raise RuntimeError('must not run')")]},
                         root, time.time()+10, max_workers=1, min_free=0)
    t.join()
    assert result == {"sleep": "failed", "never": "stopped_before_admission"}
    rows = [json.loads(x) for x in (root / "RECEIPTS_ALL.jsonl").read_text().splitlines()]
    start = next(r for r in rows if r["event"] == "start")
    assert not psutil.pid_exists(start["pid"])
    assert next(r for r in rows if r["event"] == "end")["reason"] == "deadline_or_STOP"


def test_expired_queue_never_spawns(tmp_path):
    out = tmp_path / "unexpected"
    r = Q.run_queue({"jobs": [job("never", f"open({str(out)!r},'w').close()")]},
                    tmp_path / "run", time.time()-1, min_free=0)
    assert r == {"never": "stopped_before_admission"}
    assert not out.exists()
