#!/usr/bin/env python3
"""Bounded CPU task queue with prospective receipts and identity-checked resume.

Jobs JSON: {"jobs": [{"id": "E01", "command": [...], "cwd": "...",
"inputs": [paths], "outputs": [paths], "dependencies": [], "reserve_bytes": 0,
"memory_bytes": 268435456, "script": optional_guarded_module_path,
"config": optional_guarded_config_path, "exclusive": ["shared-output-key"]}]}.
No shell expansion, model calls, GPU admission, or automatic failed-job retries.
"""
import argparse
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time

import psutil


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(1048576), b""):
            h.update(buf)
    return h.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_queue(spec, root, deadline, max_workers=4, min_free=256 * 1024**2):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    owner = open(root / "queue.lock", "a+")
    try:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        owner.close()
        raise RuntimeError("another queue owns this run root")
    receipt = root / "RECEIPTS_ALL.jsonl"
    lock = threading.Lock()
    reserved = {}

    def emit(record):
        with lock, open(receipt, "a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def stopped():
        return time.time() >= deadline or (root / "STOP_REQUESTED").exists()

    def execute(job, identity):
        attempt = job["id"] + "-" + secrets.token_hex(6)
        token = secrets.token_hex(32)
        command = job["command"]
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1",
                   CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   PHASEDX_RUN_ROOT=str(root), PHASEDX_ATTEMPT_ID=attempt,
                   PHASEDX_TOKEN=token, PHASEDX_RECEIPTS=str(receipt))
        env.update(job.get("env", {}))
        # Job-specific paths are allowed; a job cannot silently relax CPU/thread bounds.
        env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                   VECLIB_MAXIMUM_THREADS="1", CUDA_VISIBLE_DEVICES="")
        register = {"event": "register", "utc": utc(), "job": job["id"],
                    "attempt_id": attempt, "identity": identity, "command": command,
                    "cwd": job.get("cwd"), "launcher_pid": os.getpid(),
                    "token_sha256": hashlib.sha256(token.encode()).hexdigest(),
                    "script_sha256": sha(job["script"]) if job.get("script") else None,
                    "config_sha256": sha(job["config"]) if job.get("config") else None,
                    "device": "CPU"}
        emit(register)
        if stopped():
            emit({"event": "end", "utc": utc(), "job": job["id"], "attempt_id": attempt,
                  "identity": identity, "exit_code": 124, "reason": "stop_before_spawn",
                  "elapsed_seconds": 0, "peak_rss_bytes": 0, "outputs": {}})
            return "stopped"
        start = time.monotonic()
        peak = 0
        reason = ""
        with open(root / "logs" / (attempt + ".stdout"), "wb") as out, open(root / "logs" / (attempt + ".stderr"), "wb") as err:
            p = subprocess.Popen(command, cwd=job.get("cwd"), env=env, stdin=subprocess.DEVNULL,
                                 stdout=out, stderr=err, start_new_session=True)
            proc = psutil.Process(p.pid)
            birth = proc.create_time()
            emit({"event": "start", "utc": utc(), "job": job["id"], "attempt_id": attempt,
                  "pid": p.pid, "create_time": birth, "pgid": p.pid})
            term_time = None
            while p.poll() is None:
                try:
                    members = [proc] + proc.children(recursive=True)
                    rss = sum(x.memory_info().rss for x in members if x.is_running())
                    peak = max(peak, rss)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                timed_out = job.get("max_seconds") is not None and time.monotonic() - start >= job["max_seconds"]
                if stopped() or timed_out or psutil.disk_usage(root).free < min_free:
                    reason = "deadline_or_STOP" if stopped() else "job_timeout" if timed_out else "free_space"
                    if term_time is None:
                        # Popen child, recorded birth and process group; never an account-wide kill.
                        try:
                            if proc.create_time() == birth and os.getpgid(p.pid) == p.pid:
                                os.killpg(p.pid, signal.SIGTERM)
                                term_time = time.monotonic()
                        except ProcessLookupError:
                            pass
                    elif time.monotonic() - term_time > 15:
                        try:
                            if proc.create_time() == birth and os.getpgid(p.pid) == p.pid:
                                os.killpg(p.pid, signal.SIGKILL)
                        except (ProcessLookupError, psutil.NoSuchProcess):
                            pass
                time.sleep(0.1)
            code = p.wait()
        outputs = {str(x): sha(x) for x in job.get("outputs", []) if Path(x).is_file()}
        missing = [str(x) for x in job.get("outputs", []) if not Path(x).is_file()]
        emit({"event": "end", "utc": utc(), "job": job["id"], "attempt_id": attempt,
              "identity": identity, "exit_code": code, "elapsed_seconds": time.monotonic() - start,
              "peak_rss_bytes": peak, "reason": reason, "outputs": outputs, "missing_outputs": missing})
        return "done" if code == 0 and not missing else "failed"

    jobs = spec["jobs"]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("job ids must be unique")
    old = [json.loads(s) for s in receipt.read_text().splitlines()] if receipt.exists() else []
    state = {}
    identities = {}
    def resolve_identity(j):
        files = j.get("inputs", []) + ([j["script"]] if j.get("script") else []) + ([j["config"]] if j.get("config") else [])
        try:
            hashes = {str(x): sha(x) for x in files}
        except FileNotFoundError:
            return "blocked_missing_input"
        identities[j["id"]] = hashlib.sha256(json.dumps({"job": j, "inputs": hashes}, sort_keys=True).encode()).hexdigest()
        good = [r for r in old if r.get("event") == "end" and r.get("job") == j["id"]
                and r.get("identity") == identities[j["id"]] and r.get("exit_code") == 0 and not r.get("missing_outputs")]
        if good and all(Path(p).is_file() and sha(p) == h for p, h in good[-1]["outputs"].items()):
            return "done"
        return "pending"
    # Dependent inputs may be created by an upstream job in this same queue.
    # Resolve their identity only after dependencies have completed.
    for j in jobs:
        state[j["id"]] = "pending"
    running = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            while True:
                for future in list(running):
                    if future.done():
                        jid = running.pop(future)
                        try:
                            state[jid] = future.result()
                        except Exception as e:
                            state[jid] = "failed"
                            emit({"event": "launcher_error", "utc": utc(), "job": jid, "error": repr(e)})
                        reserved.pop(jid, None)
                if stopped():
                    for jid, val in state.items():
                        if val == "pending":
                            state[jid] = "stopped_before_admission"
                if not stopped():
                    for j in jobs:
                        if len(running) >= max_workers:
                            break
                        if state[j["id"]] != "pending":
                            continue
                        deps = j.get("dependencies", [])
                        if any(state.get(d) != "done" for d in deps):
                            continue
                        state[j["id"]] = resolve_identity(j)
                        if state[j["id"]] != "pending":
                            continue
                        occupied = {k for v in reserved.values() for k in v.get("exclusive", [])}
                        if occupied.intersection(j.get("exclusive", [])):
                            continue
                        need_mem = j.get("memory_bytes", 256 * 1024**2)
                        need_disk = j.get("reserve_bytes", 0)
                        if psutil.virtual_memory().available - sum(v.get("memory_bytes", 0) for v in reserved.values()) < need_mem:
                            continue
                        if psutil.disk_usage(root).free - sum(v.get("reserve_bytes", 0) for v in reserved.values()) < min_free + need_disk:
                            continue
                        state[j["id"]] = "running"
                        reserved[j["id"]] = j
                        running[pool.submit(execute, j, identities[j["id"]])] = j["id"]
                (root / "QUEUE_STATE.json").write_text(json.dumps({"utc": utc(), "jobs": state}, indent=2) + "\n")
                if not running:
                    for jid, val in state.items():
                        if val == "pending":
                            state[jid] = "blocked_dependency_or_resource"
                    break
                time.sleep(0.1)
        (root / "QUEUE_STATE.json").write_text(json.dumps({"utc": utc(), "jobs": state}, indent=2) + "\n")
    finally:
        fcntl.flock(owner, fcntl.LOCK_UN)
        owner.close()
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--deadline-utc", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    deadline = dt.datetime.fromisoformat(a.deadline_utc.replace("Z", "+00:00"))
    if deadline.tzinfo is None or a.workers < 1:
        ap.error("timezone-aware deadline and positive worker count required")
    spec = json.loads(Path(a.jobs).read_text())
    if a.dry_run:
        print(json.dumps({"jobs": [j["id"] for j in spec["jobs"]], "deadline": deadline.isoformat()}))
        return 0
    state = run_queue(spec, a.run_root, deadline.timestamp(), a.workers)
    print(json.dumps(state))
    return 0 if all(s == "done" for s in state.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
