"""Live dashboard for training and evals on both Macs: http://localhost:8765

Every POLL_S seconds runs collector.py here and on the worker (over ssh), merges the two
snapshots and serves them at /api/status; index.html polls that and redraws.

Usage: python dashboard/server.py [--port 8765]
"""
import argparse
import json
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent
POLL_S = 5
PYTHON = "/Users/mohammed/miniconda3/envs/em-mlx/bin/python"
MACHINES = {
    "main": [PYTHON, str(HERE / "collector.py")],
    "worker": ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", "worker", f"{PYTHON} -"],
}
STATE = {"machines": {}, "logs": [], "results": [], "time": None}
LOCK = threading.Lock()


def collect(name, cmd):
    src = (HERE / "collector.py").read_text() if name == "worker" else None
    try:
        r = subprocess.run(cmd, input=src, capture_output=True, text=True, timeout=30)
        snap = json.loads(r.stdout)
        snap["ok"] = True
    except Exception as e:
        snap = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"}
    snap["name"], snap["polled"] = name, time.time()
    return snap


def rank(log):
    # the same log name can exist on both Macs (worker.sh push/fetch copy logs): prefer the live one
    return (log["status"] == "running", log.get("done_n") or 0, log["mtime"])


def merge(snaps):
    logs, results = {}, {}
    for name, s in snaps.items():
        for log in s.get("logs", []):
            log["machine"] = name
            if log["name"] not in logs or rank(log) > rank(logs[log["name"]]):
                logs[log["name"]] = log
        for res in s.get("results", []):
            res["machine"] = name
            old = results.get(res["path"])
            if not old or (res["scored"], res["mtime"]) > (old["scored"], old["mtime"]):
                results[res["path"]] = res
    # fixed order (main, worker): snaps fills in thread-completion order, which made the cards swap places
    machines = {n: {k: snaps[n].get(k) for k in ("ok", "error", "machine", "procs", "polled", "name")}
                for n in MACHINES if n in snaps}
    return {"machines": machines, "logs": sorted(logs.values(), key=lambda l: -l["mtime"]),
            "results": sorted(results.values(), key=lambda r: -r["mtime"]), "time": time.time()}


def poll_forever():
    while True:
        threads, snaps = [], {}
        for name, cmd in MACHINES.items():
            t = threading.Thread(target=lambda n=name, c=cmd: snaps.__setitem__(n, collect(n, c)))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        merged = merge(snaps)
        with LOCK:
            STATE.update(merged)
        time.sleep(POLL_S)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/status"):
            with LOCK:
                body, ctype = json.dumps(STATE).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = (HERE / "index.html").read_bytes(), "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    port = ap.parse_args().port
    threading.Thread(target=poll_forever, daemon=True).start()
    print(f"dashboard on http://localhost:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
