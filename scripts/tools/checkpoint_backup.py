"""Back up training checkpoints from the GPU server to this machine and keep only the latest one on the server
(runs on the desktop, in its own tmux session; Python standard library + ssh/rsync).

Every --interval seconds, for each run folder outputs/<prefix>*/checkpoints on the server:
  1. A checkpoint is complete when `last` points to it or to a newer one, and (for the one `last` points to) nothing
     in it has changed for --settle_minutes. A folder newer than `last` is still being written and is left alone.
  2. Each complete checkpoint not yet backed up is copied with rsync to <dest>/<run>/<step>/ (if there is room:
     its size + --min_free_gb must be free here), then every file's md5 is compared on both machines.
     Only a fully matching copy gets a `.backup_verified.json` marker.
  3. On the server, checkpoints older than `last` that have a verified copy here are deleted, so the server keeps
     one checkpoint per run (the latest, needed to resume). Nothing else on the server is ever deleted.
Logs to <dest>/backup.log; exits when every run has finished (TRAIN_EXIT in its log) and all is backed up, or after
--max_hours.

Usage (from ~/trumi):
    tmux new-session -d -s trumi-backup "python3 scripts/tools/checkpoint_backup.py"
"""

import argparse
import datetime
import json
import pathlib
import re
import shutil
import subprocess
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
SAFE_DELETE = re.compile(r"^/home/ubuntu/ur5e-lerobot/outputs/pi05_trumi_conveyor_[A-Za-z0-9_]+/checkpoints/\d{6}$")


def ssh(host, cmd, check=True):
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, cmd], capture_output=True, text=True, timeout=3600)
    if check and r.returncode:
        raise RuntimeError(f"ssh failed ({r.returncode}): {cmd[:80]}: {r.stderr.strip()[:200]}")
    return r.stdout


class Backup:
    def __init__(self, a):
        self.a, self.dest = a, a.dest
        self.dest.mkdir(parents=True, exist_ok=True)

    def log(self, msg):
        line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
        print(line, flush=True)
        with open(self.dest / "backup.log", "a") as fh:
            fh.write(line + "\n")

    def remote_state(self):
        """{run folder: {"last": step or None, "steps": {step: (bytes, newest mtime)}}} for all matching runs."""
        out = ssh(self.a.host, f"cd {self.a.outputs} && for r in {self.a.prefix}*/; do r=${{r%/}}; c=$r/checkpoints; "
                               f"[ -d $c ] || continue; echo RUN $r $(readlink $c/last); for s in $c/[0-9]*/; do s=${{s%/}}; "
                               f"echo STEP $r $(basename $s) $(du -sb $s | cut -f1) $(find $s -type f -printf '%T@\\n' | sort -n | tail -1); done; done")
        runs = {}
        for line in out.splitlines():
            f = line.split()
            if f[0] == "RUN":
                runs[f[1]] = {"last": f[2] if len(f) > 2 else None, "steps": {}}
            elif f[0] == "STEP" and len(f) >= 5:
                runs[f[1]]["steps"][f[2]] = (int(f[3]), float(f[4]))
        return runs

    def md5s(self, files_cmd_dir, remote):
        cmd = f"cd {files_cmd_dir} && find . -type f ! -name .backup_verified.json -exec md5sum {{}} + | sort -k2"
        text = ssh(self.a.host, cmd) if remote else subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout
        return dict((l.split(None, 1)[1], l.split(None, 1)[0]) for l in text.splitlines() if l.strip())

    def backup(self, run, step, size):
        local = self.dest / run / step
        marker = local / ".backup_verified.json"
        if marker.exists():
            return True
        free = shutil.disk_usage(self.dest).free
        if free < size + self.a.min_free_gb * 1e9:
            self.log(f"ALERT not enough room here for {run}/{step} ({size / 1e9:.0f} GB + {self.a.min_free_gb} GB margin, "
                     f"{free / 1e9:.0f} GB free): not copied, so it stays on the server")
            return False
        local.mkdir(parents=True, exist_ok=True)
        remote = f"{self.a.outputs}/{run}/checkpoints/{step}"
        self.log(f"copying {run}/{step} ({size / 1e9:.1f} GB)")
        t0 = time.time()
        r = subprocess.run(["rsync", "-a", "--partial", f"{self.a.host}:{remote}/", f"{local}/"], capture_output=True, text=True)
        if r.returncode:
            self.log(f"ERROR rsync {run}/{step} failed: {r.stderr.strip()[:200]}")
            return False
        ours, theirs = self.md5s(local, False), self.md5s(remote, True)
        if ours != theirs or not ours:
            bad = sorted(set(ours.items()) ^ set(theirs.items()))[:3]
            self.log(f"ERROR verification failed for {run}/{step}: {len(ours)} files here vs {len(theirs)} there, first differences {bad}")
            return False
        json.dump({"run": run, "step": step, "bytes": size, "files": ours, "verified": datetime.datetime.now().isoformat()},
                  open(marker, "w"), indent=1)
        self.log(f"BACKED UP {run}/{step}: {len(ours)} files, md5 identical, {(time.time() - t0) / 60:.0f} min")
        return True

    def prune(self, run, step):
        path = f"{self.a.outputs}/{run}/checkpoints/{step}"
        if not SAFE_DELETE.match(path):
            self.log(f"REFUSED to delete {path}: not a trumi checkpoint folder")
            return
        ssh(self.a.host, f"rm -rf '{path}'")
        self.log(f"PRUNED {run}/{step} on the server (verified copy here)")

    def finished(self, runs):
        """The server-side watchdog (train_watchdog.py) logs 'watchdog done' once training is over."""
        return ssh(self.a.host, f"grep -c 'watchdog done' {self.a.watchdog_log} 2>/dev/null", check=False).strip() not in ("", "0")

    def run(self):
        self.log(f"backup job started: {self.a.host}:{self.a.outputs}/{self.a.prefix}* -> {self.dest}")
        t_end = time.time() + self.a.max_hours * 3600
        while time.time() < t_end:
            try:
                runs = self.remote_state()
                now = time.time()
                pending = 0
                for run, info in sorted(runs.items()):
                    last = info["last"]
                    for step, (size, newest) in sorted(info["steps"].items()):
                        if last is None or step > last:
                            continue  # still being written
                        if step == last and now - newest < self.a.settle_minutes * 60:
                            pending += 1
                            continue
                        if not self.backup(run, step, size):
                            pending += 1
                            continue
                        if step < last:
                            self.prune(run, step)
                state = "; ".join(f"{r}: last {i['last']}, on server {sorted(i['steps'])}" for r, i in sorted(runs.items()))
                self.log(f"status: {state or 'no checkpoints yet'} | free here {shutil.disk_usage(self.dest).free / 1e9:.0f} GB")
                if self.a.once:
                    return
                if self.finished(runs) and pending == 0:
                    self.log("all runs finished and every checkpoint is backed up: done")
                    return
            except Exception as e:  # network hiccups etc.: try again next round
                self.log(f"ERROR {type(e).__name__}: {e}")
            time.sleep(self.a.interval)
        self.log("max hours reached: stopping")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Back up training checkpoints here and prune old ones on the server.")
    ap.add_argument("--host", default="zerogrid2")
    ap.add_argument("--outputs", default="/home/ubuntu/ur5e-lerobot/outputs")
    ap.add_argument("--prefix", default="pi05_trumi_conveyor_")
    ap.add_argument("--dest", type=pathlib.Path, default=REPO / "data" / "checkpoints")
    ap.add_argument("--interval", type=int, default=600)
    ap.add_argument("--settle_minutes", type=float, default=5)
    ap.add_argument("--min_free_gb", type=float, default=30)
    ap.add_argument("--max_hours", type=float, default=36)
    ap.add_argument("--watchdog_log", default="~/trumi_runs/watchdog.log")
    ap.add_argument("--once", action="store_true", help="one pass, then exit (for testing)")
    Backup(ap.parse_args()).run()
