"""Watchdog for the two pi0.5 training runs on the GPU server (runs ON the server, in its own tmux session, so it keeps
working if the desktop or the network goes away). Python standard library only.

Every --interval seconds, for each run: running / finished (TRAIN_EXIT=0 in its log) / failed (TRAIN_EXIT != 0) /
died (no process, no exit line) / hung (process alive, log silent for --hang_minutes).
  - A RESOURCE failure (out of GPU or system memory, disk full, killed, CUDA fault, hang) of either run:
      stop run v2 to free the machine, then restart run v1 alone: resume from its latest checkpoint if there is one,
      otherwise start it again in a fresh output folder. At most --max_restarts restarts.
  - Any other failure (e.g. a bug) is logged and NOT retried, so a broken setup cannot loop.
  - Disk: warns when free space drops below --min_free_gb (the desktop backup job prunes old checkpoints).
Writes ~/trumi_runs/watchdog.log (events) and ~/trumi_runs/status.txt (current state); exits when v1 has finished
and v2 is no longer running.

Usage (on the server):
    tmux new-session -d -s trumi-watchdog "python3 ~/trumi_runs/train_watchdog.py"
"""

import argparse
import datetime
import os
import pathlib
import re
import shutil
import subprocess
import time

HOME = pathlib.Path.home()
REPO = HOME / "ur5e-lerobot"
RUNS = {  # v1 is the run to keep alive; v2 is stopped first when resources are short
    "v1": {"session": "trumi-v1", "script": HOME / "trumi_runs/train_v1_finger_mask.sh", "gpu": 0,
           "log": HOME / "logs_pi05_trumi_v1_finger_mask.log", "out": "outputs/pi05_trumi_conveyor_v1_finger_mask_b16_40k"},
    "v2": {"session": "trumi-v2", "script": HOME / "trumi_runs/train_v2_fingers_visible.sh", "gpu": 1,
           "log": HOME / "logs_pi05_trumi_v2_fingers_visible.log", "out": "outputs/pi05_trumi_conveyor_v2_fingers_visible_b16_40k"},
}
RESOURCE = re.compile(r"OutOfMemoryError|CUDA out of memory|No space left on device|Cannot allocate memory|"
                      r"DataLoader worker \(pid.*\) is killed|Killed|Bus error|CUDA error|cudaError|NCCL error|"
                      r"illegal memory access|unspecified launch failure|ECC error|MemoryError")
EVENTS = HOME / "trumi_runs" / "watchdog.log"
STATUS = HOME / "trumi_runs" / "status.txt"


def log(msg):
    line = f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(EVENTS, "a") as fh:
        fh.write(line + "\n")


def tail(path, n=400):
    try:
        text = path.read_text(errors="replace").replace("\r", "\n")
    except FileNotFoundError:
        return ""
    return "\n".join(text.splitlines()[-n:])


def alive(run):
    pattern = pathlib.Path(RUNS[run]["out"]).name.rsplit("_b16", 1)[0]  # matches the first start and resumes
    r = subprocess.run(["pgrep", "-f", f"lerobot-train.*{pattern}"], capture_output=True, text=True)
    return bool(r.stdout.strip())


def state(run, a):
    cfg = RUNS[run]
    text = tail(cfg["log"])
    exits = re.findall(r"TRAIN_EXIT=(-?\d+)", text)
    if exits:
        code = int(exits[-1])
        return ("finished" if code == 0 else "failed"), code, text
    if alive(run):
        try:
            silent = (time.time() - cfg["log"].stat().st_mtime) / 60
        except FileNotFoundError:
            silent = 0
        return ("hung" if silent > a.hang_minutes else "running"), None, text
    return "died", None, text


def progress(text):
    m = re.findall(r"(\d+)/40000 \[[^<]*<([^,]+), *([\d.]+)s/step", text)
    return f"step {m[-1][0]}/40000, {m[-1][2]} s/step, ~{m[-1][1]} left" if m else "starting"


def stop(run):
    cfg = RUNS[run]
    pattern = pathlib.Path(cfg["out"]).name.rsplit("_b16", 1)[0]
    subprocess.run(["pkill", "-f", f"lerobot-train.*{pattern}"])
    time.sleep(20)
    subprocess.run(["pkill", "-9", "-f", f"lerobot-train.*{pattern}"])
    subprocess.run(["tmux", "kill-session", "-t", cfg["session"]], capture_output=True)


def restart_v1(attempt):
    cfg = RUNS["v1"]
    out = REPO / cfg["out"]
    last = out / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    newlog = HOME / f"logs_pi05_trumi_v1_finger_mask_restart{attempt}.log"
    if last.exists():
        cmd = (f"cd {REPO} && export CUDA_VISIBLE_DEVICES={cfg['gpu']} OMP_NUM_THREADS=8 && "
               f".venv/bin/lerobot-train --config_path={last} --resume=true")
        how = f"resuming from {os.path.realpath(out / 'checkpoints' / 'last')}"
    else:  # no checkpoint yet: start over in a fresh output folder (lerobot refuses an existing one)
        script = HOME / "trumi_runs" / f"train_v1_finger_mask_restart{attempt}.sh"
        script.write_text(cfg["script"].read_text().replace(cfg["out"], f"{cfg['out']}_restart{attempt}"))
        script.chmod(0o755)
        cmd = str(script)
        cfg["out"] = f"{cfg['out']}_restart{attempt}"
        how = f"from scratch into {cfg['out']} (no checkpoint yet)"
    subprocess.run(["tmux", "kill-session", "-t", cfg["session"]], capture_output=True)
    shell = f"{cmd} 2>&1 | tee {newlog}; echo TRAIN_EXIT=${{PIPESTATUS[0]}} | tee -a {newlog}; exec bash"
    subprocess.run(["tmux", "new-session", "-d", "-s", cfg["session"], "bash", "-c", shell], check=True)
    cfg["log"] = newlog
    log(f"ACTION restarted v1 (attempt {attempt}) {how}; log {newlog}")


def main(a):
    if a.once:  # status only, no actions
        for run in RUNS:
            st, code, text = state(run, a)
            print(f"{run}: {st}{f' (exit {code})' if code is not None else ''} | {progress(text)}")
        print(f"disk free {shutil.disk_usage(HOME).free / 1e9:.0f} GB")
        return
    log("watchdog started")
    restarts, reported = 0, {}
    while True:
        states = {run: state(run, a) for run in RUNS}
        free_gb = shutil.disk_usage(HOME).free / 1e9
        lines = [f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}  disk free {free_gb:.0f} GB  restarts {restarts}/{a.max_restarts}"]
        for run, (st, code, text) in states.items():
            lines.append(f"  {run}: {st}{f' (exit {code})' if code is not None else ''} | {progress(text)}")
        STATUS.write_text("\n".join(lines) + "\n")
        if free_gb < a.min_free_gb and reported.get("disk") != int(free_gb):
            log(f"ALERT disk free {free_gb:.0f} GB < {a.min_free_gb} GB (is the desktop backup job pruning?)")
            reported["disk"] = int(free_gb)

        bad = {run: s for run, s in states.items() if s[0] in ("failed", "died", "hung")}
        for run, (st, code, text) in bad.items():
            key = (run, st, code, str(RUNS[run]["log"]))
            if reported.get(key):
                continue
            reported[key] = True
            resource = st in ("hung", "died") or code in (137, -9) or bool(RESOURCE.search(text))
            match = RESOURCE.search(text)
            log(f"FAILED {run}: {st}{f' exit {code}' if code is not None else ''} | resource problem: {resource}"
                + (f" ({match.group(0)})" if match else ""))
            if not resource:
                log(f"NO ACTION for {run}: not a resource problem, needs a look (log {RUNS[run]['log']})")
                continue
            if states["v2"][0] in ("running", "hung"):
                log("ACTION stopping v2 to free resources for v1")
                stop("v2")
            v1_state = state("v1", a)[0]
            if v1_state in ("running",):
                log("v1 is still running fine: leaving it alone")
            elif v1_state == "finished":
                log("v1 already finished: nothing to restart")
            elif restarts >= a.max_restarts:
                log(f"NO ACTION: v1 needs a restart but the limit of {a.max_restarts} restarts is reached")
            else:
                if v1_state == "hung":
                    stop("v1")
                restarts += 1
                restart_v1(restarts)

        if states["v1"][0] == "finished" and states["v2"][0] not in ("running", "hung"):
            log("v1 finished and v2 is not running: watchdog done")
            break
        time.sleep(a.interval)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Keep the pi0.5 training runs healthy on the GPU server.")
    ap.add_argument("--interval", type=int, default=300)
    ap.add_argument("--hang_minutes", type=float, default=30)
    ap.add_argument("--max_restarts", type=int, default=3)
    ap.add_argument("--min_free_gb", type=float, default=40)
    ap.add_argument("--once", action="store_true", help="print the current state and exit (no actions)")
    main(ap.parse_args())
