"""Post-process a TRumi session into a robot-ready training dataset, driven by one config file.
Every step and why it exists is described in docs/POST_PROCESSING.md.

Stages (in order; --stages picks a subset):
  pipeline  TRumi steps 00-07. With "map_from_session" the verified map, mapping re-localisation and therefore the
            marker frame of an earlier session are reused, so a touch-test correction measured there stays valid.
  glitches  exclude episodes whose SLAM path jumps or strays (check_result.txt "false ..."), regenerate the plan
  retarget  per episode, scripts/sim/preflight.py without video: jump clean-up, smoothing, touch-test correction,
            marker -> robot calibration, IK at the real fingertips (TCP + measured arc), one consistent arm
            configuration, measured gripper width -> command table; reports reach, joint limits, collisions in the
            twin and time above the UR5e's 180 deg/s
  export    LeRobot v3.0 dataset (scripts/tools/export_lerobot.py, run in a LeRobot 0.6 environment)
  report    <session>/postprocess_report.md

Usage (from ~/trumi):
    uv run python scripts/postprocess_session.py --session data/<session> --config configs/postprocess/<name>.json
For a new session put the episode videos in <session>/raw_videos/ (plus mapping.mp4 and gripper_calibration/ unless
the config reuses a map).
"""

import argparse
import concurrent.futures as cf
import datetime
import json
import pathlib
import pickle
import re
import shutil
import subprocess
import sys

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "tools"))
from episode_report import path_checks  # noqa: E402

STEPS = REPO / "scripts" / "scripts_slam_pipeline"
CALIB = REPO / "example" / "calibration"
SIM_PY = REPO / "data" / "sim" / ".venv" / "bin" / "python"
STAGES = ["pipeline", "glitches", "retarget", "export", "report"]


def run(cmd, log, **kw):
    """Run a command, appending its output to the session log; stop on failure."""
    print("  $", " ".join(map(str, cmd)))
    with open(log, "a") as fh:
        fh.write(f"\n===== {datetime.datetime.now():%Y-%m-%d %H:%M:%S} $ {' '.join(map(str, cmd))}\n")
        fh.flush()
        r = subprocess.run(list(map(str, cmd)), stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, **kw)
    if r.returncode:
        raise SystemExit(f"failed (exit {r.returncode}): {' '.join(map(str, cmd))}; see {log}")


def py(script, *args):
    return [sys.executable, STEPS / script, *args]


def stage_pipeline(s, cfg, st, log):
    if (s / "dataset_plan.pkl").exists():
        print("  dataset_plan.pkl exists: pipeline already run (delete it to redo)")
        return
    ref = cfg.get("map_from_session")
    if not ref:
        run([sys.executable, REPO / "scripts" / "dataset_generation_pipeline.py", s], log)
        return
    ref = REPO / ref
    ref_map = next((ref / "demos").glob("mapping_*"))
    ref_cal = next((ref / "demos").glob("gripper_calibration_*"))
    if not (s / "demos").exists():
        raw = s / "raw_videos"
        if not raw.is_dir():
            raise SystemExit(f"put the episode videos in {raw}/ first")
        (raw / "gripper_calibration").mkdir(exist_ok=True)
        shutil.copy2(ref_map / "raw_video.mp4", raw / "mapping.mp4")
        shutil.copy2(ref_cal / "raw_video.mp4", raw / "gripper_calibration" / "calibration.MP4")
        print(f"  reusing the mapping and gripper-calibration videos of {ref.name}")
        run(py("00_process_videos.py", s), log)
    run(py("01_extract_gopro_imu.py", s), log)
    new_map = next((s / "demos").glob("mapping_*"))
    if new_map.name != ref_map.name:
        raise SystemExit(f"mapping folder {new_map.name} != {ref_map.name}: not the same mapping video")
    # the map, and the mapping video's re-localisation (which defines the marker frame in step 05)
    for f in ("map_atlas.osa", "mapping_camera_trajectory.csv", "camera_trajectory.csv", "slam_mask.png"):
        shutil.copy2(ref_map / f, new_map / f)
    m = new_map / "map_atlas.osa"
    run(py("03_batch_slam.py", "--input_dir", s / "demos", "--map_path", m), log)
    run(py("04_detect_aruco.py", "--input_dir", s / "demos", "--camera_intrinsics", CALIB / "gopro13_intrinsics_2_7k.json",
           "--aruco_yaml", CALIB / "aruco_config.yaml", "--slam_frame_stride", "2"), log)
    run(py("05_run_calibrations.py", "--input_dir", s), log)
    a, b = (np.array(json.load(open(d / "tx_slam_tag.json"))["tx_slam_tag"]) for d in (ref_map, new_map))
    st["marker_frame_matches_reference"] = bool(np.allclose(a, b, atol=1e-6))
    print(f"  marker frame identical to {ref.name}: {st['marker_frame_matches_reference']}")
    regenerate_plan(s, cfg, log)


def regenerate_plan(s, cfg, log):
    run(py("06_generate_dataset_plan.py", "--input_dir", s, "--slam_frame_stride", "2"), log)
    if cfg.get("mcap", False):
        if (s / "dataset_mcap").exists():
            shutil.rmtree(s / "dataset_mcap")
        run(py("07_generate_mcap_dataset.py", "--output", s / "dataset_mcap", "--slam_frame_stride", "2", s), log)


def stage_glitches(s, cfg, st, log):
    g = cfg["glitches"]
    excluded = st.setdefault("glitch_excluded", [])
    for _ in range(2):
        plan = pickle.load(open(s / "dataset_plan.pkl", "rb"))
        bad = []
        for ep in plan:
            t = np.asarray(ep["episode_timestamps"], float)
            n, far = path_checks(t, np.asarray(ep["grippers"][0]["tcp_pose"], float)[:, :3], g["max_jump_speed_m_s"])
            if n >= g["max_jumps"] or far > g["max_dist_from_marker_m"]:
                d = s / "demos" / pathlib.Path(ep["cameras"][0]["video_path"]).parent
                reason = f"{n} jumps > {g['max_jump_speed_m_s']} m/s, farthest {far:.2f} m from the marker"
                (d / "check_result.txt").write_text(f"false (post-process: SLAM glitch, {reason})\n")
                bad.append({"video": "_".join(d.name.split("_")[4:]), "reason": reason})
        if not bad:
            break
        excluded += bad
        print(f"  excluded {len(bad)}: " + "; ".join(f"{b['video']} ({b['reason']})" for b in bad))
        regenerate_plan(s, cfg, log)
    print(f"  episodes in the plan: {len(pickle.load(open(s / 'dataset_plan.pkl', 'rb')))}")


def stage_retarget(s, cfg, st, log):
    r = cfg["retarget"]
    n = len(pickle.load(open(s / "dataset_plan.pkl", "rb")))
    out = s / r["out_subdir"]
    if out.exists():
        shutil.rmtree(out)
    (out / "logs").mkdir(parents=True)
    common = ["--session", s, "--no_video", "--out_subdir", r["out_subdir"], "--calibration", REPO / r["calibration"],
              "--cell", REPO / r["cell"], "--tcp_z_mm", r["tcp_z_mm"], "--smooth_s", r["smooth_s"],
              "--correction_mm", *r["correction_mm"], "--gripper_tables", REPO / r["gripper_tables"],
              "--reference_q_deg", *r["reference_q_deg"]]

    def one(k):
        with open(out / "logs" / f"ep{k}.log", "w") as fh:
            return k, subprocess.run(list(map(str, [SIM_PY, REPO / "scripts" / "sim" / "preflight.py", "--episode", k, *common])),
                                     stdout=fh, stderr=subprocess.STDOUT, cwd=REPO, env={"MUJOCO_GL": "egl", "PATH": "/usr/bin:/bin"}).returncode

    print(f"  re-solving {n} episodes for the robot ({r['workers']} in parallel)")
    with cf.ThreadPoolExecutor(r["workers"]) as ex:
        failed = [k for k, rc in ex.map(one, range(1, n + 1)) if rc]
    if failed:
        raise SystemExit(f"preflight failed for episodes {failed}; see {out / 'logs'}")
    rows = []
    for k in range(1, n + 1):
        rep = (out / f"ep{k}_right_report.txt").read_text()
        tr = json.load(open(out / f"ep{k}_right_robot_trajectory.json"))
        c = re.search(r"collisions: (.*)", rep).group(1)
        rows.append({"episode": k, "pass": "PASS" in rep.splitlines()[0], "ik_mm": float(re.search(r"position error max ([\d.]+) mm", rep).group(1)),
                     "near_limit": "joint limits: OK" not in rep, "contacts": sorted(set(re.findall(r"(\w+) at", c))),
                     "over_180_s": tr["time_over_180_deg_s"]})
    st["retarget"] = rows
    print(f"  pass {sum(r_['pass'] for r_ in rows)}/{n} | IK max {max(r_['ik_mm'] for r_ in rows):.1f} mm | near a joint limit: "
          f"{sum(r_['near_limit'] for r_ in rows)} | episodes over 180 deg/s for > {r['max_over_180_s']} s: "
          f"{sum(r_['over_180_s'] > r['max_over_180_s'] for r_ in rows)}")


def stage_export(s, cfg, st, log):
    e, r = cfg["export"], cfg["retarget"]
    run([pathlib.Path(e["python"]).expanduser(), REPO / "scripts" / "tools" / "export_lerobot.py", "--session", s,
         "--trajectories", s / r["out_subdir"], "--out", REPO / e["out"], "--repo_id", e["repo_id"], "--task", e["task"],
         "--robot_type", e["robot_type"], "--mask", REPO / e["mask"], "--fps", e["fps"], "--width", e["width"],
         "--height", e["height"], "--overwrite"], log)
    st["export"] = str(REPO / e["out"])
    print(f"  saved {REPO / e['out']}")


def stage_report(s, cfg, st, log):
    plan = pickle.load(open(s / "dataset_plan.pkl", "rb"))
    demos = sorted(d for d in (s / "demos").glob("demo_*"))
    lost = [d for d in demos if not (d / "camera_trajectory.csv").exists()]
    lines = [f"# Post-processing report: {s.name}", f"{datetime.datetime.now():%Y-%m-%d %H:%M}, config `{cfg['_path']}`", "",
             f"- demo videos: {len(demos)}; SLAM produced no trajectory: {len(lost)}",
             f"- episodes in the final plan: {len(plan)}"]
    if "marker_frame_matches_reference" in st:
        lines.append(f"- marker frame identical to {cfg['map_from_session']}: {st['marker_frame_matches_reference']}")
    for b in st.get("glitch_excluded", []):
        lines.append(f"- excluded (SLAM glitch): {b['video']}: {b['reason']}")
    if st.get("retarget"):
        rows, r = st["retarget"], cfg["retarget"]
        lines += ["", "## Retargeting (twin pre-flight, no video)",
                  f"- pass {sum(x['pass'] for x in rows)}/{len(rows)}, IK max {max(x['ik_mm'] for x in rows):.1f} mm, "
                  f"near a joint limit: {sum(x['near_limit'] for x in rows)}",
                  f"- over the UR5e's 180 deg/s for more than {r['max_over_180_s']} s: "
                  + (", ".join(f"ep{x['episode']} ({x['over_180_s']:.2f} s)" for x in rows if x["over_180_s"] > r["max_over_180_s"]) or "none"),
                  "- twin contacts (gripper near box/belt/rail, within the twin's ~1 cm uncertainty): "
                  + (", ".join(f"ep{x['episode']}: {'/'.join(x['contacts'])}" for x in rows if x["contacts"]) or "none")]
    if st.get("export"):
        lines += ["", f"## Export", f"- LeRobot v3.0 dataset: `{st['export']}`"]
    (s / "postprocess_report.md").write_text("\n".join(lines) + "\n")
    print(f"  saved {s / 'postprocess_report.md'}")


def main(a):
    cfg = json.load(open(a.config))
    cfg["_path"] = str(a.config)
    s = a.session.resolve()
    log = s / "postprocess.log"
    state_file = s / "postprocess_state.json"
    st = json.load(open(state_file)) if state_file.exists() else {}
    for stage in a.stages:
        print(f"== {stage}")
        globals()[f"stage_{stage}"](s, cfg, st, log)
        json.dump(st, open(state_file, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Post-process a TRumi session into a robot-ready LeRobot dataset.")
    ap.add_argument("--session", type=pathlib.Path, required=True)
    ap.add_argument("--config", type=pathlib.Path, required=True)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    main(ap.parse_args())
