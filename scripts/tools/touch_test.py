"""Measure TRumi's real accuracy from a "touch test" recording.

Record one extra video in the session (e.g. named touchtest.MP4): with TRumi's fingertips closed, touch the mapping
marker's centre and its 4 corners, holding each touch still for ~1-2 s. Process the session with the normal pipeline
(the video becomes an episode). This script finds the moments where the fingertip is held still on the marker and
compares them with where those points really are (the marker frame is the pipeline's coordinate frame, so the true
positions are exactly known: centre (0, 0, 0), corners (+-8, +-8, 0) cm).

Usage (from ~/trumi):
    uv run python scripts/tools/touch_test.py --session data/<session> --video touchtest
    uv run python scripts/tools/touch_test.py --self_test
    # afterwards, keep the touch test out of the training data (then re-run steps 06-07):
    uv run python scripts/tools/touch_test.py --session data/<session> --video touchtest --exclude
"""

import argparse
import pathlib
import pickle

import numpy as np

H = 0.08  # half of the 16 cm black square
TARGETS = {"centre": (0, 0, 0), "TL": (-H, H, 0), "TR": (H, H, 0), "BR": (H, -H, 0), "BL": (-H, -H, 0)}
STILL_SPEED = 0.02  # m/s: fingertip counts as "held still" below this ...
STILL_TIME = 0.5  # ... for at least this long (s)
NEAR_MARKER = 0.06  # a still moment must be within 6 cm of a target (and 3 cm of the marker surface) to count


def find_touches(t, pos):
    """Still segments (speed < STILL_SPEED for >= STILL_TIME) near the marker surface; returns their median positions."""
    dt = np.median(np.diff(t))
    k = max(1, int(round(0.25 / dt)))  # speed over 0.25 s, so small position jitter doesn't break up a touch
    speed = np.linalg.norm(pos[k:] - pos[:-k], axis=1) / (t[k:] - t[:-k])
    still = np.concatenate([speed < STILL_SPEED, np.zeros(k, bool)])
    gap = int(round(0.2 / dt))  # bridge brief interruptions (< 0.2 s) within one touch
    i = 0
    while i < len(still):
        if not still[i]:
            j = i
            while j < len(still) and not still[j]:
                j += 1
            if 0 < i and j < len(still) and j - i <= gap:
                still[i:j] = True
            i = j
        else:
            i += 1
    touches, start = [], None
    for i, s in enumerate(np.append(still, False)):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if t[i - 1] - t[start] >= STILL_TIME:
                p = np.median(pos[start:i], 0)
                if abs(p[2]) < 0.03:
                    touches.append((t[start], t[i - 1], p))
            start = None
    return touches


def score(touches):
    rows = []
    for t0, t1, p in touches:
        name, q = min(TARGETS.items(), key=lambda kv: np.linalg.norm(p - np.array(kv[1])))
        err = p - np.array(q)
        if np.linalg.norm(err) < NEAR_MARKER:
            rows.append((name, t0, t1, err))
    return rows


def report(rows):
    if not rows:
        print("no touches found near the marker: check the episode, or that the fingertip was held still on each point")
        return
    print(f"{'point':7s} {'time (s)':>13s} {'error x':>8s} {'y':>7s} {'z':>7s} {'total':>7s}  (mm)")
    for name, t0, t1, e in rows:
        print(f"{name:7s} {t0:6.1f}-{t1:5.1f} {e[0]*1000:8.1f} {e[1]*1000:7.1f} {e[2]*1000:7.1f} {np.linalg.norm(e)*1000:7.1f}")
    E = np.array([e for *_, e in rows])
    print(f"\n{len(rows)} touches on {len({r[0] for r in rows})} of 5 points | "
          f"mean error {np.linalg.norm(E, axis=1).mean()*1000:.1f} mm, worst {np.linalg.norm(E, axis=1).max()*1000:.1f} mm | "
          f"average offset x {E[:,0].mean()*1000:+.1f}, y {E[:,1].mean()*1000:+.1f}, z {E[:,2].mean()*1000:+.1f} mm"
          f" (a consistent offset = systematic error, e.g. the fingertip offset; scatter = random error)")


def load_episode(session, video):
    plan = pickle.load(open(session / "dataset_plan.pkl", "rb"))
    for ep in plan:
        if any(video in c["video_path"] for c in ep["cameras"]):
            t = np.asarray(ep["episode_timestamps"], float)
            return t - t[0], np.asarray(ep["grippers"][0]["tcp_pose"], float)[:, :3]
    raise SystemExit(f"no episode whose video name contains '{video}' in {session/'dataset_plan.pkl'}")


def self_test():
    rng = np.random.default_rng(0)
    t = np.arange(0, 20, 1 / 60)
    order = ["centre", "TL", "TR", "BR", "BL"]
    pos = np.zeros((len(t), 3)) + [0, 0, 0.10]
    for i, n in enumerate(order):  # 1.5 s touch every 4 s, moving between them in the air
        sel = (t >= 1 + 4 * i) & (t < 2.5 + 4 * i)
        pos[sel] = np.array(TARGETS[n]) + [0.003, -0.002, 0.004]  # known systematic error (3, -2, 4) mm
    pos += rng.normal(0, 0.0005, pos.shape)
    rows = score(find_touches(t, pos))
    report(rows)
    E = np.array([e for *_, e in rows])
    ok = len(rows) == 5 and np.allclose(E.mean(0), [0.003, -0.002, 0.004], atol=0.001)
    print("SELF-TEST", "PASSED" if ok else "FAILED")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Measure TRumi accuracy from a touch-test recording.")
    ap.add_argument("--session", type=pathlib.Path)
    ap.add_argument("--video", default="touchtest", help="part of the touch-test video's file name")
    ap.add_argument("--exclude", action="store_true", help="mark the touch-test video so step 06 leaves it out of the dataset")
    ap.add_argument("--self_test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.exclude:
        dirs = [d for d in (a.session / "demos").glob("demo_*") if a.video in d.name]
        for d in dirs:
            (d / "check_result.txt").write_text("false (touch test: excluded from the dataset)\n")
            print(f"excluded {d.name} (wrote check_result.txt); re-run steps 06-07 to rebuild the dataset without it")
        if not dirs:
            print(f"no demo folder containing '{a.video}' in {a.session/'demos'}")
    else:
        t, pos = load_episode(a.session, a.video)
        report(score(find_touches(t, pos)))
