"""Render a video of one episode: both camera views + the 3D gripper-tip paths drawing in over time.

Usage (from ~/trumi):
    uv run python scripts/tools/make_3d_video.py --session data/2026-10-01_stationary_bimanual --episode 1
Output: <session>/ep<N>_3d_video.mp4
"""

import pathlib
import pickle
import subprocess

import av
import click
import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT_FPS = 15  # video frame rate; SLAM steps are 60 Hz, so we show every 4th step (real-time playback)
STEP_EVERY = 60 // OUT_FPS
SLAM_STRIDE = 2  # raw video frame = video_start + SLAM_STRIDE * step
GRIPPERS = {0: ("right gripper", "tab:red"), 1: ("left gripper", "tab:blue")}


def read_frames(video_path, frame_idxs):
    """Decode only the requested frame indices (downscaled) from a video."""
    wanted, out = set(frame_idxs), {}
    with av.open(str(video_path)) as c:
        stream = c.streams.video[0]
        stream.thread_type = "AUTO"
        for i, f in enumerate(c.decode(stream)):
            if i in wanted:
                out[i] = cv2.resize(f.to_ndarray(format="rgb24"), (480, 360))
            if i >= max(wanted):
                break
    return out


@click.command()
@click.option("--session", "session", required=True, type=click.Path(exists=True, file_okay=False, path_type=pathlib.Path), help="Processed session directory (contains dataset_plan.pkl).")
@click.option("--episode", default=1, show_default=True, help="Episode number (1-based).")
def main(session, episode):
    SESSION = session.resolve()
    ep = pickle.load(open(SESSION / "dataset_plan.pkl", "rb"))[episode - 1]
    t = ep["episode_timestamps"] - ep["episode_timestamps"][0]
    P = {g: ep["grippers"][g]["tcp_pose"][:, :3] for g in GRIPPERS}
    W = {g: ep["grippers"][g]["gripper_width"] for g in GRIPPERS}
    steps = list(range(0, len(t), STEP_EVERY))

    # camera g in the plan corresponds to gripper g (camera_idx 0 = right, 1 = left)
    frames = {}
    for g in GRIPPERS:
        cam = ep["cameras"][g]
        s0 = cam["video_start_end"][0]
        idx = [s0 + SLAM_STRIDE * s for s in steps]
        got = read_frames(SESSION / "demos" / cam["video_path"], idx)
        frames[g] = [got[i] for i in idx]

    allp = np.vstack(list(P.values()) + [np.zeros((1, 3))])
    lo, hi = allp.min(0) - 0.03, allp.max(0) + 0.03
    c, r = (lo + hi) / 2, (hi - lo).max() / 2
    xx, yy = np.meshgrid(np.linspace(lo[0], hi[0], 2), np.linspace(lo[1], hi[1], 2))

    tmp = SESSION / f"ep{episode}_3d_video_tmp.mp4"
    out = SESSION / f"ep{episode}_3d_video.mp4"
    fig = plt.figure(figsize=(16, 9), dpi=80)
    writer = None
    for n, s in enumerate(steps):
        fig.clf()
        fig.suptitle(f"Episode {episode}   t = {t[s]:5.2f} s", fontsize=16)
        for k, g in enumerate((1, 0)):  # left camera on the left, right camera on the right
            ax = fig.add_axes([0.01 + 0.25 * k, 0.52, 0.24, 0.40])
            ax.imshow(frames[g][n])
            ax.set_title(f"{GRIPPERS[g][0]} camera", color=GRIPPERS[g][1])
            ax.axis("off")
        axw = fig.add_axes([0.05, 0.08, 0.43, 0.36])
        for g, (lab, col) in GRIPPERS.items():
            axw.plot(t, W[g] * 100, color=col, alpha=0.25)
            axw.plot(t[: s + 1], W[g][: s + 1] * 100, color=col, lw=2, label=lab)
        axw.axvline(t[s], color="k", lw=1)
        axw.set_xlim(0, t[-1])
        axw.set_xlabel("time (s)")
        axw.set_ylabel("gripper opening (cm)")
        axw.grid(alpha=0.3)
        axw.legend(loc="lower right")
        ax3 = fig.add_axes([0.50, 0.04, 0.44, 0.88], projection="3d")
        ax3.plot_surface(xx, yy, np.zeros_like(xx), color="peru", alpha=0.2)
        ax3.scatter(0, 0, 0, color="k", s=80, marker="s")
        for g, (lab, col) in GRIPPERS.items():
            p = P[g]
            ax3.plot(*p.T, color=col, alpha=0.15)  # full path, faint
            ax3.plot(*p[: s + 1].T, color=col, lw=2.5, label=lab)  # path so far
            ax3.scatter(*p[s], color=col, s=120, edgecolor="k", depthshade=False)  # current position
        ax3.set_xlim(c[0] - r, c[0] + r)
        ax3.set_ylim(c[1] - r, c[1] + r)
        ax3.set_zlim(c[2] - r, c[2] + r)
        ax3.set_xlabel("x (m)")
        ax3.set_ylabel("y (m)")
        ax3.set_zlabel("height above table (m)")
        ax3.view_init(elev=25, azim=-60 + 60 * n / max(1, len(steps) - 1))  # slow rotation for depth
        ax3.legend(loc="upper left")
        ax3.set_title("gripper tip paths (brown = table, black square = marker)")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        if writer is None:
            writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (img.shape[1], img.shape[0]))
        writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    writer.release()
    # re-encode to H.264 so it plays everywhere (browser, VS Code, phone)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    tmp.unlink()
    print(f"saved {out}  ({len(steps)} frames, {len(steps) / OUT_FPS:.1f} s)")


if __name__ == "__main__":
    main()
