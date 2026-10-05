# TRumi → robot post-processing

Everything that happens between "GoPro videos from a TRumi" and "a training dataset in the robot's joint space",
why each step exists, and where its parameters live. Per-session steps are scripted in
`scripts/postprocess_session.py` and driven by one config file (`configs/postprocess/<robot>_<task>.json`).
One-time steps need the robot or new recordings and are run by hand when the hardware or the cell changes.

Numbers in *italics* are from the first conveyor session (2026-10-02, UR5e + Robotiq 2F-85 with printed fingertips).

## Quick start (new session in an already set-up cell)

```bash
# 1. put the episode videos in data/<session>/raw_videos/ (any names; mapping/calibration come from the config)
# 2. run everything
uv run python scripts/postprocess_session.py --session data/<session> --config configs/postprocess/ur5e_robot1_conveyor.json
# 3. read data/<session>/postprocess_report.md; the dataset is where the config's export.out points
```
Stages can be run one at a time: `--stages pipeline glitches retarget export report`. A full log is written to
`data/<session>/postprocess.log`, the state between stages to `postprocess_state.json`.

## A. One-time setup (redo when the cell, the robot position or the gripper hardware changes)

| # | Step | Tool | Output | Redo when |
|---|---|---|---|---|
| A1 | Record 2-3 mapping takes, pick the best map: all demos localize AND marker spread ~1 cm | `scripts/tools/test_mappings.sh`, `eval_maps.sh` | a session whose `demos/mapping_*` holds the map | the scene changes (marker, tables, belt moved) |
| A2 | Touch test: touch the marker centre + 4 corners with the TRumi; the average offset is a fixed TRumi error, corrected by adding its negative | `scripts/tools/touch_test.py` | `correction_mm` in the config | new map (the correction is only valid in that map's marker frame) |
| A3 | Robot marker calibration: touch the marker centre + corners with the robot in freedrive | `scripts/robot/calibrate_marker.py` | `data/robot/marker_in_robot_base.json` | robot base or marker moved |
| A4 | Digital twin of the cell: phone scan + belt, tables, box | `scripts/sim/view_twin.py`, `render_cell.py`, `data/sim/cell.json` | collision model for the checks | cell layout changes |
| A5 | Wrist camera + fingertips on the robot look like the TRumi's: compare open/close videos | `scripts/tools/compare_gripper_views.py` | go / no-go *(after moving the camera 10 mm: open within ~1 mm, finger mask covers 99.96%)* | camera mount or fingertips change |
| A6 | Gripper sweep: Robotiq width -> command table and fingertip arc | `scripts/robot/gripper_sweep.py` + `scripts/tools/gripper_sweep_table.py` | `data/robot/gripper_sweep_*_tables.json` | fingertips or gripper unit change |
| A7 | Robot TCP = closed fingertip ends, set on the pendant *(257.2 mm, printed tips)* | pendant | `tcp_z_mm` in the config | fingertips change |
| A8 | Policy image mask (fingers + gripper body, grown 10 px); the robot must apply the same mask | `data/robot/policy_mask_2704x2028.png` | mask file | camera mount or fingertips change |
| A9 | Reference arm configuration, so every episode uses the same joint solution | `reference_q_deg` in the config | | robot or cell changes |

## B. Every session (scripted)

### B1. `pipeline`: TRumi steps 00-07
- 00 organise videos, 01 GoPro IMU, 02 ORB-SLAM3 map, 03 localise every episode in the map, 04 ArUco detection,
  05 marker frame + gripper range, 06 dataset plan, 07 MCAP (optional: `"mcap": true`).
- Step 06 drops episodes with > 10 lost tracking frames *(7 of 48: tracking broke near the plain white belt)*.
- **Map reuse** (`map_from_session`): the verified map, the mapping video's re-localisation and the gripper-calibration
  video of an earlier session are reused. Without this, step 03 re-localises the mapping video (not deterministic) and
  the marker frame shifts by a few mm and ~1 deg *(seen: 5 mm, 1.05 deg)*, which would invalidate the touch-test
  correction. The script checks that the marker frame is identical to the reference.

### B2. `glitches`: drop episodes whose SLAM path is wrong
- SLAM sometimes reports an episode as tracked while the path is wrong (relocalised in the wrong place). The pipeline's
  lost-frame filter does not catch it.
- Excluded if the fingertip moves faster than 3 m/s on >= 10 frames, or strays > 0.9 m from the marker
  (`glitches` in the config). Writes `check_result.txt` ("false ...") in the demo folder and regenerates the plan.
  *(2 excluded: ep_27 drifted 6.7 m, ep_31 jumped 90 cm.)*
- Short spikes (a few frames) are not excluded; smoothing (B3b) handles them.

### B3. `retarget`: per episode, TRumi fingertip -> robot joints (`scripts/sim/preflight.py`)
a. **Jump clean-up**: single steps > 2.5 cm are interpolated; the fingertip is kept >= 5 mm above the table.
b. **Smoothing** (`smooth_s`, 0.15 s Savitzky-Golay, 2nd order) on position and orientation. Removes frame-to-frame
   tracking jitter, which otherwise shows up as one-frame joint-speed spikes. *Path change: ~0.5 mm typical, < 1 mm at
   grasp and release, 8-15 mm only at the spikes; peak joint speeds 143-384 -> 87-226 deg/s.*
c. **Touch-test correction** (`correction_mm`, marker frame) *(-12.3, +0.3, -8.5 mm)*.
d. **Gripper axes**: TRumi fingertip frame -> Robotiq TCP frame (fixed rotation `TRUMI_TO_ROBOTIQ`).
e. **Marker -> robot base** with the robot calibration (A3).
f. **Gripper width -> Robotiq command** from the measured table (A6). Widths above the robot's range clip to fully
   open *(robot opens ~5 mm less than the TRumi)*.
g. **Arc compensation**: the Robotiq's fingertips sit up to *20 mm* closer to the flange when open (curved linkage;
   the TRumi's fingers run on rails). IK targets the real fingertip position at each opening (TCP minus the measured
   set-back), so the fingertips land where the TRumi's were. Checked with UR's own kinematics: < 1 mm.
h. **IK with one consistent arm configuration**: several start configurations are tried and the one closest to
   `reference_q_deg` is kept (among those without extra contacts). Without this some episodes used elbow-down, a base
   turned +360 deg or a flipped wrist, i.e. different joint values for the same hand pose.
i. **Checks** (in each `<out_subdir>/ep<N>_right_report.txt`): reach (IK error), joint limits, contacts in the twin
   (approximate: printed fingers are modelled as blocks, the twin's belt is ~1 cm high, rail height is estimated),
   and time above the UR5e's 180 deg/s *(after smoothing: 2 of 39 episodes above it for > 0.1 s)*.

### B4. `export`: LeRobot v3.0 (`scripts/tools/export_lerobot.py`, LeRobot 0.6 environment)
- 30 fps (every 2nd 60 Hz step), H.264 video.
- `observation.images.wrist`: the TRumi frame with the policy mask (A8), resized to 640 x 480 (pi0.5 resizes to 224).
- `observation.state`: UR joints (rad, 6) + gripper = Robotiq command / 255 (0 open, 1 closed).
- `action`: the same at the next frame (absolute targets).
- `task`: one language prompt per episode.
- Quantile statistics (q01 ... q99, needed by pi0.5) are computed by LeRobot when the dataset is written.
- Provenance: `trumi_sources.csv` (dataset episode -> source video) and `trumi_export_info.json` next to `meta/`.

### B4b. Review and verification (run after every export)
- `uv run python scripts/tools/review_retarget.py --session data/<session> --retarget data/<session>/<retarget folder>`:
  per episode, raw vs final fingertip path, change at grasp/release, joint speeds vs 180 deg/s, gripper and arc;
  `review/overview.png` for all episodes. Large changes should only appear at tracking spikes (brief raw jumps).
  Joint-speed bursts with a smooth fingertip path mean the arm passes near a singularity (e.g. the wrist close to the
  base axis); such episodes are worth excluding *(ep_4: 0.45 s up to 497 deg/s, wrist 15 cm from the base axis)*.
- Review videos (twin + GoPro): `postprocess_session.py ... --stages retarget --videos` writes
  `<retarget folder>/ep<N>_right_preflight.mp4`. (Runs with `--no_video` write no video at all.)
- `~/YAM/yam-lerobot/.venv/bin/python scripts/tools/verify_lerobot.py --root data/lerobot/<name>`: decodes every frame
  as the trainer will; flags black/frozen frames, bad values, timing, length mismatches; compares random frames with
  the original GoPro video (PSNR > 30 dB = exactly the right frame; one frame off gives ~22 dB); writes
  `thumbnails.png`.
- Twin replay straight from the dataset (what the policy trains on): `~/YAM/yam-lerobot/.venv/bin/python
  scripts/tools/lerobot_dump_states.py --root data/lerobot/<name>`, then `MUJOCO_GL=egl data/sim/.venv/bin/python
  scripts/sim/replay_dataset.py --dump data/lerobot/<name>_review`: the twin is driven by each frame's stored state,
  next to the stored wrist image; reports contacts and time above 180 deg/s from the stored joints.
- Visual check in Rerun: `~/YAM/yam-lerobot/.venv/bin/lerobot-dataset-viz --repo-id <repo_id> --root
  data/lerobot/<name> --episode-index N` (`trumi_sources.csv` maps dataset episodes to source videos).

### B5. `report`
`data/<session>/postprocess_report.md`: episode counts, exclusions with reasons, retargeting checks, output path.

## C. Deliberately NOT done to the training data

| What | Why not | Where it happens instead |
|---|---|---|
| Slowing demos down to 180 deg/s | On a moving belt, a slowed demo would show the hand slower than the belt: wrong timing to learn. After smoothing the remaining time above the limit is tiny. | Robot side: a joint-speed limiter on every command at deployment (also needed because a policy can output fast moves) |
| Relative actions | Absolute joint targets first; pi0.5 can convert (`use_relative_actions`) | training config |
| Latency compensation | Depends on the camera -> policy -> robot delay | deployment |

## D. Known limitations / open items
- The touch-test correction is tied to one map; a new map needs a new touch test (A1 + A2).
- Twin contacts are approximate (belt ~1 cm high, rail height estimated, printed fingers as blocks; STL pending).
- Episodes that lose tracking near the plain white belt are lost (*7 of 48*); adding texture near the pick zone
  (stickers on the rail/frame) should help.
- The robot must run with the same gripper assembly (printed fingertips, tags, camera mount) as measured in A5/A6.
