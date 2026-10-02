"""Plan a recording session in the digital twin (phone scan + UR5e / Robotiq 2F-85 at the real base position).

Using inverse kinematics plus collision checks against the cell geometry in data/sim/cell.json, it checks:
  1. calibration touches: can the gripper touch the marker centre and its 4 corners, pointing straight down?
  2. belt picks: where can it do a top-down pick (fingertip 2 cm above the belt) with a 15 cm hover above it?
  3. box drops: can it release above the box, and reach into it?
  4. box placement: where on the tables could the box go so that drops work?
  5. parking: a collision-free pose that keeps the arm as far as possible from the work area.

Outputs in data/sim/plan/: plan_topdown.png, plan_poses.png, plan_summary.txt, plan_summary.json

Usage (from ~/trumi):
    MUJOCO_GL=egl data/sim/.venv/bin/python scripts/sim/plan_cell.py [--cell data/sim/cell.json]
"""

import argparse
import json
import pathlib
import subprocess
import sys

import mujoco
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from replay_ur5e import solve_ik  # noqa: E402
from view_twin import REPO, SCAN, twin_spec  # noqa: E402

PICK_ABOVE_BELT = 0.02  # fingertip height above the belt at the grasp
HOVER = 0.15  # hover height above the grasp / release point
BELT_GRID = 0.02
BOX_GRID = 0.05
POS_TOL = 0.002  # IK must hit the target within 2 mm ...
ROT_TOL_DEG = 1.0  # ... and 1 deg
LIMIT_MARGIN_DEG = 5.0
MAX_JUMP_DEG = 60.0  # hover -> grasp must stay on the same IK branch
CLEARANCE = 0.01  # report contacts closer than 1 cm as collisions
OBSTACLE_RGBA = [0.2, 0.6, 1.0, 0.35]


def axes(yaw_deg):
    a = np.radians(yaw_deg)
    return np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0])


def quat_z(yaw_deg):
    return np.roll(R.from_euler("z", yaw_deg, degrees=True).as_quat(), 1).tolist()  # mujoco wxyz


def add_obstacles(spec, cell):
    """Add the cell's tables, belt, rails, marker board and box as (semi-transparent) collision boxes."""
    wb = spec.worldbody
    names = {}

    def box(name, centre, half, yaw_deg, rgba=OBSTACLE_RGBA, collide=True):
        wb.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(centre), quat=quat_z(yaw_deg),
                    rgba=rgba, contype=1 if collide else 0, conaffinity=1 if collide else 0, margin=CLEARANCE, group=1)
        names[name] = True

    mt = cell["main_table"]
    box("main_table", [*mt["centre"], mt["top_z"] - 0.02], [mt["length"] / 2, mt["width"] / 2, 0.02], mt["yaw_deg"])
    lt = cell["lower_table"]
    ex, ey = axes(lt["yaw_deg"])
    c = np.array([*lt["corner_near_belt_and_robot"], 0.0]) - 0.5 * ex - 0.325 * ey
    box("lower_table", [c[0], c[1], lt["top_z"] - 0.02], [0.5, 0.325, 0.02], lt["yaw_deg"])
    b = cell["belt"]
    bex, bey = axes(b["yaw_deg"])
    bc = np.array([*b["centre"], b["top_z"] - 0.05])
    box("belt", bc, [b["length"] / 2, b["width"] / 2, 0.05], b["yaw_deg"])
    rh = b["rail_height_above_belt"]
    for side in (-1, 1):
        rc = bc + side * (b["width"] / 2 + 0.01) * bey
        rc[2] = b["top_z"] - 0.05 + (0.1 + rh) / 2
        box(f"rail_{'left' if side > 0 else 'right'}", rc, [b["length"] / 2, 0.01, (0.1 + rh) / 2], b["yaw_deg"])
    mk = cell["marker"]
    box("marker_board", [mk["centre"][0], mk["centre"][1], mk["centre"][2] - 0.0015], [0.095, 0.095, 0.0015],
        mk["yaw_deg"], rgba=[1, 1, 1, 0.9])
    box("marker_black", [mk["centre"][0], mk["centre"][1], mk["centre"][2] + 0.0002], [mk["size"] / 2, mk["size"] / 2, 0.0002],
        mk["yaw_deg"], rgba=[0, 0, 0, 1], collide=False)
    bx = cell.get("box")
    if not bx:  # e.g. during calibration the box is not on the table yet
        return names
    kx, ky = axes(bx["yaw_deg"])
    L, W, Hh = bx["size"]
    bb = np.array(bx["centre"], dtype=float)
    brown = [0.65, 0.45, 0.25, 0.6]
    box("box_floor", bb + [0, 0, 0.003], [L / 2, W / 2, 0.003], bx["yaw_deg"], rgba=brown)
    for nm, off, half in [("box_wall_px", L / 2 * kx, [0.003, W / 2, Hh / 2]), ("box_wall_nx", -L / 2 * kx, [0.003, W / 2, Hh / 2]),
                          ("box_wall_py", W / 2 * ky, [L / 2, 0.003, Hh / 2]), ("box_wall_ny", -W / 2 * ky, [L / 2, 0.003, Hh / 2])]:
        box(nm, bb + off + [0, 0, Hh / 2], half, bx["yaw_deg"], rgba=brown)
    return names


SIM_TCP_Z_MM = 155.8  # Robotiq 2F-85 'pinch' point in the Menagerie model, measured from the UR tool flange


class Planner:
    def __init__(self, cell, controller_yaw_deg=None, tcp_z_mm=None):
        spec, self.base, _ = twin_spec(show_scanned_robot=False, controller_yaw_deg=controller_yaw_deg)
        if tcp_z_mm is not None:  # match the real robot's TCP (fingertip) offset from the flange
            spec.site("g_pinch").pos = [0.0, 0.0, 0.145 + (tcp_z_mm - SIM_TCP_Z_MM) / 1000]
        self.obstacle_names = add_obstacles(spec, cell)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.site = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_SITE, "g_pinch")
        self.obst = {mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_GEOM, n): n for n in self.obstacle_names}
        self.lim = np.degrees(self.m.jnt_range[:6])
        self.seeds = [np.array([pan, lift, elb, w1, -np.pi / 2, 0.0])
                      for pan in np.radians(np.arange(-180, 180, 45))
                      for lift, elb, w1 in [(-1.57, 1.57, -1.57), (-1.0, 1.8, -2.4), (-2.0, 1.2, -0.8)]]

    @staticmethod
    def down(yaw_deg):
        """Gripper pointing straight down, fingers closing along the horizontal direction yaw_deg."""
        y = np.array([np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg)), 0.0])
        z = np.array([0.0, 0.0, -1.0])
        return np.column_stack([np.cross(y, z), y, z])

    def contacts(self, q, allowed=()):
        """Robot-vs-obstacle contacts closer than CLEARANCE (excluding allowed obstacle names), and self-contacts."""
        self.d.qpos[:] = 0
        self.d.qpos[:6] = q
        mujoco.mj_forward(self.m, self.d)
        hits = []
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            g1, g2 = c.geom1, c.geom2
            b1, b2 = self.m.geom_bodyid[g1], self.m.geom_bodyid[g2]
            if c.dist >= CLEARANCE:
                continue
            if b1 == 0 or b2 == 0:  # one side is a world obstacle
                name = self.obst.get(g1 if b1 == 0 else g2)
                if name and name not in allowed and c.dist < 0.0:
                    hits.append(name)
            elif c.dist < 0.0:
                hits.append("self")
        return hits

    def solve(self, p, Rm, q0=None):
        """IK with warm start, falling back to the seed set. Returns (q, ok_pose) for the first accurate solution."""
        tries = ([q0] if q0 is not None else []) + self.seeds
        best = None
        for s in tries:
            q, ep, er = solve_ik(self.m, self.d, self.site, p, Rm, s)
            if ep < POS_TOL and er < ROT_TOL_DEG:
                return q, True
            if best is None or ep < best[1]:
                best = (q, ep)
        return best[0], False

    def solutions(self, p, Rm, q0=None, allowed=()):
        """All distinct, accurate AND feasible IK solutions, best first (close to q0, small wrist angles)."""
        found = []
        for s in ([q0] if q0 is not None else []) + self.seeds:
            q, ep, er = solve_ik(self.m, self.d, self.site, p, Rm, s)
            if ep >= POS_TOL or er >= ROT_TOL_DEG or any(np.max(np.abs(q - f)) < 0.05 for f in found):
                continue
            found.append(q)
        ok = [q for q in found if self.feasible(q, True, allowed)[0]]
        ref = q0 if q0 is not None else np.zeros(6)
        return sorted(ok, key=lambda q: np.abs(q - ref).sum() + 0.5 * np.abs(q[3:]).sum())

    def feasible(self, q, ok_pose, allowed=()):
        if not ok_pose:
            return False, "unreachable"
        deg = np.degrees(q)
        if np.any(deg < self.lim[:, 0] + LIMIT_MARGIN_DEG) or np.any(deg > self.lim[:, 1] - LIMIT_MARGIN_DEG):
            return False, "joint limit"
        hits = self.contacts(q, allowed)
        return (not hits), (",".join(sorted(set(hits))) if hits else "ok")

    def pick_check(self, p, yaws, allowed_at_target=(), q_warm=None):
        """Hover HOVER above p, then descend to p. Returns (status, q_target, yaw, reason)."""
        def descend(Rm, qh):
            qt, okt = self.solve(p, Rm, qh)
            ft, rt = self.feasible(qt, okt, allowed_at_target)
            if ft and np.max(np.abs(np.degrees(qt - qh))) < MAX_JUMP_DEG:
                return qt, "ok"
            return None, (rt if not ft else "branch jump")

        hover_ok_any, last_reason = False, ""
        # fast path: continue from the previous grid point's configuration
        if q_warm is not None:
            for yaw in yaws:
                Rm = self.down(yaw)
                qh, okh = self.solve(p + [0, 0, HOVER], Rm, q_warm)
                if self.feasible(qh, okh)[0]:
                    hover_ok_any = True
                    qt, why = descend(Rm, qh)
                    if qt is not None:
                        return "green", qt, yaw, "ok"
        # full search: every feasible hover configuration, for every gripper rotation
        for yaw in yaws:
            Rm = self.down(yaw)
            hovers = self.solutions(p + [0, 0, HOVER], Rm, q_warm)
            if not hovers:
                last_reason = last_reason or "hover: no collision-free configuration"
                continue
            hover_ok_any = True
            for qh in hovers[:4]:
                qt, why = descend(Rm, qh)
                if qt is not None:
                    return "green", qt, yaw, "ok"
                last_reason = f"target: {why}"
        return ("yellow" if hover_ok_any else "red"), None, None, last_reason


def robot_points(pl, q, n_per_geom=60):
    pl.d.qpos[:] = 0
    pl.d.qpos[:6] = q
    mujoco.mj_kinematics(pl.m, pl.d)
    pts = []
    for g in range(pl.m.ngeom):
        if pl.m.geom_bodyid[g] == 0 or pl.m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = pl.m.geom_dataid[g]
        v = pl.m.mesh_vert[pl.m.mesh_vertadr[mid]: pl.m.mesh_vertadr[mid] + pl.m.mesh_vertnum[mid]]
        v = v[:: max(1, len(v) // n_per_geom)]
        pts.append(v @ pl.d.geom_xmat[g].reshape(3, 3).T + pl.d.geom_xpos[g])
    return np.vstack(pts)


# ---------------------------------------------------------------- drawing helpers (numpy only)
class TopDown:
    def __init__(self):
        meta = json.load(open(SCAN / "wide_topdown.json"))
        self.s, self.cx, self.cy, self.W, self.H = meta["m_per_px"], meta["centre_x"], meta["centre_y"], meta["width_px"], meta["height_px"]
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(SCAN / "wide_topdown.png"), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        self.img = (np.frombuffer(raw, np.uint8).reshape(self.H, self.W, 3) * 0.65).astype(np.uint8)  # dimmed background

    def px(self, p):
        return int(round((p[0] - self.cx) / self.s + self.W / 2)), int(round(self.H / 2 - (p[1] - self.cy) / self.s))

    def dot(self, p, rgb, r=7):
        u, v = self.px(p)
        y, x = np.ogrid[-r: r + 1, -r: r + 1]
        m = x * x + y * y <= r * r
        u0, v0 = u - r, v - r
        if 0 <= u0 and u0 + 2 * r < self.W and 0 <= v0 and v0 + 2 * r < self.H:
            self.img[v0: v0 + 2 * r + 1, u0: u0 + 2 * r + 1][m] = rgb

    def line(self, a, b, rgb, w=3):
        for t in np.linspace(0, 1, int(np.linalg.norm(np.subtract(a, b)) / self.s) + 2):
            self.dot(np.add(a, np.multiply(t, np.subtract(b, a))), rgb, w)

    def poly(self, pts, rgb, w=3):
        for i in range(len(pts)):
            self.line(pts[i], pts[(i + 1) % len(pts)], rgb, w)

    def save(self, path, legend):
        tmp = path.with_suffix(".raw.png")
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.W}x{self.H}", "-i", "-",
                        "-frames:v", "1", str(tmp)], input=self.img.tobytes(), check=True)
        font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        filt = ",".join(f"drawtext=fontfile={font}:text='{t}':x=30:y={30 + 44 * i}:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.6"
                        for i, t in enumerate(legend))
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-vf", f"{filt},scale=1400:-1", str(path)], check=True)
        tmp.unlink()


GREEN, YELLOW, RED = (40, 220, 60), (250, 210, 40), (230, 40, 40)
COL = {"green": GREEN, "yellow": YELLOW, "red": RED}


def main(cell_path):
    cell = json.load(open(cell_path))
    out = REPO / "data" / "sim" / "plan"
    out.mkdir(parents=True, exist_ok=True)
    pl = Planner(cell, controller_yaw_deg=cell.get("controller_yaw_deg"), tcp_z_mm=cell.get("tcp_z_mm"))
    td = TopDown()
    summary = {"cell": str(cell_path)}
    lines = []

    # ---- 1. calibration touches (fingertip 5 mm above the marker; touching the marker itself is allowed)
    mk = cell["marker"]
    mex, mey = axes(mk["yaw_deg"])
    mc = np.array(mk["centre"], dtype=float)
    h = mk["size"] / 2
    touch = {"centre": mc, "corner TL": mc - h * mex + h * mey, "corner TR": mc + h * mex + h * mey,
             "corner BR": mc + h * mex - h * mey, "corner BL": mc - h * mex - h * mey}
    yaws = [mk["yaw_deg"] + k for k in (0, 90, 180, 270)]
    summary["calibration_touches"] = {}
    lines.append("1) CALIBRATION TOUCHES (gripper pointing down, fingertip at the marker; box not yet placed)")
    q_touch_example = None
    box_geoms = tuple(n for n in pl.obstacle_names if n.startswith("box_"))
    for name, p in touch.items():
        st, q, yaw, why = pl.pick_check(p + [0, 0, 0.005], yaws, allowed_at_target=("marker_board",) + box_geoms)  # box not there yet
        st_box, _, _, why_box = pl.pick_check(p + [0, 0, 0.005], yaws, allowed_at_target=("marker_board",))
        summary["calibration_touches"][name] = {"status": st, "gripper_yaw_deg": yaw, "reason": why, "with_box_in_place": st_box,
                                                "joints_deg": None if q is None else np.degrees(q).round(1).tolist()}
        lines.append(f"   {name:10s} {'OK' if st == 'green' else 'PROBLEM (' + st + ': ' + why + ')'}"
                     + (f"  gripper yaw {yaw:.0f} deg" if yaw is not None else "")
                     + ("" if st_box == "green" else f"   [blocked if the box is already in place: {why_box}]"))
        td.dot(p, COL[st], 12)
        if st == "green" and name.startswith("corner") and q_touch_example is None:
            q_touch_example = q
    td.poly([touch[k] for k in ("corner TL", "corner TR", "corner BR", "corner BL")], (255, 255, 255), 3)

    # ---- 2. belt pick map
    b = cell["belt"]
    bex, bey = axes(b["yaw_deg"])
    bc = np.array([*b["centre"], b["top_z"] + PICK_ABOVE_BELT])
    near_end_sign = 1 if np.dot(bex, mc - bc) > 0 else -1  # belt end nearest the marker/box
    belt_yaws = [b["yaw_deg"] + 90, b["yaw_deg"]]  # fingers across the belt, then along it
    us = np.arange(-b["length"] / 2 + 0.03, b["length"] / 2 - 0.03 + 1e-9, BELT_GRID)
    vs = np.arange(-b["width"] / 2 + 0.04, b["width"] / 2 - 0.04 + 1e-9, BELT_GRID)
    belt_res, q_warm, q_pick_example = [], None, None
    for u in us:
        for v in vs:
            p = bc + u * bex + v * bey
            st, q, yaw, why = pl.pick_check(p, belt_yaws, allowed_at_target=("belt",), q_warm=q_warm)
            if q is not None:
                q_warm = q
            dist_from_near_end = b["length"] / 2 - near_end_sign * u
            belt_res.append((dist_from_near_end, v, st))
            td.dot(p, COL[st], 6)
            if st == "green" and abs(v) < BELT_GRID and q_pick_example is None and dist_from_near_end > 0.3:
                q_pick_example = q
    belt_res = np.array(belt_res, dtype=object)
    centre_line = [r for r in belt_res if abs(r[1]) < BELT_GRID / 2 + 1e-6]
    green_d = sorted(r[0] for r in centre_line if r[2] == "green")
    frac = {k: float(np.mean([r[2] == k for r in belt_res])) for k in ("green", "yellow", "red")}
    summary["belt"] = {"fraction": frac, "centre_line_green_from_m": green_d[0] if green_d else None,
                       "centre_line_green_to_m": green_d[-1] if green_d else None,
                       "measured_from": "belt end nearest the marker/box"}
    lines.append("2) BELT PICKS (top-down, fingertip 2 cm above the belt, 15 cm hover)")
    lines.append(f"   green {frac['green']*100:.0f}%  yellow {frac['yellow']*100:.0f}%  red {frac['red']*100:.0f}% of the belt surface")
    if green_d:
        lines.append(f"   along the belt centre line: pickable from {green_d[0]*100:.0f} cm to {green_d[-1]*100:.0f} cm,"
                     f" measured from the belt end nearest the box/marker")

    # ---- 3. drops into the box (as placed in cell.json)
    bx = cell["box"]
    kx, ky = axes(bx["yaw_deg"])
    L, W, Hh = bx["size"]
    bb = np.array(bx["centre"], dtype=float)
    td.poly([bb + sx * L / 2 * kx + sy * W / 2 * ky for sx, sy in [(1, 1), (1, -1), (-1, -1), (-1, 1)]], (255, 150, 30), 4)
    box_res, q_drop_example = [], None
    for a in np.arange(-L / 2 + 0.07, L / 2 - 0.07 + 1e-9, BOX_GRID):
        for c in np.arange(-W / 2 + 0.07, W / 2 - 0.07 + 1e-9, BOX_GRID):
            p_in = bb + a * kx + c * ky + [0, 0, 0.10]  # fingertip 10 cm above the box floor (5 cm below the rim)
            st, q, yaw, why = pl.pick_check(p_in, [bx["yaw_deg"], bx["yaw_deg"] + 90])
            box_res.append(st)
            td.dot(p_in, COL[st], 10)
            if st == "green" and q_drop_example is None:
                q_drop_example = q
    bfrac = {k: float(np.mean([r == k for r in box_res])) for k in ("green", "yellow", "red")}
    summary["box_drop"] = {"fraction": bfrac}
    lines.append("3) BOX DROPS (reach 5 cm below the rim, hover above)")
    lines.append(f"   green {bfrac['green']*100:.0f}%  yellow (only from above) {bfrac['yellow']*100:.0f}%  red {bfrac['red']*100:.0f}% of the box opening")

    # ---- 4. where could the box go? (ignoring the box walls: release 5 cm above a 15 cm box)
    lt = cell["lower_table"]
    lex, ley = axes(lt["yaw_deg"])
    corner = np.array([*lt["corner_near_belt_and_robot"], lt["top_z"]])
    place_ok = []
    for a in np.arange(0.15, 1.0, BOX_GRID):
        for c in np.arange(0.12, 0.62, BOX_GRID):
            p = corner - a * lex - c * ley + [0, 0, Hh + 0.05]
            if np.linalg.norm(p[:2] - mc[:2]) < 0.25:  # keep clear of the marker
                continue
            st, _, _, _ = pl.pick_check(p, [lt["yaw_deg"], lt["yaw_deg"] + 90], allowed_at_target=("box_floor",))
            place_ok.append((p, st))
            if st == "green":
                td.dot(p, (120, 180, 255), 5)
    good = [p for p, st in place_ok if st == "green"]
    summary["box_placement_green_points_lower_table"] = len(good)
    if good:
        g = np.array(good)
        best = g[np.argmin(np.linalg.norm(g[:, :2] - bb[:2], axis=1))]  # working spot nearest to the planned box position
        move = best[:2] - bb[:2]
        summary["box_suggested_centre"] = best[:2].round(3).tolist()
        lines.append("4) BOX PLACEMENT on the lower table (light blue dots = box centre positions where drops work)")
        lines.append(f"   {len(good)} of {len(place_ok)} candidate spots work; nearest to the planned spot: move the box centre"
                     f" {np.dot(move, kx[:2])*100:+.0f} cm along the belt (+ = toward the robot) and"
                     f" {np.dot(move, ky[:2])*100:+.0f} cm across (+ = toward the belt)")

    # ---- 5. parking pose
    belt_pts = np.array([bc + u * bex + v * bey for u in us for v in vs])
    box_pts = np.array([bb + a * kx + c * ky + [0, 0, z] for a in (-L / 2, 0, L / 2) for c in (-W / 2, 0, W / 2) for z in (0, Hh)])
    work_pts = np.vstack([belt_pts, box_pts, list(touch.values())])
    tree = cKDTree(work_pts)
    best_park, best_compact = None, None
    for pan in np.radians(np.arange(-180, 180, 15)):
        for lift in np.radians([-30, -45, -60, -90, -120, -135, -150]):
            for elb in np.radians([-160, -150, -135, -120, 120, 135, 150, 160]):
                for w1 in np.radians([-90, 0, 90, 180]):
                    q = np.array([pan, lift, elb, w1, -np.pi / 2, 0.0])
                    if pl.contacts(q):
                        continue
                    P = robot_points(pl, q)
                    if P[:, 2].min() < pl.base[2] - 0.02:  # keep the whole arm above the robot's own table level
                        continue
                    clear = tree.query(P)[0].min()
                    if best_park is None or clear > best_park[0]:
                        best_park = (clear, q)
                    if P[:, 2].max() < pl.base[2] + 0.55 and (best_compact is None or clear > best_compact[0]):
                        best_compact = (clear, q)
    if best_park:
        clear, qp = best_park
        summary["parking"] = {"joints_deg": np.degrees(qp).round(0).tolist(), "clearance_to_work_area_m": round(float(clear), 3)}
        lines.append("5) PARKING POSE (set these joint angles on the pendant; arm stays still for the whole session)")
        lines.append(f"   joints (deg) base..wrist3: {np.degrees(qp).round(0).astype(int).tolist()}  ->  closest approach to the work area {clear*100:.0f} cm")
        if best_compact:
            cc, qc = best_compact
            summary["parking_compact"] = {"joints_deg": np.degrees(qc).round(0).tolist(), "clearance_to_work_area_m": round(float(cc), 3)}
            lines.append(f"   compact alternative (arm stays below 55 cm above its base): {np.degrees(qc).round(0).astype(int).tolist()}"
                         f"  ->  closest approach {cc*100:.0f} cm")
            qp = qc
        for p in robot_points(pl, qp, 15)[::3]:
            td.dot(p, (230, 80, 230), 3)

    # ---- outputs
    lines.insert(0, f"Robot base (scan coords): {pl.base.round(3).tolist()}  |  marker + box positions are ESTIMATES until calibration")
    (out / "plan_summary.txt").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(out / "plan_summary.json", "w"), indent=1, default=float)
    td.dot(pl.base, (230, 80, 230), 20)
    td.save(out / "plan_topdown.png", ["GREEN = pick/drop OK   YELLOW = only from above   RED = out of reach",
                                       "big dots = 5 marker touches   orange = box   light blue = good box spots   magenta = robot + parking pose"])
    render_poses(pl, out / "plan_poses.png", {"calibration touch": q_touch_example, "belt pick": q_pick_example,
                                              "box drop": q_drop_example, "parking (compact)": best_compact[1] if best_compact else (best_park[1] if best_park else None)})
    print("\n".join(lines))
    print(f"\nsaved {out/'plan_topdown.png'}, {out/'plan_poses.png'}, {out/'plan_summary.txt'}, {out/'plan_summary.json'}")


def render_poses(pl, path, poses):
    r = mujoco.Renderer(pl.m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [pl.base[0] - 0.25, pl.base[1] + 0.0, 0.95]
    cam.distance, cam.azimuth, cam.elevation = 2.0, 110.0, -30.0
    font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    tiles = []
    for name, q in poses.items():
        pl.d.qpos[:] = 0
        if q is not None:
            pl.d.qpos[:6] = q
        mujoco.mj_forward(pl.m, pl.d)
        r.update_scene(pl.d, camera=cam)
        img = r.render()
        if q is None:
            img = (img * 0.3).astype(np.uint8)
        tiles.append(img)
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    labels = ",".join(f"drawtext=fontfile={font}:text='{n}{'' if q is not None else ' (none found)'}':x={10 + 640 * (i % 2)}:y={10 + 480 * (i // 2)}:"
                      f"fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6" for i, (n, q) in enumerate(poses.items()))
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1280x960", "-i", "-",
                    "-vf", labels, "-frames:v", "1", str(path)], input=grid.tobytes(), check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Plan a recording session in the digital twin.")
    ap.add_argument("--cell", type=pathlib.Path, default=REPO / "data" / "sim" / "cell.json")
    main(ap.parse_args().cell)
