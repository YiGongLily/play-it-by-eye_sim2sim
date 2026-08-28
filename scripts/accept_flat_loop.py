#!/usr/bin/env python3
"""Headless flat-ground closed-loop acceptance: vx=0 / vx=0.5 for a few seconds.

Reports pelvis dx, dy, z, and pitch proxy (quat) so we can compare against the
known failure mode (backward lean / retreat under real depth).

Example::

    .venv/bin/python scripts/accept_flat_loop.py --vx 0 --duration 4
    .venv/bin/python scripts/accept_flat_loop.py --vx 0.5 --duration 4
    .venv/bin/python scripts/accept_flat_loop.py --vx 0 --duration 4 --depth-mode constant
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")


def _pelvis_xyz_quat(env) -> tuple[np.ndarray, np.ndarray]:
    # freejoint qpos: x y z qw qx qy qz
    q = np.asarray(env.data.qpos[:7], dtype=np.float64)
    return q[:3].copy(), q[3:7].copy()


def _pitch_from_quat_wxyz(q: np.ndarray) -> float:
    """Approximate body pitch (rad): positive = nose up / lean back."""
    w, x, y, z = q
    # pitch about Y for wxyz
    sinp = 2.0 * (w * y - z * x)
    sinp = float(np.clip(sinp, -1.0, 1.0))
    return float(np.arcsin(sinp))


def main() -> None:
    ap = argparse.ArgumentParser(description="Flat-ground sim2sim acceptance loop.")
    ap.add_argument("--config", default=os.path.join(_ROOT, "configs", "sim2sim.yaml"))
    ap.add_argument("--vx", type=float, default=0.0)
    ap.add_argument("--vy", type=float, default=0.0)
    ap.add_argument("--wz", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=4.0)
    ap.add_argument(
        "--depth-mode",
        choices=("real", "constant"),
        default="real",
        help="real=raycast; constant=fill mid-range depth (ablation)",
    )
    ap.add_argument("--constant-m", type=float, default=6.0, help="metres when --depth-mode constant")
    args = ap.parse_args()

    from x2_depth_sim2sim.cfg import load_all
    from x2_depth_sim2sim.runner import Sim2SimRunner

    s2s, deploy = load_all(args.config)
    s2s.setdefault("viewer", {})["enable"] = False
    s2s.setdefault("viewer", {})["show_depth"] = False
    s2s.setdefault("keyboard", {})["enable"] = False
    s2s.setdefault("sim", {})["duration_s"] = float(args.duration)

    runner = Sim2SimRunner(s2s, deploy)
    runner.reset()
    runner.cmd[:] = (float(args.vx), float(args.vy), float(args.wz))

    # Monkeypatch depth feed for constant ablation
    if args.depth_mode == "constant":
        const = float(args.constant_m)
        h, w = runner.depth.height, runner.depth.width
        d_max = runner.depth.d_max

        def _const_nchw(_data=None):
            raw = np.full((h, w), const, dtype=np.float32)
            runner.depth._last_raw = raw
            from rl_deploy.core.depth_camera import normalize_depth_nchw

            nchw = normalize_depth_nchw(raw, height=h, width=w, d_max=d_max)
            runner.depth._last_nchw = nchw
            return nchw

        runner.depth.render_nchw = _const_nchw  # type: ignore[method-assign]

    xyz0, quat0 = _pelvis_xyz_quat(runner.env)
    pitch0 = _pitch_from_quat_wxyz(quat0)
    sim_t = 0.0
    action = np.zeros(runner.env.n, dtype=np.float64)

    while sim_t < float(args.duration):
        # Set cmd before policy_step (it calls _read_cmd at start).
        runner.cmd[:] = (float(args.vx), float(args.vy), float(args.wz))
        action, _ms = runner.policy_step(t_s=sim_t)
        for _ in range(runner.decimation):
            runner.env.apply_pd(action)
            runner.env.step(1)
            sim_t += runner.timestep

    xyz1, quat1 = _pelvis_xyz_quat(runner.env)
    pitch1 = _pitch_from_quat_wxyz(quat1)
    dx, dy, dz = (xyz1 - xyz0).tolist()
    z = float(xyz1[2])
    dpitch = pitch1 - pitch0

    print("[accept_flat_loop]")
    print(
        f"  depth_mode={args.depth_mode} vx={args.vx} duration={args.duration:.1f}s "
        f"sim_t={sim_t:.2f}"
    )
    print(f"  dx={dx:+.3f} m  dy={dy:+.3f} m  z={z:.3f} m  dz={dz:+.3f}")
    print(f"  pitch0={pitch0:+.3f} rad  pitch1={pitch1:+.3f} rad  dpitch={dpitch:+.3f}")
    # Heuristic flags for the known failure mode
    retreat = dx < -0.3
    lean_back = pitch1 > 0.15
    standing = z > 0.45
    print(f"  flags: retreat={retreat} lean_back={lean_back} standing_z={standing}")
    runner.close()

    # Non-zero exit if obvious collapse
    if not standing or not np.isfinite(xyz1).all():
        sys.exit(1)


if __name__ == "__main__":
    main()
