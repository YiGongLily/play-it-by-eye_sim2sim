#!/usr/bin/env python3
"""Run X2 Depth CNN MuJoCo sim2sim (flat ground, Phase-0).

Usage (from this repo, with local .venv):
  .venv/bin/python run_sim2sim.py
  .venv/bin/python run_sim2sim.py --config configs/sim2sim.yaml --duration 30
"""

from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Must set before importing mujoco (see x2_depth_sim2sim.cfg).
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")


def main() -> None:
    ap = argparse.ArgumentParser(description="X2 Depth CNN MuJoCo sim2sim")
    ap.add_argument(
        "--config",
        default=os.path.join(_ROOT, "configs", "sim2sim.yaml"),
        help="sim2sim yaml (use configs/sim2sim_stairs.yaml for stairs+slope)",
    )
    ap.add_argument("--duration", type=float, default=None, help="override duration_s")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-depth-window", action="store_true")
    ap.add_argument("--no-sync", action="store_true", help="run as fast as possible")
    ap.add_argument("--viser-port", type=int, default=None)
    args = ap.parse_args()

    from x2_depth_sim2sim.cfg import load_all
    from x2_depth_sim2sim.runner import Sim2SimRunner

    s2s, deploy = load_all(args.config)
    if args.duration is not None:
        s2s.setdefault("sim", {})["duration_s"] = float(args.duration)
    if args.no_viewer:
        s2s.setdefault("viewer", {})["enable"] = False
    if args.no_depth_window:
        s2s.setdefault("viewer", {})["show_depth"] = False
    if args.no_sync:
        s2s.setdefault("viewer", {})["sync_realtime"] = False
    if args.viser_port is not None:
        s2s.setdefault("viewer", {})["port"] = int(args.viser_port)
    runner = Sim2SimRunner(s2s, deploy)
    runner.run()


if __name__ == "__main__":
    main()
