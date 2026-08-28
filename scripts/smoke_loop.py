#!/usr/bin/env python3
"""Headless smoke: load scene, reset, render depth, one policy step (no viewer)."""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")


def main() -> None:
    from x2_depth_sim2sim.cfg import load_all
    from x2_depth_sim2sim.runner import Sim2SimRunner

    cfg = os.path.join(_ROOT, "configs", "sim2sim.yaml")
    s2s, deploy = load_all(cfg)
    s2s.setdefault("viewer", {})["enable"] = False
    s2s.setdefault("viewer", {})["show_depth"] = False
    s2s.setdefault("keyboard", {})["enable"] = False
    s2s.setdefault("sim", {})["duration_s"] = 0.2

    runner = Sim2SimRunner(s2s, deploy)
    runner.reset()
    act, ms = runner.policy_step(t_s=0.0)
    for _ in range(runner.decimation):
        runner.env.apply_pd(act)
        runner.env.step(1)
    raw = runner.depth.last_raw
    assert raw is not None and raw.shape == (40, 64), raw.shape if raw is not None else None
    assert act.shape == (31,), act.shape
    assert np.isfinite(act).all(), act
    print(f"[smoke_loop] ok action[0:3]={act[:3]} infer_ms={ms:.2f} depth_mean={float(raw.mean()):.3f}")
    runner.close()


if __name__ == "__main__":
    main()
