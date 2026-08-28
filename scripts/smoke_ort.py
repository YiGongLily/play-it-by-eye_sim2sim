#!/usr/bin/env python3
"""Acceptance gate: offline ORT dump alignment (reuse deploy_real tool).

Usage:
  .venv/bin/python scripts/smoke_ort.py
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEPLOY = os.path.normpath(os.path.join(_ROOT, "..", "deploy_real"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=os.path.join(_DEPLOY, "configs", "x2", "depth_cnn_real.yaml"),
    )
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    tool = os.path.join(_DEPLOY, "tools", "offline_depth_cnn_ort.py")
    py = sys.executable
    cmd = [py, tool, "--config", args.config, "--tol", str(args.tol)]
    print("[smoke_ort]", " ".join(cmd))
    env = os.environ.copy()
    # Prefer CPU EP in headless gates
    env.setdefault("OMP_NUM_THREADS", "1")
    raise SystemExit(subprocess.call(cmd, cwd=_DEPLOY, env=env))


if __name__ == "__main__":
    main()
