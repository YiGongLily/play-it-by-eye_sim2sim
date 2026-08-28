# X2 Depth CNN Sim2Sim (MuJoCo + viser)

[English](README.md) | [中文](README_zh.md)

Companion MuJoCo closed-loop for [**Play It By Eye**](https://github.com/YiGongLily/play-it-by-eye) (`play-it-by-eye`).

## Overview

MuJoCo + viser closed-loop sim2sim for the **LeggedLab Depth Rough-CNN** policy on X2.

- Does **not** modify the [Play It By Eye](https://github.com/YiGongLily/play-it-by-eye) training code; runs in a plain Python venv (no Isaac Lab conda env required).
- Policy pipeline: **Play It By Eye export ONNX → sibling `deploy_real` → this repo**.

## Prerequisites

Clone siblings next to this repo (same parent directory):

| Repo | Role |
|------|------|
| [play-it-by-eye](https://github.com/YiGongLily/play-it-by-eye) | Train / export ONNX |
| `deploy_real` | ONNX pack, `deployment_alignment.yaml`, deploy config |
| `x2_rl_deploy` | MuJoCo robot meshes (`meshdir` in `assets/x2_with_camera.xml`) |

Default layout:

```text
workspace/
├── play-it-by-eye/          # or local name: leggedlab
├── deploy_real/
├── x2_rl_deploy/
└── play-it-by-eye_sim2sim/  # this repo (local name may be x2_depth_sim2sim)
```

If your mesh path differs, edit `meshdir` in `assets/x2_with_camera.xml` (relative to that XML file).

For the full sim2sim workflow, see the docs under [Play It By Eye](https://github.com/YiGongLily/play-it-by-eye).

## Installation

```bash
cd /path/to/play-it-by-eye_sim2sim   # or your local checkout name
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Smoke Tests

```bash
.venv/bin/python scripts/smoke_ort.py    # pass=True
.venv/bin/python scripts/smoke_loop.py   # headless
```

## Run (viser)

```bash
# Flat ground
.venv/bin/python run_sim2sim.py --viser-port 8080

# Stairs + slope (Easy / Medium / Hard via viser Terrain Difficulty panel)
.venv/bin/python run_sim2sim.py --config configs/sim2sim_stairs.yaml --viser-port 8080
```

Open the URL printed in the terminal. Use the right-side panel for speed and terrain difficulty.

## Stairs Difficulty

| Difficulty | Stairs | Slope | Spawn y |
|------------|--------|-------|---------|
| Easy | 4×8 cm | 0.20 | 0 |
| Medium | 5×12 cm | 0.25 | 5 |
| Hard | 5×20 cm | 0.38 | 10 |
