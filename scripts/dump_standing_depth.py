#!/usr/bin/env python3
"""Freeze default pose, dump MuJoCo head depth (raw m + D_norm) for Isaac alignment.

Does not run the policy — only ``reset`` + ``mj_forward`` via ``render_meters``.

Prints miss / near-clip-reject / valid fractions and optional Isaac upper-half near
comparison when ``artifacts/isaac_standing_depth.npz`` is present.

Example::

    cd /path/to/x2_depth_sim2sim
    .venv/bin/python scripts/dump_standing_depth.py \\
      --out artifacts/mujoco_standing_depth.npz
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


def _row_band_stats(raw: np.ndarray, *, k: int = 5) -> dict[str, float]:
    h = raw.shape[0]
    k = min(k, h // 2)
    mid0 = max(0, h // 2 - k // 2)
    mid1 = min(h, mid0 + k)
    top = float(raw[:k].mean())
    mid = float(raw[mid0:mid1].mean())
    bot = float(raw[-k:].mean())
    return {
        "top_mean": top,
        "mid_mean": mid,
        "bot_mean": bot,
        "bot_minus_top": bot - top,
        "near_frac_top_half": float((raw[: h // 2] < 1.5).mean()),
        "near_frac_bot_half": float((raw[h // 2 :] < 1.5).mean()),
    }


def _print_diag(diag: dict[str, float] | None) -> None:
    if not diag:
        print("  diag: (none)")
        return
    print(
        "  diag fractions: "
        f"true_miss={diag['true_miss_frac']:.4f}  "
        f"near_clip={diag['near_clip_frac']:.4f}  "
        f"near_clip_rejected_robot={diag['near_clip_rejected_robot_frac']:.4f}  "
        f"valid={diag['valid_frac']:.4f}  "
        f"far_fill={diag['far_fill_frac']:.4f}"
    )
    print(
        f"  clip_near={diag['clip_near']:.3f} clip_far={diag['clip_far']:.3f}  "
        f"robot_hit_n={diag['robot_hit_n']:.0f} robot_kept_n={diag['robot_kept_n']:.0f}"
    )
    print(
        "  robot hit plane p10/p50/p90="
        f"{diag['robot_hit_plane_p10']:.4f}/"
        f"{diag['robot_hit_plane_p50']:.4f}/"
        f"{diag['robot_hit_plane_p90']:.4f}  "
        f"kept_p50={diag['robot_kept_plane_p50']:.4f}"
    )


def _compare_isaac(raw: np.ndarray, isaac_path: str) -> None:
    if not os.path.isfile(isaac_path):
        print(f"  isaac compare: skip (missing {isaac_path})")
        return
    data = np.load(isaac_path)
    key = "raw_m" if "raw_m" in data.files else ("depth" if "depth" in data.files else None)
    if key is None:
        print(f"  isaac compare: skip (no raw_m in {isaac_path}: {data.files})")
        return
    isaac = np.asarray(data[key], dtype=np.float32)
    if isaac.ndim == 3:
        isaac = isaac[0]
    if isaac.shape != raw.shape:
        print(f"  isaac compare: shape mismatch mujoco={raw.shape} isaac={isaac.shape}")
        return
    h = raw.shape[0]
    m_top = float((raw[: h // 2] < 1.5).mean())
    i_top = float((isaac[: h // 2] < 1.5).mean())
    m_bot = float((raw[h // 2 :] < 1.5).mean())
    i_bot = float((isaac[h // 2 :] < 1.5).mean())
    print(
        f"  isaac compare ({os.path.basename(isaac_path)}): "
        f"near(<1.5) top_half M={m_top:.3f} I={i_top:.3f}  "
        f"bot_half M={m_bot:.3f} I={i_bot:.3f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Dump MuJoCo standing-pose depth tensors.")
    ap.add_argument(
        "--config",
        default=os.path.join(_ROOT, "configs", "sim2sim.yaml"),
        help="sim2sim yaml path",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(_ROOT, "artifacts", "mujoco_standing_depth.npz"),
        help="output npz path",
    )
    ap.add_argument(
        "--isaac",
        default=os.path.join(_ROOT, "artifacts", "isaac_standing_depth.npz"),
        help="optional Isaac dump for upper-half near frac compare",
    )
    ap.add_argument("--band", type=int, default=5, help="rows for top/mid/bot means")
    args = ap.parse_args()

    from x2_depth_sim2sim.cfg import load_all
    from x2_depth_sim2sim.runner import Sim2SimRunner

    s2s, deploy = load_all(args.config)
    s2s.setdefault("viewer", {})["enable"] = False
    s2s.setdefault("viewer", {})["show_depth"] = False
    s2s.setdefault("keyboard", {})["enable"] = False

    runner = Sim2SimRunner(s2s, deploy)
    runner.reset()

    # Forward-only depth (no policy / no mj_step).
    raw = runner.depth.render_meters(runner.env.data)
    nchw = runner.depth.render_nchw(runner.env.data)
    diag = runner.depth.last_diag
    cam_id = runner.depth.camera_id
    cam_xpos = np.asarray(runner.env.data.cam_xpos[cam_id], dtype=np.float64).copy()
    cam_xmat = np.asarray(runner.env.data.cam_xmat[cam_id], dtype=np.float64).copy()

    stats = _row_band_stats(raw, k=args.band)
    print("[dump_standing_depth] MuJoCo standing pose")
    print(f"  config={os.path.abspath(args.config)}")
    print(f"  raw shape={raw.shape} min={raw.min():.3f} max={raw.max():.3f} mean={raw.mean():.3f}")
    print(f"  D_norm shape={nchw.shape} min={nchw.min():.4f} max={nchw.max():.4f}")
    print(
        f"  top{args.band}_mean={stats['top_mean']:.3f}  "
        f"mid_mean={stats['mid_mean']:.3f}  "
        f"bot{args.band}_mean={stats['bot_mean']:.3f}  "
        f"bot-top={stats['bot_minus_top']:+.3f}"
    )
    print(
        f"  near(<1.5m) top_half={stats['near_frac_top_half']:.3f}  "
        f"bot_half={stats['near_frac_bot_half']:.3f}"
    )
    _print_diag(diag)
    _compare_isaac(raw, args.isaac)
    if stats["bot_minus_top"] < 0:
        print("  layout hint: nearer (smaller D) toward bottom rows (matches README B3)")
    else:
        print("  layout hint: nearer toward top rows (opposite of README B3)")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    save_kw: dict = {
        "raw_m": raw.astype(np.float32),
        "D_norm": nchw.astype(np.float32),
        "cam_xpos": cam_xpos,
        "cam_xmat": cam_xmat,
        "source": np.asarray("mujoco_standing"),
        "band_k": np.int32(args.band),
        "top_mean": np.float32(stats["top_mean"]),
        "mid_mean": np.float32(stats["mid_mean"]),
        "bot_mean": np.float32(stats["bot_mean"]),
        "bot_minus_top": np.float32(stats["bot_minus_top"]),
        "near_frac_top_half": np.float32(stats["near_frac_top_half"]),
        "near_frac_bot_half": np.float32(stats["near_frac_bot_half"]),
    }
    if diag:
        for k, v in diag.items():
            save_kw[f"diag_{k}"] = np.float32(v)
    np.savez(out, **save_kw)
    print(f"  saved {out}")
    runner.close()


if __name__ == "__main__":
    main()
