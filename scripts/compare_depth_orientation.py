#!/usr/bin/env python3
"""Compare Isaac vs MuJoCo standing depth orientation (row means + flip corr).

Decision rule (printed as recommendation)::

  - If corr(I, flipud(M)) >> corr(I, M) AND bot-top signs differ → apply flipud after raycast
  - If vertical signs match but Isaac has more near pixels in one half → content / self-hit gap
  - If corr(I, fliplr(M)) is best → consider fliplr (unexpected for this camera)

Example::

    .venv/bin/python scripts/compare_depth_orientation.py \\
      --isaac artifacts/isaac_standing_depth.npz \\
      --mujoco artifacts/mujoco_standing_depth.npz
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEPLOY = os.path.normpath(os.path.join(_ROOT, "..", "deploy_real"))
if _DEPLOY not in sys.path:
    sys.path.insert(0, _DEPLOY)


def _as_hw(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float64)
    d = np.squeeze(d)
    if d.ndim != 2:
        raise ValueError(f"expected HxW after squeeze, got {d.shape}")
    return d


def _load_raw(path: str) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    if "raw_m" in z.files:
        return _as_hw(z["raw_m"])
    if "depth" in z.files:
        # env_dump D_norm NCHW — map to metres-ish for layout only
        d = z["depth"]
        if d.ndim >= 4:
            d = d[0]
        d = _as_hw(d)
        if float(d.min()) >= -0.6 and float(d.max()) <= 0.6:
            return (d + 0.5) * 6.0
        return d
    raise KeyError(f"{path}: need key raw_m or depth; got {z.files}")


def _resize_hw(img: np.ndarray, h: int, w: int) -> np.ndarray:
    try:
        from rl_deploy.core.depth_camera import _resize_hw as resize_hw

        return np.asarray(resize_hw(img.astype(np.float32), h, w), dtype=np.float64)
    except Exception:
        # Nearest-neighbor fallback (no OpenCV / scipy required)
        ys = (np.arange(h) + 0.5) * img.shape[0] / h
        xs = (np.arange(w) + 0.5) * img.shape[1] / w
        yi = np.clip(ys.astype(np.int64), 0, img.shape[0] - 1)
        xi = np.clip(xs.astype(np.int64), 0, img.shape[1] - 1)
        return img[yi][:, xi]


def _band_stats(raw: np.ndarray, *, k: int = 5) -> dict[str, float]:
    h = raw.shape[0]
    k = min(k, max(1, h // 2))
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


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel()
    b = b.ravel()
    if a.size != b.size:
        raise ValueError(f"size mismatch {a.size} vs {b.size}")
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Isaac/MuJoCo depth vertical orientation.")
    ap.add_argument("--isaac", required=True, help="npz with raw_m (or depth dump)")
    ap.add_argument("--mujoco", required=True, help="npz with raw_m")
    ap.add_argument("--band", type=int, default=5)
    ap.add_argument(
        "--margin",
        type=float,
        default=0.05,
        help="min corr improvement to prefer flipud/fliplr over identity",
    )
    args = ap.parse_args()

    isaac = _load_raw(args.isaac)
    mujoco = _load_raw(args.mujoco)
    if isaac.shape != mujoco.shape:
        mujoco = _resize_hw(mujoco, isaac.shape[0], isaac.shape[1])

    si = _band_stats(isaac, k=args.band)
    sm = _band_stats(mujoco, k=args.band)

    c_id = _corr(isaac, mujoco)
    c_ud = _corr(isaac, np.flipud(mujoco))
    c_lr = _corr(isaac, np.fliplr(mujoco))
    c_ud_lr = _corr(isaac, np.flipud(np.fliplr(mujoco)))

    print("[compare_depth_orientation]")
    print(
        f"  isaac  shape={isaac.shape} top={si['top_mean']:.3f} mid={si['mid_mean']:.3f} "
        f"bot={si['bot_mean']:.3f} bot-top={si['bot_minus_top']:+.3f}"
    )
    print(
        f"  mujoco shape={mujoco.shape} top={sm['top_mean']:.3f} mid={sm['mid_mean']:.3f} "
        f"bot={sm['bot_mean']:.3f} bot-top={sm['bot_minus_top']:+.3f}"
    )
    print(
        f"  near(<1.5m) isaac top/bot={si['near_frac_top_half']:.3f}/{si['near_frac_bot_half']:.3f}  "
        f"mujoco top/bot={sm['near_frac_top_half']:.3f}/{sm['near_frac_bot_half']:.3f}"
    )
    print(
        f"  corr identity={c_id:.4f}  flipud={c_ud:.4f}  fliplr={c_lr:.4f}  "
        f"flipud+fliplr={c_ud_lr:.4f}"
    )

    sign_opposite = (si["bot_minus_top"] * sm["bot_minus_top"]) < 0
    scores = [
        ("identity", c_id),
        ("flipud", c_ud),
        ("fliplr", c_lr),
        ("flipud+fliplr", c_ud_lr),
    ]
    best = max(scores, key=lambda x: (-1.0 if np.isnan(x[1]) else x[1]))
    print(f"  best transform: {best[0]} (corr={best[1]:.4f})")
    print(f"  bot-top signs opposite: {sign_opposite}")

    if (
        not np.isnan(c_ud)
        and c_ud >= c_id + args.margin
        and c_ud >= c_lr - 1e-9
        and sign_opposite
    ):
        rec = "APPLY_FLIPUD"
        detail = (
            "corr(I, flipud(M)) clearly better and bot-top signs differ → "
            "flipud after raycast in depth.py (do NOT change ray Y)."
        )
    elif not np.isnan(c_lr) and c_lr >= c_id + args.margin and c_lr > c_ud:
        rec = "CONSIDER_FLIPLR"
        detail = "left-right mismatch dominates; unexpected — inspect camera quat / X sign."
    elif not sign_opposite and (
        abs(si["near_frac_bot_half"] - sm["near_frac_bot_half"]) > 0.15
        or abs(si["near_frac_top_half"] - sm["near_frac_top_half"]) > 0.15
    ):
        rec = "CONTENT_GAP"
        detail = (
            "vertical layout already same-sign; near-pixel halves differ → "
            "self-hit / body occlusion (Phase 3), not flipud."
        )
    elif best[0] == "identity" or (not np.isnan(c_id) and c_id + args.margin >= best[1]):
        rec = "KEEP"
        detail = "identity best or close enough → keep current ray Y; dig content / camera pitch."
    else:
        rec = "INVESTIGATE"
        detail = f"best={best[0]}; re-check poses and FOV before changing axes."

    print(f"  RECOMMENDATION: {rec}")
    print(f"  {detail}")
    code = {"APPLY_FLIPUD": 2, "CONSIDER_FLIPLR": 3, "CONTENT_GAP": 0, "KEEP": 0}.get(rec, 3)
    sys.exit(code)


if __name__ == "__main__":
    main()
