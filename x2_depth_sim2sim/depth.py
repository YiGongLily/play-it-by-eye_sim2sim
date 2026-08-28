"""Head depth → deploy_real normalize_depth_nchw.

Backends:
- ``raycast`` (default): ``mj_ray`` per pixel — no OpenGL, safe with viser.
- ``egl``: ``mujoco.Renderer`` depth buffer (needs free EGL context).
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

import mujoco
import numpy as np


class DepthRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera_name: str,
        height: int = 40,
        width: int = 64,
        d_max: float = 6.0,
        clip_near: float = 0.2,
        clip_far: float | None = None,
        backend: Literal["raycast", "egl"] = "raycast",
        exclude_robot: bool = True,
        terrain_geom_group: int = 0,
        robot_visual_group: int = 1,
        ray_geom_groups: Sequence[int] | None = None,
        hold_on_far_frac: float = 0.85,
        ray_maxdist: float | None = None,
    ) -> None:
        self.model = model
        self.camera_name = camera_name
        self.camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if self.camera_id < 0:
            raise RuntimeError(f"camera '{camera_name}' not found")
        self.height = int(height)
        self.width = int(width)
        self.d_max = float(d_max)
        self.clip_near = float(clip_near)
        self.clip_far = float(clip_far if clip_far is not None else d_max)
        self.backend = backend
        self.exclude_robot = bool(exclude_robot)
        self.terrain_geom_group = int(terrain_geom_group)
        self.robot_visual_group = int(robot_visual_group)
        # Grazing floor rays need ray-length >> plane depth; clip_far*1.5 is too short.
        self.ray_maxdist = float(
            ray_maxdist if ray_maxdist is not None else max(50.0, self.clip_far * 8.0)
        )
        # If far-fill exceeds this, reuse last good frame (breaks pitch-up → sky-red spiral).
        self.hold_on_far_frac = float(hold_on_far_frac)
        self._renderer = None
        if backend == "egl":
            self._renderer = mujoco.Renderer(model, height=self.height, width=self.width)
            self._renderer.enable_depth_rendering()
        self._last_raw: np.ndarray | None = None
        self._last_nchw: np.ndarray | None = None
        self._last_diag: dict[str, float] | None = None
        self._last_good_raw: np.ndarray | None = None
        # Camera pose used for the last depth frame (FOV must use this, not post-physics).
        self._last_cam_xpos: np.ndarray | None = None
        self._last_cam_xmat: np.ndarray | None = None
        # Precompute camera-frame ray dirs (MuJoCo cam looks along -Z, +Y up).
        self._ray_dirs_cam = self._make_ray_dirs()
        # geomgroup masks for mj_multiRay (mjNGROUP). Rays ignore contype:
        #   group 0 = floor/terrain, group 1 = visual meshes (arms/legs/torso),
        #   group 3 = collision. Isaac RTX sees visual → include group 1 to match.
        #
        # MuJoCo 3.x quirk: a single mj_multiRay cannot hit both static planes and
        # dynamic bodies. flg_static=1 → terrain only; flg_static=0 → dynamics only
        # (floor miss → all-far/red). When robot is included we use a two-pass merge.
        ng = int(getattr(mujoco, "mjNGROUP", 6))

        def _mask(groups: Sequence[int]) -> np.ndarray:
            gg = np.zeros(ng, dtype=np.uint8)
            for g in groups:
                if 0 <= int(g) < ng:
                    gg[int(g)] = 1
            return gg

        if ray_geom_groups is not None:
            groups = [int(g) for g in ray_geom_groups]
            has_t = self.terrain_geom_group in groups
            has_r = self.robot_visual_group in groups
            self._two_pass = has_t and has_r
            if self._two_pass:
                self._gg_terrain = _mask([self.terrain_geom_group])
                self._gg_robot = _mask([self.robot_visual_group])
            else:
                self._gg_terrain = _mask(groups)
                self._gg_robot = None
            self.ray_geom_groups = tuple(groups)
            self.exclude_robot = not has_r
        elif self.exclude_robot:
            self._two_pass = False
            self._gg_terrain = _mask([self.terrain_geom_group])
            self._gg_robot = None
            self.ray_geom_groups = (self.terrain_geom_group,)
        else:
            self._two_pass = True
            self._gg_terrain = _mask([self.terrain_geom_group])
            self._gg_robot = _mask([self.robot_visual_group])
            self.ray_geom_groups = (self.terrain_geom_group, self.robot_visual_group)
        mode = "two-pass(static terrain + dynamic visual)" if self._two_pass else "single-pass"
        print(
            f"[depth] ray geom groups={list(self.ray_geom_groups)} "
            f"exclude_robot={self.exclude_robot} mode={mode} "
            f"ray_maxdist={self.ray_maxdist:.1f} hold_on_far_frac={self.hold_on_far_frac:.2f}"
        )

    def _make_ray_dirs(self) -> np.ndarray:
        """Unit rays in camera frame for each pixel, shape (H, W, 3)."""
        fovy = float(self.model.cam_fovy[self.camera_id])
        aspect = self.width / max(self.height, 1)
        fy = 0.5 * self.height / math.tan(math.radians(fovy) * 0.5)
        fx = fy  # square pixels; HFOV implied by aspect
        ys, xs = np.meshgrid(
            np.arange(self.height, dtype=np.float64),
            np.arange(self.width, dtype=np.float64),
            indexing="ij",
        )
        # Pixel centers; OpenGL-style NDC with +Y up → row 0 is top.
        # Standing-pose dump (artifacts/*_standing_depth.npz, 2026-08-19): Isaac and
        # MuJoCo both have nearer rows at the TOP (bot-top > 0); corr(I,M)≈0.92 while
        # corr(I, flipud(M))≈-0.52 → do NOT flipud / do NOT invert this Y sign.
        # Blind Y-sign flip previously caused action explosion (joints lock).
        x = (xs + 0.5 - 0.5 * self.width) / fx
        y = -(ys + 0.5 - 0.5 * self.height) / fy
        # Look along -Z
        z = -np.ones_like(x)
        dirs = np.stack([x, y, z], axis=-1)
        dirs /= np.linalg.norm(dirs, axis=-1, keepdims=True).clip(min=1e-12)
        return dirs

    def render_meters(self, data: mujoco.MjData) -> np.ndarray:
        if self.backend == "egl" and self._renderer is not None:
            try:
                return self._render_egl(data)
            except Exception as e:
                print(f"[depth] EGL render failed ({e}); falling back to raycast")
                self.backend = "raycast"
        return self._render_raycast(data)

    def _render_egl(self, data: mujoco.MjData) -> np.ndarray:
        assert self._renderer is not None
        self._renderer.update_scene(data, camera=self.camera_name)
        depth = np.asarray(self._renderer.render(), dtype=np.float32)
        bad = ~np.isfinite(depth) | (depth <= 0.0)
        depth = np.clip(depth, self.clip_near, self.clip_far)
        depth[bad] = self.clip_far
        self._last_raw = depth
        self._last_cam_xpos = np.asarray(
            data.cam_xpos[self.camera_id], dtype=np.float64
        ).reshape(3).copy()
        self._last_cam_xmat = np.asarray(
            data.cam_xmat[self.camera_id], dtype=np.float64
        ).reshape(3, 3).copy()
        self._last_diag = {
            "nray": float(depth.size),
            "true_miss_frac": float(bad.mean()),
            "near_clip_frac": 0.0,
            "near_clip_rejected_robot_frac": 0.0,
            "valid_frac": float((~bad).mean()),
            "far_fill_frac": float((depth >= self.clip_far - 1e-6).mean()),
            "far_fill_raw": float((depth >= self.clip_far - 1e-6).mean()),
            "terrain_hit_frac": float("nan"),
            "look_z": float((-self._last_cam_xmat[:, 2])[2]),
            "held": 0.0,
            "robot_hit_n": float("nan"),
            "robot_kept_n": float("nan"),
            "robot_hit_plane_p10": float("nan"),
            "robot_hit_plane_p50": float("nan"),
            "robot_hit_plane_p90": float("nan"),
            "robot_kept_plane_p50": float("nan"),
            "clip_near": float(self.clip_near),
            "clip_far": float(self.clip_far),
        }
        return depth

    def _multi_ray(
        self,
        data: mujoco.MjData,
        pos: np.ndarray,
        vec: np.ndarray,
        geomgroup: np.ndarray,
        *,
        flg_static: int,
        nray: int,
    ) -> np.ndarray:
        """Return per-ray hit distance (negative = miss)."""
        dist = np.full(nray, -1.0, dtype=np.float64)
        geomid = np.full(nray, -1, dtype=np.int32)
        mujoco.mj_multiRay(
            self.model,
            data,
            pos,
            vec,
            np.ascontiguousarray(geomgroup),
            int(flg_static),
            -1,
            geomid,
            dist,
            None,
            nray,
            self.ray_maxdist,
        )
        return dist

    def _render_raycast(self, data: mujoco.MjData) -> np.ndarray:
        """Plane-style depth ≈ distance along camera look (ray * |dir_cam_z|).

        When ``exclude_robot`` is false, merge two passes (MuJoCo cannot hit static
        terrain and dynamic robot visuals in one ``mj_multiRay`` call):
          1) flg_static=1 + terrain group → floor/stairs
          2) flg_static=0 + visual group → arms/legs/torso
        Per pixel keep the nearer positive ray distance.

        Robot hits with image-plane depth ``< clip_near`` are discarded (MuJoCo
        head/torso visual sits ~6–9 cm in front of the camera). Those would
        otherwise be max-filled to ``clip_far`` (red) and occlude valid
        limb/terrain hits that Isaac RTX shows as near (blue).

        Catastrophic sky frames (body pitch-up → most rays miss → all-red with
        only arms blue) reuse the last good depth when ``hold_on_far_frac`` is set.
        """
        mujoco.mj_forward(self.model, data)
        pos = np.asarray(data.cam_xpos[self.camera_id], dtype=np.float64).reshape(3)
        R = np.asarray(data.cam_xmat[self.camera_id], dtype=np.float64).reshape(3, 3)
        self._last_cam_xpos = pos.copy()
        self._last_cam_xmat = R.copy()
        look_z = float((-R[:, 2])[2])
        nray = self.height * self.width
        dirs_cam = self._ray_dirs_cam.reshape(nray, 3)
        dirs_w = (R @ dirs_cam.T).T  # (nray, 3)
        vec = np.ascontiguousarray(dirs_w.reshape(-1), dtype=np.float64)
        z_scale = (-dirs_cam[:, 2]).clip(min=1e-8)

        rejected_robot = 0
        terrain_hit_frac = 0.0
        robot_planes_all = np.array([], dtype=np.float64)
        robot_planes_kept = np.array([], dtype=np.float64)

        if self._two_pass:
            assert self._gg_robot is not None
            dist_t = self._multi_ray(
                data, pos, vec, self._gg_terrain, flg_static=1, nray=nray
            )
            dist_r = self._multi_ray(
                data, pos, vec, self._gg_robot, flg_static=0, nray=nray
            )
            dist = np.full(nray, np.inf, dtype=np.float64)
            hit_t = dist_t > 0.0
            hit_r = dist_r > 0.0
            terrain_hit_frac = float(hit_t.mean())
            plane_r = dist_r * z_scale
            # Keep only robot hits at/after clip_near (Isaac-valid self-occlusion).
            hit_r_valid = hit_r & np.isfinite(plane_r) & (plane_r >= self.clip_near)
            hit_r_reject = hit_r & (~hit_r_valid)
            rejected_robot = int(hit_r_reject.sum())
            if hit_r.any():
                robot_planes_all = plane_r[hit_r]
            if hit_r_valid.any():
                robot_planes_kept = plane_r[hit_r_valid]
            dist[hit_t] = dist_t[hit_t]
            dist[hit_r_valid] = np.minimum(dist[hit_r_valid], dist_r[hit_r_valid])
            miss_ray = ~np.isfinite(dist)
        else:
            # Terrain-only (or single explicit group): static pass hits infinite planes.
            dist = self._multi_ray(
                data, pos, vec, self._gg_terrain, flg_static=1, nray=nray
            )
            miss_ray = dist < 0.0
            terrain_hit_frac = float((~miss_ray).mean())
            dist = np.where(miss_ray, np.inf, dist)

        # distance_to_image_plane; near-clip → clip_far (Isaac depth_clipping_behavior=max)
        plane = dist * z_scale
        true_miss = miss_ray | ~np.isfinite(plane)
        near_clip = (~true_miss) & (plane < self.clip_near)
        miss = true_miss | near_clip
        plane = np.where(miss, self.clip_far, np.minimum(plane, self.clip_far))
        depth = plane.reshape(self.height, self.width).astype(np.float32)
        far_fill_raw = float((depth >= self.clip_far - 1e-6).mean())
        held = 0.0
        if (
            self.hold_on_far_frac > 0.0
            and self._last_good_raw is not None
            and far_fill_raw >= self.hold_on_far_frac
        ):
            depth = self._last_good_raw.copy()
            held = 1.0
            far_fill = float((depth >= self.clip_far - 1e-6).mean())
        else:
            far_fill = far_fill_raw
            # Never promote a sky/far-dominated frame to "last good".
            if far_fill_raw < self.hold_on_far_frac:
                self._last_good_raw = depth.copy()

        self._last_raw = depth

        valid = ~miss
        def _pct(arr: np.ndarray, q: float) -> float:
            return float(np.percentile(arr, q)) if arr.size else float("nan")

        self._last_diag = {
            "nray": float(nray),
            "true_miss_frac": float(true_miss.mean()),
            "near_clip_frac": float(near_clip.mean()),
            "near_clip_rejected_robot_frac": float(rejected_robot) / float(nray),
            "valid_frac": float(valid.mean()),
            "far_fill_frac": far_fill,
            "far_fill_raw": far_fill_raw,
            "terrain_hit_frac": terrain_hit_frac,
            "look_z": look_z,
            "held": held,
            "robot_hit_n": float(robot_planes_all.size),
            "robot_kept_n": float(robot_planes_kept.size),
            "robot_hit_plane_p10": _pct(robot_planes_all, 10),
            "robot_hit_plane_p50": _pct(robot_planes_all, 50),
            "robot_hit_plane_p90": _pct(robot_planes_all, 90),
            "robot_kept_plane_p50": _pct(robot_planes_kept, 50),
            "clip_near": float(self.clip_near),
            "clip_far": float(self.clip_far),
        }
        return depth

    def render_nchw(self, data: mujoco.MjData) -> np.ndarray:
        from x2_depth_sim2sim.cfg import ensure_deploy_on_path

        ensure_deploy_on_path()
        from rl_deploy.core.depth_camera import normalize_depth_nchw

        raw = self.render_meters(data)
        nchw = normalize_depth_nchw(
            raw, height=self.height, width=self.width, d_max=self.d_max
        )
        self._last_nchw = nchw
        return nchw

    @property
    def last_raw(self) -> np.ndarray | None:
        return self._last_raw

    @property
    def last_nchw(self) -> np.ndarray | None:
        return self._last_nchw

    @property
    def last_cam_xpos(self) -> np.ndarray | None:
        return self._last_cam_xpos

    @property
    def last_cam_xmat(self) -> np.ndarray | None:
        return self._last_cam_xmat

    @property
    def last_diag(self) -> dict[str, float] | None:
        """Per-frame raycast diagnostics (miss / near-clip reject / valid fracs)."""
        return self._last_diag

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None
