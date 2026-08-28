"""Closed-loop MuJoCo + ONNX depth CNN runner (50 Hz policy).

Visualization is **viser** (browser), not native MuJoCo GUI / OpenCV windows.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from x2_depth_sim2sim.cfg import ensure_deploy_on_path
from x2_depth_sim2sim.depth import DepthRenderer
from x2_depth_sim2sim.env import X2MujocoEnv
from x2_depth_sim2sim.policy import DepthCnnPolicy


def _action_scale_from_cfg(cfg: dict[str, Any]) -> float | np.ndarray:
    s = cfg.get("action_scale", 0.25)
    if isinstance(s, (list, tuple, np.ndarray)):
        return np.asarray(s, dtype=np.float64)
    return float(s)


def _maybe_start_keyboard(s2s: dict[str, Any]):
    kb = s2s.get("keyboard") or {}
    if not kb.get("enable", True):
        return None
    ensure_deploy_on_path()
    from rl_deploy.core.keyboard_vel import try_start_tty_vel_listener

    return try_start_tty_vel_listener(
        float(kb.get("vx_mag", 0.05)),
        float(kb.get("wz_mag", 0.05)),
        vx_limit=float(kb.get("vx_limit", 0.5)),
        wz_limit=float(kb.get("wz_limit", 0.5)),
        vy_step=float(kb.get("vy_mag", 0.05)),
        vy_limit=float(kb.get("vy_limit", 0.2)),
        initial_vx=0.0,
        initial_vy=0.0,
        initial_wz=0.0,
    )


class Sim2SimRunner:
    def __init__(self, s2s: dict[str, Any], deploy: dict[str, Any]) -> None:
        ensure_deploy_on_path()
        self.s2s = s2s
        self.deploy = deploy
        sim = s2s.get("sim") or {}
        depth_cfg = s2s.get("depth") or {}
        dmeta = deploy.get("depth") or {}
        self.timestep = float(sim.get("timestep", 0.005))
        self.decimation = int(sim.get("decimation", 4))
        self.policy_dt = self.timestep * self.decimation
        self.duration_s = float(sim.get("duration_s", 0.0))

        names = list(deploy["policy_joint_names"])
        self.env = X2MujocoEnv(
            str(s2s["scene_xml"]),
            names,
            default_angles=np.asarray(deploy["default_angles"], dtype=np.float64),
            kps=np.asarray(deploy["kps"], dtype=np.float64),
            kds=np.asarray(deploy["kds"], dtype=np.float64),
            action_scale=_action_scale_from_cfg(deploy),
            timestep=self.timestep,
            camera_name=str(depth_cfg.get("camera_name", "rgbd_head_front")),
        )
        h = int(dmeta.get("height", depth_cfg.get("render_height", 40)))
        w = int(dmeta.get("width", depth_cfg.get("render_width", 64)))
        d_max = float(dmeta.get("d_max", 6.0))
        self.depth = DepthRenderer(
            self.env.model,
            camera_name=self.env.camera_name,
            height=h,
            width=w,
            d_max=d_max,
            clip_near=float(depth_cfg.get("clip_near", 0.2)),
            clip_far=float(depth_cfg.get("clip_far", d_max)),
            backend=str(depth_cfg.get("backend", "raycast")),  # type: ignore[arg-type]
            exclude_robot=bool(depth_cfg.get("exclude_robot", True)),
            terrain_geom_group=int(
                depth_cfg.get("terrain_geom_group", depth_cfg.get("robot_geom_group", 0))
            ),
            robot_visual_group=int(depth_cfg.get("robot_visual_group", 1)),
            ray_geom_groups=depth_cfg.get("ray_geom_groups"),
            hold_on_far_frac=float(depth_cfg.get("hold_on_far_frac", 0.50)),
            ray_maxdist=depth_cfg.get("ray_maxdist"),
        )
        self.policy = DepthCnnPolicy(deploy)
        self.cmd = np.zeros(3, dtype=np.float64)
        self._kb = _maybe_start_keyboard(s2s)
        self._viewer = None
        kb = s2s.get("keyboard") or {}
        self._vx_limit = float(kb.get("vx_limit", 0.5))
        self._vy_limit = float(kb.get("vy_limit", 0.2))
        self._wz_limit = float(kb.get("wz_limit", 0.5))
        self._vx_step = float(kb.get("vx_mag", 0.05))
        self._vy_step = float(kb.get("vy_mag", 0.05))
        self._wz_step = float(kb.get("wz_mag", 0.05))

        terrain_cfg = s2s.get("terrain") or {}
        self._multi_difficulty = bool(terrain_cfg.get("multi_difficulty", False))
        # Auto-enable if scene looks like stairs multi-lane
        scene_path = str(s2s.get("scene_xml", ""))
        if "stairs_slope" in scene_path:
            self._multi_difficulty = True
        from x2_depth_sim2sim.terrain import DEFAULT_DIFFICULTY, DIFFICULTY_LANES, SPAWN_Z

        diff = str(terrain_cfg.get("default_difficulty", DEFAULT_DIFFICULTY)).lower()
        if diff not in DIFFICULTY_LANES:
            diff = DEFAULT_DIFFICULTY
        self._difficulty = diff
        self._spawn_z = float(terrain_cfg.get("spawn_z", SPAWN_Z))

    def _spawn_xy(self) -> tuple[float, float]:
        from x2_depth_sim2sim.terrain import DIFFICULTY_LANES

        if not self._multi_difficulty:
            return (0.0, 0.0)
        return tuple(DIFFICULTY_LANES[self._difficulty]["spawn_xy"])  # type: ignore[return-value]

    def reset(self) -> None:
        self.env.reset(base_height=self._spawn_z, base_xy=self._spawn_xy())
        self.policy.reset()
        self.cmd[:] = 0.0
        if self._viewer is not None:
            try:
                self._viewer.set_cmd(0.0, 0.0, 0.0, push_sliders=True)
            except Exception:
                pass

    def set_difficulty(self, key: str) -> None:
        from x2_depth_sim2sim.terrain import DIFFICULTY_LANES

        key = str(key).lower()
        if key not in DIFFICULTY_LANES:
            return
        self._difficulty = key
        self.reset()
        print(
            f"[sim2sim] difficulty={key} "
            f"spawn_xy={self._spawn_xy()} ({DIFFICULTY_LANES[key]['blurb']})"
        )

    def _read_cmd(self) -> np.ndarray:
        # Prefer viser panel (browser); fall back to TTY keyboard.
        if self._viewer is not None:
            try:
                self.cmd[:] = self._viewer.read_cmd()
                return self.cmd
            except Exception:
                pass
        if self._kb is not None:
            try:
                vx, vy, wz = self._kb.read_cmd()
                self.cmd[:] = (float(vx), float(vy), float(wz))
            except Exception:
                pass
        return self.cmd

    def _open_viewer(self) -> None:
        viewer_cfg = self.s2s.get("viewer") or {}
        if not viewer_cfg.get("enable", True):
            return
        from x2_depth_sim2sim.viser_viewer import ViserSimViewer

        self._viewer = ViserSimViewer(
            self.env.model,
            self.env.data,
            host=str(viewer_cfg.get("host", "0.0.0.0")),
            port=int(viewer_cfg.get("port", 8080)),
            show_depth=bool(viewer_cfg.get("show_depth", True)),
            track_pelvis=bool(viewer_cfg.get("track_pelvis", True)),
            vx_limit=self._vx_limit,
            vy_limit=self._vy_limit,
            wz_limit=self._wz_limit,
            vx_step=self._vx_step,
            vy_step=self._vy_step,
            wz_step=self._wz_step,
            enable_difficulty=self._multi_difficulty,
            initial_difficulty=self._difficulty,
            show_fov=bool(viewer_cfg.get("show_fov", True)),
            camera_id=self.env.camera_id,
            fov_near=float(viewer_cfg.get("fov_near", 0.05)),
            fov_far=float(viewer_cfg.get("fov_length", 1.2)),
            fov_aspect=float(self.depth.width) / max(float(self.depth.height), 1.0),
        )

    def policy_step(self, *, t_s: float) -> tuple[np.ndarray, float]:
        cmd = self._read_cmd()
        quat, omega = self.env.get_pelvis_imu()
        q, dq = self.env.get_joint_pos_vel()
        obs = self.policy.build_obs(
            quat_wxyz=quat,
            ang_vel=omega,
            cmd=cmd,
            joint_pos=q,
            joint_vel=dq,
            phase_time_s=t_s,
        )
        depth_nchw = self.depth.render_nchw(self.env.data)
        action, ms = self.policy.forward(obs, depth_nchw)
        return action, ms

    def run(self) -> None:
        self.reset()
        self._open_viewer()
        viewer_cfg = self.s2s.get("viewer") or {}
        sync = bool(viewer_cfg.get("sync_realtime", True))

        print(
            f"[sim2sim] scene={self.s2s['scene_xml']}\n"
            f"[sim2sim] policy={self.deploy.get('policy_path')}\n"
            f"[sim2sim] dt={self.timestep} decimation={self.decimation} "
            f"policy_hz={1.0 / self.policy_dt:.1f}\n"
            f"[sim2sim] viz=viser  use right panel Velocity Command; Ctrl-C to quit"
        )

        t0 = time.time()
        sim_t = 0.0
        step_i = 0
        action = np.zeros(self.env.n, dtype=np.float64)
        try:
            while True:
                if self.duration_s > 0 and sim_t >= self.duration_s:
                    break
                if self._viewer is not None and not self._viewer.is_running:
                    break

                if self._viewer is not None and self._viewer.consume_reset_request():
                    self.reset()
                    sim_t = 0.0
                    step_i = 0
                    print("[sim2sim] reset requested from viser")

                if self._viewer is not None and self._multi_difficulty:
                    diff_req = self._viewer.consume_difficulty_request()
                    if diff_req is not None:
                        self.set_difficulty(diff_req)
                        sim_t = 0.0
                        step_i = 0

                wall0 = time.time()
                action, ms = self.policy_step(t_s=sim_t)
                for _ in range(self.decimation):
                    self.env.apply_pd(action)
                    self.env.step(1)
                    sim_t += self.timestep

                if self._viewer is not None:
                    self._viewer.sync(
                        sim_t=sim_t,
                        cmd=self.cmd,
                        infer_ms=ms,
                        depth_raw_m=self.depth.last_raw,
                        d_max=self.depth.d_max,
                        depth_diag=self.depth.last_diag,
                        depth_cam_xpos=self.depth.last_cam_xpos,
                        depth_cam_xmat=self.depth.last_cam_xmat,
                    )

                if step_i % 50 == 0:
                    print(
                        f"[sim2sim] t={sim_t:6.2f}s cmd={self.cmd.round(3).tolist()} "
                        f"infer={ms:.2f}ms"
                    )
                step_i += 1

                if sync:
                    elapsed = time.time() - wall0
                    sleep = self.policy_dt - elapsed
                    if sleep > 0:
                        time.sleep(sleep)
        except KeyboardInterrupt:
            print("\n[sim2sim] interrupted")
        finally:
            self.close()
            print(f"[sim2sim] done wall={time.time() - t0:.1f}s sim={sim_t:.1f}s")

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None
        try:
            self.depth.close()
        except Exception:
            pass
        if self._kb is not None:
            try:
                self._kb.stop()
            except Exception:
                pass
            self._kb = None
