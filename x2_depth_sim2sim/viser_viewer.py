"""Browser MuJoCo visualization via viser + mjviser (no local OpenGL GUI)."""

from __future__ import annotations

import threading
from typing import Any

import numpy as np


class ViserSimViewer:
    """Passive viewer: host updates sim, this syncs mesh poses + optional depth image.

    Right-panel **Velocity Command** sliders drive ``vx / vy / wz`` for the policy.
    """

    def __init__(
        self,
        model,
        data,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        label: str = "X2 Depth Sim2Sim",
        show_depth: bool = True,
        track_pelvis: bool = True,
        vx_limit: float = 0.5,
        vy_limit: float = 0.2,
        wz_limit: float = 0.5,
        vx_step: float = 0.05,
        vy_step: float = 0.05,
        wz_step: float = 0.05,
        enable_difficulty: bool = False,
        initial_difficulty: str = "easy",
        show_fov: bool = True,
        camera_id: int = -1,
        fov_near: float = 0.05,
        fov_far: float = 1.2,
        fov_aspect: float = 64.0 / 40.0,
    ) -> None:
        import viser
        from mjviser.scene import ViserMujocoScene

        from x2_depth_sim2sim.terrain import DEFAULT_DIFFICULTY, DIFFICULTY_LANES

        self._data = data
        self._model = model
        self.show_depth = bool(show_depth)
        self._show_fov = bool(show_fov)
        self._camera_id = int(camera_id)
        self._fov_near = float(fov_near)
        self._fov_far = float(fov_far)
        self._fov_aspect = float(fov_aspect)
        self._lock = threading.Lock()
        self._cmd = np.zeros(3, dtype=np.float64)
        self._reset_requested = False
        self._stop_requested = False
        self._enable_difficulty = bool(enable_difficulty)
        init_diff = str(initial_difficulty or DEFAULT_DIFFICULTY).lower()
        if init_diff not in DIFFICULTY_LANES:
            init_diff = DEFAULT_DIFFICULTY
        self._difficulty = init_diff
        self._difficulty_requested: str | None = None

        self.vx_limit = abs(float(vx_limit))
        self.vy_limit = abs(float(vy_limit))
        self.wz_limit = abs(float(wz_limit))
        self.vx_step = float(vx_step)
        self.vy_step = float(vy_step)
        self.wz_step = float(wz_step)

        self.server = viser.ViserServer(host=host, port=port, label=label)
        self.scene = ViserMujocoScene(self.server, model, num_envs=1)
        try:
            self.scene.create_visualization_gui()
        except Exception:
            try:
                self.scene.create_scene_gui()
            except Exception:
                pass

        if track_pelvis:
            try:
                import mujoco

                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
                if bid >= 0 and hasattr(self.scene, "_tracked_body_id"):
                    self.scene._tracked_body_id = bid
                    if hasattr(self.scene, "camera_tracking_enabled"):
                        self.scene.camera_tracking_enabled = True
            except Exception:
                pass

        self._status = self.server.gui.add_markdown("**sim2sim** starting…")
        self._cmd_md = self.server.gui.add_markdown("cmd: `[0.000, 0.000, 0.000]`")
        if self._enable_difficulty:
            self._build_difficulty_panel()
        self._build_velocity_panel()

        self._depth_handle = None
        if self.show_depth:
            placeholder = np.zeros((40, 64, 3), dtype=np.uint8)
            self._depth_handle = self.server.gui.add_image(
                placeholder, label="depth (raw m → turbo)", format="jpeg"
            )

        self._fov_handle = None
        if self._show_fov and self._camera_id >= 0:
            self._init_fov_overlay()

        print(f"[viser] open http://127.0.0.1:{port} (or LAN IP:{port})")
        try:
            share = self.server.request_share_url()
            if share:
                print(f"[viser] share {share}")
        except Exception:
            pass

        self._running = True
        self.scene.update_from_mjdata(data)

    def _build_difficulty_panel(self) -> None:
        from x2_depth_sim2sim.terrain import DIFFICULTY_LANES

        gui = self.server.gui
        with gui.add_folder("Terrain Difficulty", expand_by_default=True):
            options = tuple(DIFFICULTY_LANES.keys())
            labels = [DIFFICULTY_LANES[k]["label"] for k in options]
            # Map dropdown label → key
            self._diff_label_to_key = {
                DIFFICULTY_LANES[k]["label"]: k for k in options
            }
            init_label = DIFFICULTY_LANES[self._difficulty]["label"]
            self._diff_dropdown = gui.add_dropdown(
                "Lane",
                options=tuple(labels),
                initial_value=init_label,
                hint="Easy/Medium/Hard corridors (parallel Y lanes)",
            )
            blurb0 = DIFFICULTY_LANES[self._difficulty]["blurb"]
            self._diff_md = gui.add_markdown(
                f"**{DIFFICULTY_LANES[self._difficulty]['label']}**: {blurb0}  \n"
                f"spawn y={DIFFICULTY_LANES[self._difficulty]['spawn_xy'][1]:.0f} m"
            )

            @self._diff_dropdown.on_update
            def _(_) -> None:
                label = str(self._diff_dropdown.value)
                key = self._diff_label_to_key.get(label)
                if key is None:
                    return
                self.request_difficulty(key)

            for key in ("easy", "medium", "hard"):
                meta = DIFFICULTY_LANES[key]
                btn = gui.add_button(f"{meta['label']}: {meta['blurb']}")

                @btn.on_click
                def _(_ev: Any, k: str = key) -> None:
                    self.request_difficulty(k)

    def request_difficulty(self, key: str) -> None:
        from x2_depth_sim2sim.terrain import DIFFICULTY_LANES

        key = str(key).lower()
        if key not in DIFFICULTY_LANES:
            return
        with self._lock:
            self._difficulty = key
            self._difficulty_requested = key
        meta = DIFFICULTY_LANES[key]
        try:
            self._diff_dropdown.value = meta["label"]
        except Exception:
            pass
        try:
            self._diff_md.content = (
                f"**{meta['label']}**: {meta['blurb']}  \n"
                f"spawn y={meta['spawn_xy'][1]:.0f} m — teleport + policy reset"
            )
        except Exception:
            pass

    def consume_difficulty_request(self) -> str | None:
        with self._lock:
            key = self._difficulty_requested
            self._difficulty_requested = None
            return key

    def _scene_offset(self) -> np.ndarray:
        """Match mjviser pelvis-tracking: visual scene is shifted by -tracked_pos."""
        scene = self.scene
        if getattr(scene, "camera_tracking_enabled", False):
            bid = getattr(scene, "_tracked_body_id", None)
            if bid is not None and int(bid) >= 0:
                return -np.asarray(self._data.xpos[int(bid)], dtype=np.float64).reshape(3)
        return np.zeros(3, dtype=np.float64)

    def _init_fov_overlay(self) -> None:
        pts = _depth_fov_segments(
            self._model,
            self._data,
            self._camera_id,
            near=self._fov_near,
            far=self._fov_far,
            aspect=self._fov_aspect,
            offset=self._scene_offset(),
        )
        self._fov_handle = self.server.scene.add_line_segments(
            "/depth_fov",
            pts,
            colors=(0, 220, 210),
            thickness=2.0,
            thickness_units="screen",
        )
        try:
            self._fov_cb = self.server.gui.add_checkbox(
                "Show depth FOV",
                initial_value=True,
                hint="Short frustum from rgbd_head_front (not clip_far=6m)",
            )

            @self._fov_cb.on_update
            def _(_ev: Any) -> None:
                if self._fov_handle is not None:
                    self._fov_handle.visible = bool(self._fov_cb.value)
        except Exception:
            self._fov_cb = None

    def _sync_fov_overlay(
        self,
        *,
        cam_xpos: np.ndarray | None = None,
        cam_xmat: np.ndarray | None = None,
    ) -> None:
        if self._fov_handle is None:
            return
        if getattr(self, "_fov_cb", None) is not None and not self._fov_cb.value:
            return
        self._fov_handle.points = _depth_fov_segments(
            self._model,
            self._data,
            self._camera_id,
            near=self._fov_near,
            far=self._fov_far,
            aspect=self._fov_aspect,
            offset=self._scene_offset(),
            cam_xpos=cam_xpos,
            cam_xmat=cam_xmat,
        )

    @property
    def difficulty(self) -> str:
        with self._lock:
            return self._difficulty

    def _build_velocity_panel(self) -> None:
        gui = self.server.gui
        with gui.add_folder("Velocity Command", expand_by_default=True):
            self._slider_vx = gui.add_slider(
                "vx (forward)",
                min=-self.vx_limit,
                max=self.vx_limit,
                step=min(self.vx_step, self.vx_limit / 10.0),
                initial_value=0.0,
                hint="m/s, body-frame forward",
            )
            self._slider_vy = gui.add_slider(
                "vy (lateral)",
                min=-self.vy_limit,
                max=self.vy_limit,
                step=min(self.vy_step, self.vy_limit / 10.0),
                initial_value=0.0,
                hint="m/s, body-frame left+",
            )
            self._slider_wz = gui.add_slider(
                "wz (yaw)",
                min=-self.wz_limit,
                max=self.wz_limit,
                step=min(self.wz_step, self.wz_limit / 10.0),
                initial_value=0.0,
                hint="rad/s, yaw rate",
            )

            @self._slider_vx.on_update
            def _(_) -> None:
                self._pull_sliders_to_cmd()

            @self._slider_vy.on_update
            def _(_) -> None:
                self._pull_sliders_to_cmd()

            @self._slider_wz.on_update
            def _(_) -> None:
                self._pull_sliders_to_cmd()

            stop_btn = gui.add_button("Stop (zero cmd)")
            reset_btn = gui.add_button("Reset robot")

            @stop_btn.on_click
            def _(_) -> None:
                self.set_cmd(0.0, 0.0, 0.0, push_sliders=True)

            @reset_btn.on_click
            def _(_) -> None:
                with self._lock:
                    self._reset_requested = True
                self.set_cmd(0.0, 0.0, 0.0, push_sliders=True)

            with gui.add_folder("Presets", expand_by_default=True):
                presets = [
                    ("Walk +0.2", 0.2, 0.0, 0.0),
                    ("Walk +0.4", 0.4, 0.0, 0.0),
                    ("Walk +0.6", 0.6, 0.0, 0.0),
                    ("Walk +0.8", 0.8, 0.0, 0.0),
                    ("Walk +1.0", 1.0, 0.0, 0.0),
                    ("Walk +1.2", 1.2, 0.0, 0.0),
                    ("Walk −0.2", -0.2, 0.0, 0.0),
                    ("Strafe L", 0.0, 0.15, 0.0),
                    ("Strafe R", 0.0, -0.15, 0.0),
                    ("Turn L", 0.0, 0.0, 0.3),
                    ("Turn R", 0.0, 0.0, -0.3),
                    ("Walk+Turn", 0.4, 0.0, 0.2),
                ]
                for name, vx, vy, wz in presets:
                    btn = gui.add_button(name)

                    @btn.on_click
                    def _(
                        _ev: Any,
                        vx: float = vx,
                        vy: float = vy,
                        wz: float = wz,
                    ) -> None:
                        self.set_cmd(vx, vy, wz, push_sliders=True)

    def _pull_sliders_to_cmd(self) -> None:
        self.set_cmd(
            float(self._slider_vx.value),
            float(self._slider_vy.value),
            float(self._slider_wz.value),
            push_sliders=False,
        )

    def set_cmd(
        self,
        vx: float,
        vy: float,
        wz: float,
        *,
        push_sliders: bool = False,
    ) -> None:
        vx = float(np.clip(vx, -self.vx_limit, self.vx_limit))
        vy = float(np.clip(vy, -self.vy_limit, self.vy_limit))
        wz = float(np.clip(wz, -self.wz_limit, self.wz_limit))
        with self._lock:
            self._cmd[:] = (vx, vy, wz)
        if push_sliders:
            # Avoid feedback loops: temporarily assign without relying on callbacks alone.
            self._slider_vx.value = vx
            self._slider_vy.value = vy
            self._slider_wz.value = wz
        self._cmd_md.content = f"cmd vx/vy/wz: `[{vx:.3f}, {vy:.3f}, {wz:.3f}]`"

    def read_cmd(self) -> np.ndarray:
        with self._lock:
            return self._cmd.copy()

    def consume_reset_request(self) -> bool:
        with self._lock:
            if not self._reset_requested:
                return False
            self._reset_requested = False
            return True

    @property
    def is_running(self) -> bool:
        return self._running and not self._stop_requested

    def sync(
        self,
        *,
        sim_t: float | None = None,
        cmd: np.ndarray | None = None,
        infer_ms: float | None = None,
        depth_raw_m: np.ndarray | None = None,
        d_max: float = 6.0,
        depth_diag: dict | None = None,
        depth_cam_xpos: np.ndarray | None = None,
        depth_cam_xmat: np.ndarray | None = None,
    ) -> None:
        self.scene.update_from_mjdata(self._data)
        bits = []
        if sim_t is not None:
            bits.append(f"t={sim_t:.2f}s")
        if infer_ms is not None:
            bits.append(f"infer={infer_ms:.2f}ms")
        if depth_diag:
            far = depth_diag.get("far_fill_raw", depth_diag.get("far_fill_frac"))
            look_z = depth_diag.get("look_z")
            terr = depth_diag.get("terrain_hit_frac")
            held = depth_diag.get("held", 0.0)
            if far is not None:
                bits.append(f"far={100.0 * float(far):.0f}%")
            if terr is not None:
                bits.append(f"terr={100.0 * float(terr):.0f}%")
            if look_z is not None:
                bits.append(f"look_z={float(look_z):+.2f}")
            if held:
                bits.append("HOLD")
        if bits:
            self._status.content = "**sim2sim** " + " · ".join(bits)
        if cmd is not None:
            c = np.asarray(cmd, dtype=np.float64).reshape(-1)
            self._cmd_md.content = (
                f"cmd vx/vy/wz: `[{c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}]`"
            )
        if self._depth_handle is not None and depth_raw_m is not None:
            self._depth_handle.image = _depth_to_rgb(depth_raw_m, d_max=d_max)
        self._sync_fov_overlay(cam_xpos=depth_cam_xpos, cam_xmat=depth_cam_xmat)

    def request_stop(self) -> None:
        self._running = False
        self._stop_requested = True

    def close(self) -> None:
        self._running = False
        try:
            self.server.stop()
        except Exception:
            pass


def _depth_fov_segments(
    model,
    data,
    camera_id: int,
    *,
    near: float,
    far: float,
    aspect: float,
    offset: np.ndarray | None = None,
    cam_xpos: np.ndarray | None = None,
    cam_xmat: np.ndarray | None = None,
) -> np.ndarray:
    """World-frame frustum edges for a MuJoCo camera (looks along −Z, +Y up).

    Apex is the camera origin. ``far`` is visualization length along the look
    axis (image-plane depth), not necessarily sensor clip_far.
    ``offset`` matches mjviser pelvis tracking (typically ``-pelvis_xpos``).

    Pass ``cam_xpos`` / ``cam_xmat`` from the depth render pose so the frustum
    matches the depth image (viewer sync runs after physics steps).
    """
    fovy = float(np.deg2rad(float(model.cam_fovy[camera_id])))
    hy = float(np.tan(0.5 * fovy))
    hx = float(aspect) * hy
    if cam_xpos is not None:
        pos = np.asarray(cam_xpos, dtype=np.float64).reshape(3).copy()
    else:
        pos = np.asarray(data.cam_xpos[camera_id], dtype=np.float64).reshape(3).copy()
    if cam_xmat is not None:
        rot = np.asarray(cam_xmat, dtype=np.float64).reshape(3, 3).copy()
    else:
        rot = np.asarray(data.cam_xmat[camera_id], dtype=np.float64).reshape(3, 3).copy()
    if offset is not None:
        pos = pos + np.asarray(offset, dtype=np.float64).reshape(3)

    def plane_corners(d: float) -> np.ndarray:
        d = float(d)
        local = np.array(
            [
                [hx * d, hy * d, -d],
                [-hx * d, hy * d, -d],
                [-hx * d, -hy * d, -d],
                [hx * d, -hy * d, -d],
            ],
            dtype=np.float64,
        )
        return (rot @ local.T).T + pos

    n = plane_corners(near)
    f = plane_corners(far)
    segs = []
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 0)):
        segs.append((n[a], n[b]))
        segs.append((f[a], f[b]))
        segs.append((pos, f[a]))
    return np.asarray(segs, dtype=np.float32)


def _depth_to_rgb(depth_m: np.ndarray, *, d_max: float = 6.0) -> np.ndarray:
    """Metres → RGB uint8 (approx turbo without requiring OpenCV GUI)."""
    x = np.clip(np.asarray(depth_m, dtype=np.float32) / max(float(d_max), 1e-6), 0.0, 1.0)
    r = np.clip(1.5 * x - 0.2, 0.0, 1.0)
    g = np.clip(1.5 - abs(2.0 * x - 1.0) * 1.5, 0.0, 1.0)
    b = np.clip(1.2 - 1.5 * x, 0.0, 1.0)
    rgb = np.stack([r, g, b], axis=-1)
    return (rgb * 255.0).astype(np.uint8)
