"""MuJoCo X2 env: joint I/O in Lab-BFS policy order, pelvis IMU, PD torque."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mujoco
import numpy as np


@dataclass
class JointMaps:
    names: list[str]
    qposadr: np.ndarray  # (n,)
    dofadr: np.ndarray  # (n,)
    actid: np.ndarray  # (n,) actuator ids for motor_{name}


class X2MujocoEnv:
    """Flat-ground X2 with head depth camera ``rgbd_head_front``."""

    def __init__(
        self,
        scene_xml: str,
        joint_names: Sequence[str],
        *,
        default_angles: np.ndarray,
        kps: np.ndarray,
        kds: np.ndarray,
        action_scale: float | np.ndarray,
        timestep: float = 0.005,
        camera_name: str = "rgbd_head_front",
    ) -> None:
        self.model = mujoco.MjModel.from_xml_path(scene_xml)
        self.model.opt.timestep = float(timestep)
        self.data = mujoco.MjData(self.model)
        self.joint_names = list(joint_names)
        self.n = len(self.joint_names)
        self.default = np.asarray(default_angles, dtype=np.float64).reshape(self.n)
        self.kps = np.asarray(kps, dtype=np.float64).reshape(self.n)
        self.kds = np.asarray(kds, dtype=np.float64).reshape(self.n)
        self.action_scale = np.asarray(action_scale, dtype=np.float64)
        if self.action_scale.ndim == 0:
            self.action_scale = np.full(self.n, float(self.action_scale), dtype=np.float64)
        elif self.action_scale.shape == (1,):
            self.action_scale = np.full(self.n, float(self.action_scale[0]), dtype=np.float64)
        else:
            self.action_scale = self.action_scale.reshape(self.n)

        self.maps = self._build_maps(self.joint_names)
        self.camera_name = camera_name
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if self.camera_id < 0:
            raise RuntimeError(f"camera '{camera_name}' not found in {scene_xml}")

        # pelvis freejoint + imu_0 sensors
        self._quat_sensor = "body-orientation"
        self._gyro_sensor = "body-angular-velocity"
        self._pelvis_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis"
        )

    def _build_maps(self, names: Sequence[str]) -> JointMaps:
        qposadr = np.zeros(len(names), dtype=np.int32)
        dofadr = np.zeros(len(names), dtype=np.int32)
        actid = np.zeros(len(names), dtype=np.int32)
        for i, name in enumerate(names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0:
                raise RuntimeError(f"joint '{name}' missing in MJCF")
            qposadr[i] = int(self.model.jnt_qposadr[jid])
            dofadr[i] = int(self.model.jnt_dofadr[jid])
            aid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}"
            )
            if aid < 0:
                raise RuntimeError(f"actuator motor_{name} missing")
            actid[i] = aid
        return JointMaps(list(names), qposadr, dofadr, actid)

    def reset(
        self,
        *,
        base_height: float | None = None,
        base_xy: tuple[float, float] | None = None,
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        z = 0.68 if base_height is None else float(base_height)
        x, y = (0.0, 0.0) if base_xy is None else (float(base_xy[0]), float(base_xy[1]))
        self.data.qpos[0:3] = np.array([x, y, z])
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
        self.data.qvel[:] = 0.0
        for i in range(self.n):
            self.data.qpos[self.maps.qposadr[i]] = self.default[i]
            self.data.qvel[self.maps.dofadr[i]] = 0.0
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def get_joint_pos_vel(self) -> tuple[np.ndarray, np.ndarray]:
        q = np.empty(self.n, dtype=np.float64)
        dq = np.empty(self.n, dtype=np.float64)
        for i in range(self.n):
            q[i] = self.data.qpos[self.maps.qposadr[i]]
            dq[i] = self.data.qvel[self.maps.dofadr[i]]
        return q, dq

    def get_pelvis_imu(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (quat_wxyz, ang_vel_body) from imu_0 sensors (pelvis)."""
        quat = np.asarray(self.data.sensor(self._quat_sensor).data, dtype=np.float64).copy()
        # MuJoCo framequat is wxyz
        omega = np.asarray(self.data.sensor(self._gyro_sensor).data, dtype=np.float64).copy()
        return quat, omega

    def apply_pd(self, action: np.ndarray) -> np.ndarray:
        """τ = kp (q_des - q) + kd (0 - dq); q_des = default + scale * action."""
        action = np.asarray(action, dtype=np.float64).reshape(self.n)
        q, dq = self.get_joint_pos_vel()
        q_des = self.default + self.action_scale * action
        tau = self.kps * (q_des - q) + self.kds * (0.0 - dq)
        for i in range(self.n):
            self.data.ctrl[self.maps.actid[i]] = tau[i]
        return q_des

    def step(self, n: int = 1) -> None:
        for _ in range(int(n)):
            mujoco.mj_step(self.model, self.data)
