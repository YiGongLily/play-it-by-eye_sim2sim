"""Policy step: proprio history + depth + ONNX GRU."""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

import numpy as np

from x2_depth_sim2sim.cfg import ensure_deploy_on_path

ensure_deploy_on_path()

from rl_deploy.core.policy_observation import (  # noqa: E402
    actor_obs_deques_update,
    get_gravity_orientation,
)
from rl_deploy.core.policy_runtime import PolicyRuntime  # noqa: E402


class DepthCnnPolicy:
    def __init__(self, deploy_cfg: dict[str, Any]) -> None:
        self.cfg = deploy_cfg
        self.layout = list(deploy_cfg["alignment_obs_layout"])
        path = str(deploy_cfg["policy_path"])
        self.rt = PolicyRuntime(path)
        if self.rt.mode != PolicyRuntime.MODE_DEPTH_CNN_GRU:
            raise RuntimeError(f"expected depth_cnn_gru, got {self.rt.mode}")
        self.deques: Optional[list[deque]] = None
        self.last_action = np.zeros(int(deploy_cfg["num_actions"]), dtype=np.float64)
        self.rt.reset_recurrent()

    def reset(self) -> None:
        self.deques = None
        self.last_action = np.zeros_like(self.last_action)
        self.rt.reset_recurrent()

    def build_obs(
        self,
        *,
        quat_wxyz: np.ndarray,
        ang_vel: np.ndarray,
        cmd: np.ndarray,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        phase_time_s: float,
    ) -> np.ndarray:
        # depth_cnn: absolute joint angles (not relative to default)
        g_b = get_gravity_orientation(quat_wxyz)
        self.deques, obs = actor_obs_deques_update(
            self.layout,
            w_b=np.asarray(ang_vel, dtype=np.float64),
            g_b=np.asarray(g_b, dtype=np.float64),
            cmd_vec=np.asarray(cmd, dtype=np.float64).reshape(3),
            jp_rel=np.asarray(joint_pos, dtype=np.float64),
            jv_rel=np.asarray(joint_vel, dtype=np.float64),
            last_action=np.asarray(self.last_action, dtype=np.float64),
            phase_time_s=float(phase_time_s),
            deques=self.deques,
            quat_wxyz=np.asarray(quat_wxyz, dtype=np.float64),
        )
        return obs

    def forward(self, obs: np.ndarray, depth_nchw: np.ndarray) -> tuple[np.ndarray, float]:
        act, ms = self.rt.forward(obs, depth=depth_nchw)
        self.last_action = np.asarray(act, dtype=np.float64).reshape(-1)
        return self.last_action, float(ms)
