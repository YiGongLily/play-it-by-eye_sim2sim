"""Load sim2sim yaml + deploy_real depth_cnn config (alignment / ONNX)."""

from __future__ import annotations

import os
import sys
from typing import Any

import yaml

# Headless-friendly GL before any mujoco import in callers.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEPLOY_ROOT = os.path.normpath(os.path.join(_PKG_ROOT, "..", "deploy_real"))


def package_root() -> str:
    return _PKG_ROOT


def deploy_root() -> str:
    return _DEPLOY_ROOT


def ensure_deploy_on_path() -> str:
    if _DEPLOY_ROOT not in sys.path:
        sys.path.insert(0, _DEPLOY_ROOT)
    return _DEPLOY_ROOT


def resolve_path(path: str, *, base: str | None = None) -> str:
    if os.path.isabs(path):
        return path
    root = base or _PKG_ROOT
    return os.path.normpath(os.path.join(root, path))


def load_sim2sim_yaml(path: str) -> dict[str, Any]:
    path = resolve_path(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid sim2sim config: {path}")
    cfg["__sim2sim_config_path__"] = os.path.abspath(path)
    cfg["__sim2sim_root__"] = _PKG_ROOT
    return cfg


def load_deploy_cfg(deploy_config: str) -> dict[str, Any]:
    """Same chain as deploy_real/tools/offline_depth_cnn_ort.load_deploy_cfg."""
    ensure_deploy_on_path()
    from tools.offline_depth_cnn_ort import load_deploy_cfg as _load

    path = resolve_path(deploy_config)
    if not os.path.isfile(path):
        # also try relative to deploy_real
        alt = os.path.normpath(os.path.join(_DEPLOY_ROOT, deploy_config))
        path = alt if os.path.isfile(alt) else path
    return _load(path)


def load_all(sim2sim_config: str) -> tuple[dict[str, Any], dict[str, Any]]:
    s2s = load_sim2sim_yaml(sim2sim_config)
    deploy = load_deploy_cfg(str(s2s["deploy_config"]))
    scene = resolve_path(str(s2s["scene_xml"]), base=_PKG_ROOT)
    s2s["scene_xml"] = scene
    return s2s, deploy
