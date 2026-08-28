"""Difficulty lanes for stairs+slope sim2sim scene."""

from __future__ import annotations

# Spawn on flat approach of each Y-lane (robot faces +X).
# Scene: assets/scene_stairs_slope.xml — three parallel corridors.
DIFFICULTY_LANES: dict[str, dict] = {
    "easy": {
        "label": "Easy",
        "spawn_xy": (0.0, 0.0),
        "n_steps": 4,
        "step_h": 0.08,
        "step_w": 0.30,
        "slope": 0.20,
        "blurb": "4×8cm stairs, slope 0.20",
    },
    "medium": {
        "label": "Medium",
        "spawn_xy": (0.0, 5.0),
        "n_steps": 5,
        "step_h": 0.12,
        "step_w": 0.30,
        "slope": 0.25,
        "blurb": "5×12cm stairs, slope 0.25",
    },
    "hard": {
        "label": "Hard",
        "spawn_xy": (0.0, 10.0),
        "n_steps": 5,
        "step_h": 0.20,
        "step_w": 0.30,
        "slope": 0.38,
        "blurb": "5×20cm stairs, slope 0.38",
    },
}

DEFAULT_DIFFICULTY = "easy"
SPAWN_Z = 0.68
