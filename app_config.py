# -*- coding: utf-8 -*-
"""Application-level constants for SimplePathPlanner."""

from dataclasses import dataclass

# Logical grid dimensions
GRID_WIDTH = 6
GRID_HEIGHT = 12

# Visual tuning
BACKGROUND_ALPHA = 0.35
DEFAULT_PATH_DENSITY = 20.0


@dataclass(frozen=True)
class ProfileConfig:
    background_image_path: str
    grid_x0: float
    grid_y0: float
    grid_x1: float
    grid_y1: float


DEFAULT_PROFILE = 0

# Profile-specific image and grid mapping configs.
PROFILE_CONFIGS: dict[int, ProfileConfig] = {
    0: ProfileConfig(
        background_image_path="bkgrd.png",
        grid_x0=2,
        grid_y0=796,
        grid_x1=372,
        grid_y1=54,
    ),
    1: ProfileConfig(
        background_image_path="bkgrd_red.png",
        grid_x0=70,
        grid_y0=785,
        grid_x1=440,
        grid_y1=44,
    ),
    2: ProfileConfig(
        background_image_path="bkgrd.png",
        grid_x0=2,
        grid_y0=796,
        grid_x1=372,
        grid_y1=54,
    ),
}


def get_profile_config(profile_id: int) -> ProfileConfig:
    try:
        key = int(profile_id)
    except (TypeError, ValueError) as e:
        raise ValueError(f"invalid profile id: {profile_id}") from e
    if key not in PROFILE_CONFIGS:
        choices = ", ".join(str(k) for k in sorted(PROFILE_CONFIGS))
        raise ValueError(f"profile {key} not found; available profiles: {choices}")
    return PROFILE_CONFIGS[key]
