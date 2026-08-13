"""Resolve worktree roots for local config and data defaults."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def repo_root() -> Path:
    """Directory that contains ``pyproject.toml`` and (usually) ``data/processed``."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    # src/geodude_teleoperation/paths.py → worktree
    return here.parents[2]


def config_dir() -> Path:
    return repo_root() / "config"


def pointclouds_config_path() -> Path:
    """Deprecated: teleop/field defaults live in ``SSoT_dof.yaml``."""
    return config_dir() / "pointclouds.yaml"


def ssot_dof_config_path() -> Path:
    return config_dir() / "SSoT_dof.yaml"


def data_processed_dir() -> Path:
    return repo_root() / "data" / "processed"


def default_ply_path() -> Path:
    return data_processed_dir() / "object_tag0_crop_voxel0.0020.ply"


def default_T_tag0_base_path() -> Path:
    return data_processed_dir() / "crop_meta.json"
