"""Thin wrapper for the canonical Sim2Real bridge implementation.

The canonical implementation lives in Trico-Control:
trico_code.policy_adapter.convert.sim2real_bridge.RealRobotBridge
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path


_WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
_TRICO_CONTROL_DIR = _WORKSPACE_ROOT / "Trico-Control"
if _TRICO_CONTROL_DIR.exists() and str(_TRICO_CONTROL_DIR) not in sys.path:
  sys.path.insert(0, str(_TRICO_CONTROL_DIR))

try:
  from trico_code.policy_adapter.convert.sim2real_bridge import (
      RealRobotBridge as _RealRobotBridge,
  )
except Exception as exc:  # pragma: no cover
  raise ImportError(
      "Unable to import canonical RealRobotBridge from Trico-Control. "
      "Ensure submodule Trico-Control is initialized and importable."
  ) from exc


class RealRobotBridge(_RealRobotBridge):
  """Compatibility wrapper that forwards to the canonical implementation."""

  def __init__(self, *args, **kwargs):
    warnings.warn(
        "mujoco_playground._src.convert.sim2real_bridge is now a thin wrapper. "
        "Use trico_code.policy_adapter.convert.sim2real_bridge.RealRobotBridge as canonical "
        "source.",
        DeprecationWarning,
        stacklevel=2,
    )
    super().__init__(*args, **kwargs)


__all__ = ["RealRobotBridge"]
