"""
Deployment module - 部署脚本包

核心推理逻辑已迁移到 mujoco_playground._src.inference 模块。
本模块为向后兼容性而保留，从inference模块重导出。
"""

# 从新位置重导出 (保持向后兼容性)
from mujoco_playground._src.inference import (
    CheckpointLoader,
    ObservationAdapter,
    SimObservationAdapter,
    RealObservationAdapter,
    TricoPolicy,
    PolicyDeployer,
)

__all__ = [
    'CheckpointLoader',
    'ObservationAdapter',
    'SimObservationAdapter',
    'RealObservationAdapter',
    'TricoPolicy',
    'PolicyDeployer',
]
