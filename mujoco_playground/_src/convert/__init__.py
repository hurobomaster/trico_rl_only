"""
Sim-to-Real 转换桥梁模块

核心功能：
  - RealRobotBridge: 真机与仿真之间的完整角度转换
    * servo_angles_to_hinge_angles(): 真机舵机角 → 仿真Hinge角（观测转换）
    * hinge_angles_to_servo_angles(): 仿真Hinge角 → 真机舵机角（动作转换）
  
  - 统一接口：屏蔽硬件细节（Bank A/B映射、坐标系转换）
  - 安全保障：动作前进行碰撞检测和范围验证
"""

from .sim2real_bridge import RealRobotBridge

__all__ = [
    'RealRobotBridge',
]
