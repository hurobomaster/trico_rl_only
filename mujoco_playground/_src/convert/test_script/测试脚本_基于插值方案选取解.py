#!/usr/bin/env python3
"""
插值版本测试：从HOME逐步插值走到目标角度，确保连续选解
"""
import sys
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent  # _src/convert
sys.path.insert(0, str(PARENT_DIR))

from sim2real_bridge import RealRobotBridge

def test_four_angles_with_interpolation(num_steps=20):
    """
    使用插值方式从HOME到目标，每步都是连续的小跳变
    """
    bridge = RealRobotBridge()
    
    # 定义目标舵机角度
    target_servos = np.array([280.0, 230.0, 230.0, 280.0])
    home_servos = np.array([260.0, 170.0, 170.0, 260.0])
    
    print("\n" + "="*60)
    print("【插值测试】从HOME逐步走到目标")
    print("="*60)
    print(f"HOME:   {home_servos}")
    print(f"TARGET: {target_servos}")
    print(f"插值步数: {num_steps}")
    
    # ---------------------------------------------------------
    # 第一阶段：插值走从HOME到目标
    # ---------------------------------------------------------
    print(f"\n【第一阶段】插值路径：HOME → TARGET")
    print("-" * 60)
    
    servo_trajectory = []
    hinge_trajectory = []
    
    for step in range(num_steps + 1):
        # 线性插值
        alpha = step / num_steps
        current_servos = home_servos + alpha * (target_servos - home_servos)
        
        # 补齐8D
        servos_8d = np.concatenate([current_servos, current_servos])
        
        # 观测转换
        hinges_8d = bridge.servo_angles_to_hinge_angles(servos_8d)
        
        servo_trajectory.append(current_servos.copy())
        hinge_trajectory.append(hinges_8d[0:4].copy())
        
        if step % max(1, num_steps // 5) == 0 or step == num_steps:
            print(f"  Step {step:2d}/{num_steps}: servo={current_servos.round(1)} → hinge={hinges_8d[0:4].round(4)}")
    
    # ---------------------------------------------------------
    # 第二阶段：从最后的Hinge角反解回舵机角
    # ---------------------------------------------------------
    print(f"\n【第二阶段】反解最终位置：TARGET_hinge → SERVO")
    print("-" * 60)
    
    final_hinges = hinge_trajectory[-1]
    print(f"最终Hinge角: {final_hinges.round(6)}")
    
    # 补齐8D
    thetas_8d = np.zeros(8, dtype=np.float32)
    thetas_8d[0:4] = final_hinges
    
    # 控制转换
    recovered_servos_8d = bridge.hinge_angles_to_servo_angles(thetas_8d, apply_bank_ab=False)
    recovered_servos = recovered_servos_8d[0:4]
    
    print(f"反解舵机角: {recovered_servos.round(1)}")
    
    # ---------------------------------------------------------
    # 第三阶段：验证精度
    # ---------------------------------------------------------
    print(f"\n【第三阶段】精度验证")
    print("-" * 60)
    
    diff = np.abs(target_servos - recovered_servos)
    print(f"目标舵机角: {target_servos}")
    print(f"恢复舵机角: {recovered_servos}")
    print(f"转换误差:   {diff}")
    print(f"最大误差:   {np.max(diff):.2f}°")
    
    if np.max(diff) < 0.1:
        print("\n✓ 完美转化无缝对齐！插值方案有效")
    else:
        print("\n✗ 仍有误差，需要调查")
    
    return servo_trajectory, hinge_trajectory


if __name__ == "__main__":
    try:
        # 尝试不同的插值步数
        for steps in [5, 10, 20, 50]:
            servo_traj, hinge_traj = test_four_angles_with_interpolation(num_steps=steps)
            print("\n")
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback
        traceback.print_exc()
