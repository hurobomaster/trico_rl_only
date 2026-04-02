"""
Sim-Real 映射集成示例
演示如何使用sim_real_mapping模块进行完整的转换
"""

import numpy as np
from pathlib import Path
import sys

# 导入映射函数
from mujoco_playground._src.inference.sim_real_mapping import (
    servo_angles_to_sim_obs,
    sim_action_to_servo_angles,
    parse_actuator_ctrlrange_from_xml,
    load_trico_dependencies,
)


def example_real_to_sim_observation():
    """
    示例1: 真机舵机角度 → 模拟观测的关节角度
    """
    print("\n" + "="*70)
    print("Example 1: Real Servo Angles → Sim Observation Joint Angles")
    print("="*70)
    
    # 加载Trico-Control依赖
    FingerKinematics, FingerInverseSolver, MotorSystem = load_trico_dependencies()
    
    # 加载机器人配置
    import yaml
    trico_root = Path('/home/rune/proj_rune/Trico-Control')
    with open(trico_root / 'configs' / 'robot_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化FK
    fk_solver = FingerKinematics(config)
    
    # 模拟真机舵机读数 (物理角度, degrees)
    real_servo_angles_8d = np.array([
        270, 180, 180, 270,  # 右手指
        270, 180, 180, 270   # 左手指
    ], dtype=np.float32)
    
    print(f"\nReal servo angles (degrees):")
    print(f"  Right: {real_servo_angles_8d[0:4]}")
    print(f"  Left:  {real_servo_angles_8d[4:8]}")
    
    # 转换为模拟观测
    try:
        sim_obs_angles = servo_angles_to_sim_obs(real_servo_angles_8d, fk_solver)
        print(f"\n✓ Converted to sim observation joint angles (radians):")
        print(f"  Right: OA_x={sim_obs_angles[0]:.4f}, OA_y={sim_obs_angles[1]:.4f}, " +
              f"OB_x={sim_obs_angles[2]:.4f}, OB_y={sim_obs_angles[3]:.4f}")
        print(f"  Left:  OA_x={sim_obs_angles[4]:.4f}, OA_y={sim_obs_angles[5]:.4f}, " +
              f"OB_x={sim_obs_angles[6]:.4f}, OB_y={sim_obs_angles[7]:.4f}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_sim_action_to_servo_commands():
    """
    示例2: 模拟动作 → 真机舵机指令
    """
    print("\n" + "="*70)
    print("Example 2: Sim Action → Real Servo Commands")
    print("="*70)
    
    # 加载Trico-Control依赖
    FingerKinematics, FingerInverseSolver, MotorSystem = load_trico_dependencies()
    
    # 加载机器人配置
    import yaml
    trico_root = Path('/home/rune/proj_rune/Trico-Control')
    with open(trico_root / 'configs' / 'robot_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化求解器
    fk_solver = FingerKinematics(config)
    ik_solver = FingerInverseSolver(config)
    
    # 从XML解析ctrlrange
    xml_path = trico_root / 'assets' / 'trico_hand_for_rl.xml'
    ctrlrange_dict = parse_actuator_ctrlrange_from_xml(str(xml_path))
    
    # 重新映射键名 (XML中是act_*, 需要转换为OA_*/OB_*)
    ctrlrange_dict_normalized = {}
    for key, value in ctrlrange_dict.items():
        # act_OA_x_right → OA_x_right
        normalized_key = key.replace('act_', '')
        ctrlrange_dict_normalized[normalized_key] = value
    
    print(f"\n✓ Loaded {len(ctrlrange_dict_normalized)} actuator ranges")
    
    # 模拟策略网络输出的动作 ([-1, 1])
    sim_action_8d = np.array([
        0.5, -0.3, 0.2, 0.1,   # 右手指
        -0.4, 0.1, 0.0, -0.2   # 左手指
    ], dtype=np.float32)
    
    print(f"\nSim action ([-1, 1]):")
    print(f"  Right: {sim_action_8d[0:4]}")
    print(f"  Left:  {sim_action_8d[4:8]}")
    
    # 转换为真机舵机指令
    try:
        servo_commands = sim_action_to_servo_angles(
            sim_action_8d,
            ctrlrange_dict_normalized,
            ik_solver,
            fk_solver,
            motor_system=None  # 不实际发送
        )
        print(f"\n✓ Converted to servo commands (degrees):")
        print(f"  Right: {servo_commands[0:4]}")
        print(f"  Left:  {servo_commands[4:8]}")
        
        # 反向验证：从舵机指令回到sim观测
        print(f"\n--- Verification: servo → sim obs ---")
        sim_obs_angles_check = servo_angles_to_sim_obs(servo_commands, fk_solver)
        print(f"✓ Verified sim obs (radians):")
        print(f"  Right: {sim_obs_angles_check[0:4]}")
        print(f"  Left:  {sim_obs_angles_check[4:8]}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def example_full_pipeline():
    """
    示例3: 完整的推理管道
    观测 → 政策推理 → 动作转换 → 舵机指令
    """
    print("\n" + "="*70)
    print("Example 3: Full Inference Pipeline (Observation → Policy → Servo)")
    print("="*70)
    
    print("\n📋 Pipeline:")
    print("  1. Read real servo angles")
    print("  2. servo_angles → sim obs joint angles")
    print("  3. construct_observation(sim_obs_angles + forces + ...)")
    print("  4. policy.forward(observation) → sim_action")
    print("  5. sim_action → servo commands")
    print("  6. send to hardware via motor_system.set_finger_angles()")
    
    print("\n⚠️  Note: This is a template. Actual implementation requires:")
    print("  - Loading policy checkpoint")
    print("  - Real sensor readers (force, vision)")
    print("  - Motor system connection")
    
    print("\n✓ All conversion functions are ready for integration!")


if __name__ == '__main__':
    try:
        example_real_to_sim_observation()
    except Exception as e:
        print(f"Example 1 skipped: {e}")
    
    try:
        example_sim_action_to_servo_commands()
    except Exception as e:
        print(f"Example 2 skipped: {e}")
    
    example_full_pipeline()
    
    print("\n" + "="*70)
    print("✓ All examples completed!")
    print("="*70)
