#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path

# 配置路径以确保能够导入 sim2real_bridge
CURRENT_DIR = Path(__file__).parent
PARENT_DIR = CURRENT_DIR.parent  # _src/convert
sys.path.insert(0, str(PARENT_DIR))

from sim2real_bridge import RealRobotBridge

def test_four_angles():
    # 初始化Bridge
    bridge = RealRobotBridge()
    
    # ---------------------------------------------------------
    # 测试1: 输入 4个舵机角度 -> 返回 4个Theta (默认以右手指为例)
    # ---------------------------------------------------------
    # 定义初始舵机角度
    input_servos = np.array([260, 160, 180, 260])
    servo_ids_right = [51, 52, 53, 54]  # 右手指舵机ID
    
    # 因为Bridge的接口是8个自由度(8D)，我们将后四个填成一样的防止报错
    servos_8d = np.concatenate([input_servos, input_servos])
    
    # 调用桥接函数: 舵机角 -> Hinge角
    hinges_8d = bridge.servo_angles_to_hinge_angles(servos_8d)
    
    # 提取前4个Theta (右手指的 OA_x, OA_y, OB_x, OB_y)
    output_thetas = hinges_8d[0:4]
    
    print("\n" + "="*70)
    print("【测试 1】: 舵机角度 -> Hinge角(弧度)")
    print("-" * 70)
    
    # 按舵机ID显示输入
    print("输入舵机角度:")
    for mid, angle in zip(servo_ids_right, input_servos):
        print(f"  舵机 {mid}: {angle:.2f}°")
    
    print(f"\n生成的Hinge角(rad): {output_thetas.round(6)}")
    print("  OA_x: {:.6f}  OA_y: {:.6f}".format(output_thetas[0], output_thetas[1]))
    print("  OB_x: {:.6f}  OB_y: {:.6f}".format(output_thetas[2], output_thetas[3]))
    print("="*70)
    
    # ---------------------------------------------------------
    # 测试2: 输入 4个Theta -> 返回 4个舵机角度
    # ---------------------------------------------------------
    # 使用刚刚生成的 Theta 作为输入
    input_thetas = output_thetas.copy()
    
    # 补齐到8D (左手的后4个Theta补0即可)
    thetas_8d = np.zeros(8, dtype=np.float32)
    thetas_8d[0:4] = input_thetas
    
    # 调用桥接函数: Hinge角 -> 舵机角 (注意: 不加Bank AB映射才能和真机输入完全对应对等)
    recovered_servos_8d = bridge.hinge_angles_to_servo_angles(thetas_8d, apply_bank_ab=False)
    
    # 提取前4个恢复出的舵机角度
    output_servos = recovered_servos_8d[0:4]
    
    print("\n" + "="*70)
    print("【测试 2】: Hinge角(弧度) -> 舵机角度")
    print("-" * 70)
    print(f"输入Hinge角(rad): {input_thetas.round(6)}")
    print("  OA_x: {:.6f}  OA_y: {:.6f}".format(input_thetas[0], input_thetas[1]))
    print("  OB_x: {:.6f}  OB_y: {:.6f}".format(input_thetas[2], input_thetas[3]))
    
    print(f"\n恢复的舵机角度:")
    for mid, angle in zip(servo_ids_right, output_servos):
        print(f"  舵机 {mid}: {angle:.2f}°")
    print("="*70)
    
    # ---------------------------------------------------------
    # 附加: 检查两者的差异
    # ---------------------------------------------------------
    diff = np.abs(input_servos - output_servos)
    print(f"\n==> 🔁 逆向反解测试误差检验: 最大误差为 {np.max(diff):.3f}°")
    if np.max(diff) < 0.1:
        print("    [✓] 完美转化无缝对齐！转换算法生效")

if __name__ == "__main__":
    try:
        test_four_angles()
    except Exception as e:
        print(f"执行出错: {e}")

