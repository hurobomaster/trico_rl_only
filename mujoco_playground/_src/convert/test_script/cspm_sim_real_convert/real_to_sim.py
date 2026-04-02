'''
输入： 舵机视觉角度

输出： 仿真Hinge角

目标对象是：inner cspm
'''



import sys
import os
import numpy as np
import yaml
import math
from pathlib import Path

# --- 路径修正核心逻辑 ---
# 直接指向硬编码的 Trico-Control 根目录
TRICO_ROOT = Path("/home/rune/proj_rune/Trico-Control")

if not TRICO_ROOT.exists():
    print(f"❌ 错误: 无法找到 Trico-Control 项目根目录: {TRICO_ROOT}")
    sys.exit(1)

sys.path.append(str(TRICO_ROOT))
# -----

from trico_code.mechanics.kinematics import FingerKinematics
from trico_code.mechanics.sim2real_bridge import SimRealBridge

def load_config():
    config_path = TRICO_ROOT / "configs" / "robot_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_debug_pipeline():
    print("="*60)
    print(" 🕵️‍♂️  Real2Sim (观测通路) 分步诊断工具")
    print("="*60)

    # 0. 准备工作
    cfg = load_config()
    fk = FingerKinematics(cfg)
    bridge = SimRealBridge()

    # --- 交互式输入 ---
    print("\n请输入肉眼观测到的/实际期望的 OA 链 2个物理角度 (对应电机 ID 52, 53)")
    try:
        input_str = input("格式如 170,170 (回车默认使用Home Pose): ").strip()
        if not input_str:
            # Home Pose default for inner motors
            input_angles = [170.0, 170.0]
        else:
            # 支持中文或英文逗号
            input_str = input_str.replace("，", ",")
            input_angles = [float(x) for x in input_str.split(",")]
            
        if len(input_angles) != 2:
            print(f"❌ 需要2个角度 (Motor 52, 53)，当前检测到 {len(input_angles)} 个。")
            return

        # 补全为4个角度供FK计算 [51, 52, 53, 54]
        # 51, 54 设为默认安全值 (虽不影响OA计算)
        physical_angles = [260.0, input_angles[0], input_angles[1], 260.0]
            
    except ValueError:
        print("❌ 输入格式错误，请确保输入2个数字，用逗号分隔。")
        return
        
    print(f"\n【当前输入 OA 物理角度】: Motor 52={physical_angles[1]}, Motor 53={physical_angles[2]}")

    # =================================================================
    # 🔵 第一段：FK 正运动学解算
    # 目标：物理角度 -> 三维空间向量
    # =================================================================
    print(f"\n" + "-"*30)
    print(f"Step 1: FK 正运动学 (获取 OA 空间向量)")
    print(f"-"*30)
    
    # 调用 FK
    dir_oa, dir_ob = fk.get_link_vectors(physical_angles)
    
    if dir_oa is None:
        print("❌ FK 解算失败 (OA无解)")
        return
    
    print(f"OA 连杆真实向量 (v_real): [{dir_oa[0]:.4f}, {dir_oa[1]:.4f}, {dir_oa[2]:.4f}]")
    
    # 简单验证：Home Pose 应该是笔直或对称的
    if abs(dir_oa[0]) < 1e-4:
        print(">> 状态: X分量为0，说明处于正中平面")
    else:
        print(f">> 状态: 存在侧向偏转 X={dir_oa[0]:.4f}")


    # =================================================================
    # 🟣 第二段：Sim2Real Bridge 映射
    # 目标：空间向量 -> 虚拟 Hinge 弧度 (θx, θy)
    # =================================================================
    print(f"\n" + "-"*30)
    print(f"Step 2: Bridge 坐标系补偿与逆解 (仅 OA)")
    print(f"-"*30)
    
    # 右手 OA (Offset 121.6°)
    print(f"[OA 连杆处理 (Offset 121.6°)]")
    
    # 解算
    oa_θx, oa_θy = bridge.right_oa_mapper.get_sim_hinges_from_vector(dir_oa)
    print(f"  -> 最终输出 (Degrees): θx={math.degrees(oa_θx):.4f}°, θy={math.degrees(oa_θy):.4f}°")
    print(f"  -> 最终输出 (Radians): θx={oa_θx:.4f} rad,  θy={oa_θy:.4f} rad")
    
    print("\n" + "="*60)
    print("✅ 诊断完成")

if __name__ == "__main__":
    run_debug_pipeline()
