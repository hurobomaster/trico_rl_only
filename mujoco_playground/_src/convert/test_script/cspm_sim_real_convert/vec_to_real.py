'''


输入： 单位向量

输出： 舵机视觉角度

目标对象是：inner cspm


'''


import sys
import os
import yaml
import numpy as np
from pathlib import Path

# --- 路径修正核心逻辑 ---
# 直接指向硬编码的 Trico-Control 根目录
TRICO_ROOT = Path("/home/rune/proj_rune/Trico-Control")

if not TRICO_ROOT.exists():
    print(f"❌ 错误: 无法找到 Trico-Control 项目根目录: {TRICO_ROOT}")
    sys.exit(1)

sys.path.append(str(TRICO_ROOT))
# ---------------------

from trico_code.mechanics.inverse_kinematics.cspm_inverse_kinematics import SingleCSPMInverseSolver

def load_config():
    config_path = os.path.join(str(TRICO_ROOT), "configs", "robot_config.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_inverse_test():
    print("="*60)
    print(" 🔄 CSPM 单层逆运动学解算测试 (Unit Vector -> Servo Angles)")
    print("="*60)
    
    cfg = load_config()
    # 默认测试 OA 链 (Inner Linkage)
    if 'geometry' not in cfg:
        print("❌ 配置文件中缺少 geometry 字段")
        return

    # 实例化解算器 (选择 layer_type='cspm_inner' 对应 OA 链)
    # 也可以改成 'cspm_outer' 来测 OB 链
    layer_name = 'cspm_inner'
    solver = SingleCSPMInverseSolver(cfg, layer_type=layer_name)
    
    print(f"当前解算模型: {layer_name}")
    print(f"参数: alpha1={solver.cfg['geometry'][layer_name]['alpha1_deg']}°, alpha2={solver.cfg['geometry'][layer_name]['alpha2_deg']}°")
    
    # 初始化解算器状态 (防止首次解算找不到参考点)
    # 使用 Home Pose 参考值 (如 170, 170)
    solver.initialize_state(170.0, 170.0)
    print("解算器状态已初始化: prev_right=170.0, prev_left=170.0 (Home Pose)")
    
    print("\n请输入目标单位向量 (x, y, z):")
    print("示例: 0, 0.8517, 0.524  (对应 Home Pose 的真实向量)")
    
    try:
        input_str = input("Vector > ").strip()
        if not input_str:
            print("❌ 输入为空")
            return
            
        input_str = input_str.replace("，", ",")
        vec_parts = [float(x) for x in input_str.split(",")]
        
        if len(vec_parts) != 3:
            print(f"❌ 需要3个分量，当前检测到 {len(vec_parts)} 个")
            return
            
        target_vec = np.array(vec_parts)
        norm = np.linalg.norm(target_vec)
        
        if norm < 1e-6:
            print("❌ 向量长度过小")
            return
            
        # 归一化输入向量
        unit_vec = target_vec / norm
        print(f"目标向量 (归一化): [{unit_vec[0]:.4f}, {unit_vec[1]:.4f}, {unit_vec[2]:.4f}]")
        
        # 调用核心解算函数
        result_angles = solver.compute_only(unit_vec)
        
        if result_angles:
            t_right, t_left = result_angles
            print(f"\n✅ 解算成功!")
            print(f"   Right Motor Angle: {t_right:.2f}°")
            print(f"   Left  Motor Angle: {t_left:.2f}°")
        else:
            print(f"\n❌ 解算失败 (目标不可达或无解)")
            
    except ValueError:
        print("❌ 输入格式错误，请输入三个数字")
    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":
    run_inverse_test()
