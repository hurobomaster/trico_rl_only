#!/usr/bin/env python3
"""
RealRobotBridge - 真机与仿真之间的完整转换桥梁

核心逻辑：
  观测: 真机舵机 [51,52,53,54,61,62,63,64] → 方向向量 → 仿真Hinge角 → 网络输入
  动作: 网络输出 → 仿真Hinge角 → 方向向量 → 真机舵机 → Bank A/B映射 → 硬件指令

设计原则：
  - 所有转换都经过方向向量（球关节指向）这个中间表示
  - 不直接比对仿真和真机的角度（两者单位、范围、坐标系不同）
  - 坐标系统一在 viz_base_frame 本地坐标系下
  - IK 状态同步至关重要：防止启动时的大跳变
"""

import sys
import os
import math
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import yaml

# 路径设置
TRICO_ROOT = Path("/home/rune/proj_rune/Trico-Control")
if TRICO_ROOT.exists():
    sys.path.insert(0, str(TRICO_ROOT))

try:
    from trico_code.mechanics.kinematics import FingerKinematics
    from trico_code.mechanics.inverse_kinematics.finger_inverse_kinematics import FingerInverseSolver
    from trico_code.mechanics.sim2real_bridge import SimRealBridge, VirtualHingeMapper
    TRICO_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Trico-Control not available: {e}")
    TRICO_AVAILABLE = False

# 导入仿真转换函数
PROJ_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJ_ROOT))




class RealRobotBridge:
    """
    真机与仿真的转换桥梁
    
    核心作用：
    1. 读取真机舵机角 → 转换为仿真Hinge角（用于观测）
    2. 处理仿真动作 → 转换为真机舵机角（用于执行）
    3. 处理Bank A/B映射（硬件特异性）
    4. 确保IK状态连续性（防止启动时跳变）
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化转换桥梁
        
        Args:
            config_path: Trico-Control robot_config.yaml 的路径
        """
        if not TRICO_AVAILABLE:
            raise RuntimeError("Trico-Control not available. Please check installation.")
        
        # 加载配置
        if config_path is None:
            config_path = str(TRICO_ROOT / "configs" / "robot_config.yaml")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        
        # 初始化运动学求解器
        self.fk = FingerKinematics(self.cfg)
        self.ik = FingerInverseSolver(self.cfg)
        
        # 初始化 SimRealBridge (用于正确的坐标系补偿转换)
        self.sim_real_bridge = SimRealBridge()
        
        # 获取几何参数
        self.geometry = self.cfg.get('geometry', {})
        
        # 获取Bank A/B映射
        fingers_cfg = self.cfg.get('fingers', {})
        
        finger_right_cfg = fingers_cfg.get('finger_right', {})
        finger_left_cfg = fingers_cfg.get('finger_left', {})
        
        banks_cfg_r = finger_right_cfg.get('banks', {})
        banks_cfg_l = finger_left_cfg.get('banks', {})
        
        # 构建Motor ID到Bank的映射
        self.motor_to_bank_right = {}
        self.motor_to_bank_left = {}
        
        # 右手指: Bank A (51, 52) 和 Bank B (53, 54)
        for mid in banks_cfg_r.get('bank_a', []):
            self.motor_to_bank_right[mid] = 'a'
        for mid in banks_cfg_r.get('bank_b', []):
            self.motor_to_bank_right[mid] = 'b'
        
        # 左手指: Bank A (61, 62) 和 Bank B (63, 64)
        for mid in banks_cfg_l.get('bank_a', []):
            self.motor_to_bank_left[mid] = 'a'
        for mid in banks_cfg_l.get('bank_b', []):
            self.motor_to_bank_left[mid] = 'b'
        
        # 计算初始位置偏置 (用于将仿真零位与真机初始位置对齐)
        servo_home = np.array([260.0, 170.0, 170.0, 260.0])
        
        # FK计算初始位置的方向向量
        dir_oa_home, dir_ob_home = self.fk.get_link_vectors(servo_home)
        
        # 使用 VirtualHingeMapper 的正确方法 (包含坐标系偏置补偿)
        oa_x_home, oa_y_home = self.sim_real_bridge.right_oa_mapper.get_sim_hinges_from_vector(dir_oa_home)
        ob_x_home, ob_y_home = self.sim_real_bridge.right_ob_mapper.get_sim_hinges_from_vector(dir_ob_home)
        
        self.hinge_offset_r = np.array([oa_x_home, oa_y_home, ob_x_home, ob_y_home])
        self.hinge_offset_l = np.array([oa_x_home, oa_y_home, ob_x_home, ob_y_home])
        
        # IK状态跟踪（用于连续性和同步）
        self.ik_synced = False
        
        # 同步IK冗余角到HOME位置，确保控制通路正确工作
        servo_home = np.array([260.0, 170.0, 170.0, 260.0])
        self.sync_ik_state_to_physical(servo_home, 'right')
        self.sync_ik_state_to_physical(servo_home, 'left')
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 【核心方法】观测转换: 真机舵机 → 仿真Hinge角
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def servo_angles_to_hinge_angles(self, servo_angles_8d: np.ndarray) -> np.ndarray:
        """
        将真机8个舵机角度转换为仿真的8个Hinge角度
        
        转换流程：
          真机舵机 [θ51,θ52,θ53,θ54,θ61,θ62,θ63,θ64] (度数)
            ↓ FK (get_link_vectors)
          OA/OB方向向量 (球关节指向，单位向量)
            ↓ VirtualHingeMapper (应用坐标系偏置补偿)
          仿真Hinge角 [θOA_x_r, θOA_y_r, ...] (弧度)
        
        Args:
            servo_angles_8d: (8,) 真机舵机角 (度数)
                           [motor_51, 52, 53, 54, 61, 62, 63, 64]
        
        Returns:
            hinge_angles_8d: (8,) 仿真Hinge角 (弧度)
                           [OA_x_r, OA_y_r, OB_x_r, OB_y_r,
                            OA_x_l, OA_y_l, OB_x_l, OB_y_l]
        """
        # 同步IK求解器的状态到当前舵机角，确保后续的compute_only()选择正确的解
        self.ik.ik_inner.update_state([servo_angles_8d[1], servo_angles_8d[2]])  # [θ52, θ53]
        self.ik.ik_outer.update_state([servo_angles_8d[0], servo_angles_8d[3]])  # [θ51, θ54]
        
        hinge_angles = []
        
        # ─────────────────────────────────────────────────────────────
        # 处理右手指
        # ─────────────────────────────────────────────────────────────
        right_servo = servo_angles_8d[0:4]  # [θ51, θ52, θ53, θ54]
        
        try:
            # FK: 舵机角 → 方向向量
            dir_oa_r, dir_ob_r = self.fk.get_link_vectors(right_servo)
            
            if dir_oa_r is None or dir_ob_r is None:
                raise ValueError(f"FK failed for right finger with angles: {right_servo}")
            
            # 转换: 方向向量 → Hinge角 (使用 VirtualHingeMapper 应用坐标系偏置)
            oa_x_r, oa_y_r = self.sim_real_bridge.right_oa_mapper.get_sim_hinges_from_vector(dir_oa_r)
            ob_x_r, ob_y_r = self.sim_real_bridge.right_ob_mapper.get_sim_hinges_from_vector(dir_ob_r)
            
            # 减去初始位置偏置，得到相对于仿真零位的角度
            hinge_angles.extend([
                oa_x_r - self.hinge_offset_r[0],
                oa_y_r - self.hinge_offset_r[1],
                ob_x_r - self.hinge_offset_r[2],
                ob_y_r - self.hinge_offset_r[3]
            ])
            
        except Exception as e:
            print(f"❌ Error converting right finger servo to hinge: {e}")
            raise
        
        # ─────────────────────────────────────────────────────────────
        # 处理左手指
        # ─────────────────────────────────────────────────────────────
        left_servo = servo_angles_8d[4:8]  # [θ61, θ62, θ63, θ64]
        
        # 同步IK求解器到左手指的舵机角
        self.ik.ik_inner.update_state([servo_angles_8d[5], servo_angles_8d[6]])  # [θ62, θ63]
        self.ik.ik_outer.update_state([servo_angles_8d[4], servo_angles_8d[7]])  # [θ61, θ64]
        
        try:
            # FK: 舵机角 → 方向向量
            dir_oa_l, dir_ob_l = self.fk.get_link_vectors(left_servo)
            
            if dir_oa_l is None or dir_ob_l is None:
                raise ValueError(f"FK failed for left finger with angles: {left_servo}")
            
            # 转换: 方向向量 → Hinge角 (使用 VirtualHingeMapper 应用坐标系偏置)
            oa_x_l, oa_y_l = self.sim_real_bridge.left_oa_mapper.get_sim_hinges_from_vector(dir_oa_l)
            ob_x_l, ob_y_l = self.sim_real_bridge.left_ob_mapper.get_sim_hinges_from_vector(dir_ob_l)
            
            # 减去初始位置偏置，得到相对于仿真零位的角度
            hinge_angles.extend([
                oa_x_l - self.hinge_offset_l[0],
                oa_y_l - self.hinge_offset_l[1],
                ob_x_l - self.hinge_offset_l[2],
                ob_y_l - self.hinge_offset_l[3]
            ])
            
        except Exception as e:
            print(f"❌ Error converting left finger servo to hinge: {e}")
            raise
        
        return np.array(hinge_angles, dtype=np.float32)  # (8,) radians
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 【核心方法】动作转换: 仿真Hinge角 → 真机舵机
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def hinge_angles_to_servo_angles(self, 
                                     hinge_angles_8d: np.ndarray,
                                     apply_bank_ab: bool = False) -> np.ndarray:
        """
        将仿真Hinge角度转换为真机舵机角度（Sim2Real动作）
        
        转换流程：
          仿真Hinge角 [θOA_x_r, θOA_y_r, ...] (弧度，相对于零位)
            ↓ 加上初始位置偏置
          绝对Hinge角 (弧度)
            ↓ VirtualHingeMapper: hinge_angles_to_direction()
          OA/OB方向向量 (球关节指向)
            ↓ IK反解
          真机舵机 [θ51, θ52, θ53, θ54, θ61, θ62, θ63, θ64] (度数，无Bank A/B映射)
            ↓ (可选) Bank A/B映射 (apply_bank_ab=True时)
          硬件指令
        
        Args:
            hinge_angles_8d: (8,) 仿真Hinge角 (弧度，相对于零位)
                           [OA_x_r, OA_y_r, OB_x_r, OB_y_r, OA_x_l, OA_y_l, OB_x_l, OB_y_l]
            apply_bank_ab: 是否应用Bank A/B映射 (默认False，不需要)
        
        Returns:
            servo_angles_8d: (8,) 真机舵机角 (度数，无Bank A/B映射)
                           [51, 52, 53, 54, 61, 62, 63, 64]
        """
        servo_angles = []
        
        # ─────────────────────────────────────────────────────────────
        # 处理右手指
        # ─────────────────────────────────────────────────────────────
        try:
            # Step 1: 相对Hinge角 → 绝对Hinge角（加上初始位置偏置）
            oa_x_r_abs = hinge_angles_8d[0] + self.hinge_offset_r[0]
            oa_y_r_abs = hinge_angles_8d[1] + self.hinge_offset_r[1]
            ob_x_r_abs = hinge_angles_8d[2] + self.hinge_offset_r[2]
            ob_y_r_abs = hinge_angles_8d[3] + self.hinge_offset_r[3]
            
            # Step 2: Hinge角 → 方向向量（使用VirtualHingeMapper应用坐标系偏置）
            dir_oa_r = self.sim_real_bridge.right_oa_mapper.get_vector_from_sim_hinges(oa_x_r_abs, oa_y_r_abs)
            dir_ob_r = self.sim_real_bridge.right_ob_mapper.get_vector_from_sim_hinges(ob_x_r_abs, ob_y_r_abs)
            
            # Step 3: 求解舵机角度（IK反解）
            right_servo_angles = self._solve_servo_from_directions(
                dir_oa_r, dir_ob_r, 'right'
            )
            
            if right_servo_angles is None:
                raise ValueError(f"IK failed for right finger with directions OA={dir_oa_r}, OB={dir_ob_r}")
            
            servo_angles.extend(right_servo_angles)
            
        except Exception as e:
            print(f"❌ Error converting right finger hinge to servo: {e}")
            raise
        
        # ─────────────────────────────────────────────────────────────
        # 处理左手指
        # ─────────────────────────────────────────────────────────────
        try:
            # Step 1: 相对Hinge角 → 绝对Hinge角（加上初始位置偏置）
            oa_x_l_abs = hinge_angles_8d[4] + self.hinge_offset_l[0]
            oa_y_l_abs = hinge_angles_8d[5] + self.hinge_offset_l[1]
            ob_x_l_abs = hinge_angles_8d[6] + self.hinge_offset_l[2]
            ob_y_l_abs = hinge_angles_8d[7] + self.hinge_offset_l[3]
            
            # Step 2: Hinge角 → 方向向量（使用VirtualHingeMapper应用坐标系偏置）
            dir_oa_l = self.sim_real_bridge.left_oa_mapper.get_vector_from_sim_hinges(oa_x_l_abs, oa_y_l_abs)
            dir_ob_l = self.sim_real_bridge.left_ob_mapper.get_vector_from_sim_hinges(ob_x_l_abs, ob_y_l_abs)
            
            # Step 3: 求解舵机角度（IK反解）
            left_servo_angles = self._solve_servo_from_directions(
                dir_oa_l, dir_ob_l, 'left'
            )
            
            if left_servo_angles is None:
                raise ValueError(f"IK failed for left finger with directions OA={dir_oa_l}, OB={dir_ob_l}")
            
            servo_angles.extend(left_servo_angles)
            
        except Exception as e:
            print(f"❌ Error converting left finger hinge to servo: {e}")
            raise
        
        servo_angles = np.array(servo_angles, dtype=np.float32)
        
        # ─────────────────────────────────────────────────────────────
        # 应用Bank A/B映射（可选）
        # ─────────────────────────────────────────────────────────────
        if apply_bank_ab:
            servo_angles = self._apply_bank_ab_mapping(servo_angles)
        
        return servo_angles  # (8,) degrees
    
    def _solve_servo_from_directions(self,
                                     dir_oa: np.ndarray,
                                     dir_ob: np.ndarray,
                                     finger: str,
                                     reference_servo_angles: Optional[np.ndarray] = None) -> Optional[List[float]]:
        """
        从方向向量求解舵机角（Sim2Real动作的核心）
        
        核心原理：OA和OB是**完全独立的两条链**，各自独立反解，与Psi无关
          - OA（内层）: dir_oa × l_in → 末端坐标 → SingleCSPMInverseSolver.compute_only() → [θ52, θ53]
          - OB（外层）: dir_ob × l_out → 末端坐标 → SingleCSPMInverseSolver.compute_only() → [θ51, θ54]
        
        Args:
            dir_oa, dir_ob: (3,) 单位方向向量
            finger: 'right' 或 'left'
            reference_servo_angles: 参考舵机角（未使用，保留接口兼容性）
        
        Returns:
            servo_angles: [θ51, θ52, θ53, θ54] (度数)，或None
        """
        try:
            # 从配置中获取链长度（mm）
            geometry = self.cfg.get('geometry', {})
            cspm_outer_cfg = geometry.get('cspm_outer', {})
            cspm_inner_cfg = geometry.get('cspm_inner', {})
            
            l_out = cspm_outer_cfg.get('l2_out', 30.0)  # OB链长度
            l_in = cspm_inner_cfg.get('l2_in', 30.0)    # OA链长度
            
            # Step 1: OA独立反解（内层）
            pos_oa = dir_oa * l_in
            result_oa = self.ik.ik_inner.compute_only(pos_oa)
            
            if result_oa is None:
                print(f"⚠️  IK inner solve failed for {finger} finger")
                return None
            
            # Step 2: OB独立反解（外层）
            pos_ob = dir_ob * l_out
            result_ob = self.ik.ik_outer.compute_only(pos_ob)
            
            if result_ob is None:
                print(f"⚠️  IK outer solve failed for {finger} finger")
                return None
            
            # Step 3: 组织输出
            # 电机顺序: [Motor51(outer_right), Motor52(inner_right), Motor53(inner_left), Motor54(outer_left)]
            # result_oa = [θ_inner_right, θ_inner_left]
            # result_ob = [θ_outer_right, θ_outer_left]
            servo_angles = [result_ob[0], result_oa[0], result_oa[1], result_ob[1]]
            
            return servo_angles
        
        except Exception as e:
            print(f"❌ Error in _solve_servo_from_directions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _apply_bank_ab_mapping(self, servo_angles_8d: np.ndarray) -> np.ndarray:
        """
        应用Bank A/B映射
        
        原理：
          Bank A (motors 51, 52): 互补映射 θ_cmd = 360 - θ
          Bank B (motors 53, 54): 直接映射 θ_cmd = θ
        
        Args:
            servo_angles_8d: (8,) 原始舵机角 (度数)
                           [51, 52, 53, 54, 61, 62, 63, 64]
        
        Returns:
            mapped_angles: (8,) 映射后的舵机角 (度数)
        """
        mapped = servo_angles_8d.copy()
        
        motor_ids_right = [51, 52, 53, 54]
        motor_ids_left = [61, 62, 63, 64]
        
        # 右手指映射
        for i, mid in enumerate(motor_ids_right):
            bank = self.motor_to_bank_right.get(mid, 'b')
            if bank == 'a':  # Bank A 做互补映射
                mapped[i] = 360.0 - servo_angles_8d[i]
            # Bank B 保持不变
        
        # 左手指映射
        for i, mid in enumerate(motor_ids_left):
            bank = self.motor_to_bank_left.get(mid, 'b')
            if bank == 'a':  # Bank A 做互补映射
                mapped[4 + i] = 360.0 - servo_angles_8d[4 + i]
            # Bank B 保持不变
        
        return mapped
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # 【同步方法】同步IK状态到物理舵机角度
    # ═══════════════════════════════════════════════════════════════════════════════
    
    def sync_ik_state_to_physical(self, current_servo_angles: np.ndarray, target_type: str = 'right') -> bool:
        """
        同步IK状态到当前物理舵机角度，防止启动时的大跳变
        
        Args:
            current_servo_angles: (4,) 当前物理舵机角 (度数)
            target_type: 'right' 或 'left'
        
        Returns:
            success: 是否成功同步
        """
        if target_type not in ['right', 'left']:
            raise ValueError(f"Unknown target_type: {target_type}")
        
        try:
            # 从当前舵机角计算末端坐标
            pos_fk = self.fk.forward_kinematics(current_servo_angles)
            
            if pos_fk is None:
                print(f"⚠️ FK failed for {target_type} finger")
                return False
            
            print(f"✓ FK得到{target_type}手指末端位置: {np.round(pos_fk, 2)} mm")
            
            # 扫描冗余角Psi以找到最接近物理角的解
            best_psi = 0.0
            min_diff = float('inf')
            found_solution = False
            
            phys_arr = np.array(current_servo_angles)
            
            # 扫描范围: -90 到 90 度，步长 2 度
            for psi_deg in range(-90, 91, 2):
                psi_rad = math.radians(psi_deg)
                
                # 调用IK内部逻辑进行试探
                u, v = self.ik._get_triangle_plane_basis(np.array(pos_fk), psi_rad)
                if u is None:
                    continue
                
                # 计算几何
                dist_p = np.linalg.norm(pos_fk)
                numerator = self.ik.R_out**2 + dist_p**2 - self.ik.L_ext**2
                denominator = 2 * self.ik.R_out * dist_p
                cos_phi = np.clip(numerator / denominator, -1.0, 1.0)
                phi = math.acos(cos_phi)
                sin_phi = math.sin(phi)
                
                vec_B = self.ik.R_out * (cos_phi * u + sin_phi * v)
                vec_BP = np.array(pos_fk) - vec_B
                norm_BP = np.linalg.norm(vec_BP)
                
                if norm_BP < 1e-6:
                    continue
                
                vec_A = (vec_BP / norm_BP) * self.ik.R_in
                
                # 求解
                res_outer = self.ik.ik_outer.compute_only(vec_B)
                res_inner = self.ik.ik_inner.compute_only(vec_A)
                
                if res_outer and res_inner:
                    calc_angles = np.array([res_outer[0], res_inner[0], res_inner[1], res_outer[1]])
                    diff = np.linalg.norm(calc_angles - phys_arr)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_psi = psi_rad
                        found_solution = True
            
            if found_solution:
                # 强制覆写IK内部状态
                self.ik.prev_psi = best_psi
                print(f"✓ {target_type}手指IK同步成功! 最佳冗余角 Psi = {math.degrees(best_psi):.1f}° (偏差 {min_diff:.1f}°)")
                
                # 再解一次以确保内部状态更新
                self.ik.solve(pos_fk)
                self.ik_synced = True
                return True
            else:
                print(f"⚠️ {target_type}手指IK同步失败: 未找到匹配的数学解")
                return False
        
        except Exception as e:
            print(f"❌ 同步IK状态时出错: {e}")
            import traceback
            traceback.print_exc()
            return False

