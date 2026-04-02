"""
JAX IK Utils for Trico Manipulation Task
包含几何参数、常量定义和核心 IK 算子
"""

import jax
import jax.numpy as jnp

# ==============================================================================
# 1. 几何参数常量 (必须定义在模块顶层)
# ==============================================================================

# 长度 [R_out, R_in, L_ext] (米)
# 85mm -> 0.085, 34mm -> 0.034, 97.45mm -> 0.09745
GEOM_LENS = jnp.array([0.085, 0.034, 0.09745], dtype=jnp.float32)

# 角度参数 [Out_a1, Out_a2, In_a1, In_a2] (弧度)
# 45, 62, 45, 62 度
_deg2rad = jnp.pi / 180.0
GEOM_ANGLES = jnp.array([45.0, 62.0, 45.0, 62.0], dtype=jnp.float32) * _deg2rad

# 初始冗余角 Psi (弧度)
# -90 度
PSI_RAD = -90.0 * _deg2rad

# 【修复点】Sim-to-Real 偏移量 (Radians)
# [Inner_Offset, Outer_Offset]
# 对应 XML 中的 euler="116 0 0" 和 "206 0 0"
OFFSETS = jnp.array([116.0, 206.0], dtype=jnp.float32) * _deg2rad

# ==============================================================================
# 2. 参数获取函数
# ==============================================================================

def get_ik_params():
    """生成并返回静态 IK 参数数组"""
    R_out, R_in, L_ext = GEOM_LENS
    
    # Outer CSPM
    a1_out, a2_out = GEOM_ANGLES[0], GEOM_ANGLES[1]
    out_sin_a1 = jnp.sin(a1_out)
    out_x_proj = jnp.cos(a1_out)
    out_cos_a2 = jnp.cos(a2_out)
    
    # Inner CSPM
    a1_in, a2_in = GEOM_ANGLES[2], GEOM_ANGLES[3]
    in_sin_a1 = jnp.sin(a1_in)
    in_x_proj = jnp.cos(a1_in)
    in_cos_a2 = jnp.cos(a2_in)
    
    return jnp.array([
        R_out, R_in, L_ext,
        out_sin_a1, out_x_proj, out_cos_a2,
        in_sin_a1, in_x_proj, in_cos_a2,
        PSI_RAD
    ], dtype=jnp.float32)

# ==============================================================================
# 3. IK Kernels (JIT Compiled)
# ==============================================================================

@jax.jit
def solve_cspm_layer(vec_target, sin_a1, x_proj, cos_a2, x_sign):
    """
    单层 CSPM 解算 Kernel
    """
    gx, gy, gz = vec_target
    
    # 构建方程 A*sin(t) + B*cos(t) = C
    A = sin_a1 * gy
    B = sin_a1 * (-gz)
    
    # x_proj_sign: 右手=+1, 左手=-1
    x_proj_val = x_proj * x_sign
    C = cos_a2 - (x_proj_val * gx)
    
    R = jnp.sqrt(A**2 + B**2 + 1e-9)
    
    # 基础角度
    base_angle = jnp.arcsin(jnp.clip(C / (R + 1e-9), -1.0, 1.0))
    phi = jnp.arctan2(B, A)
    
    # 两个解
    t1 = base_angle - phi
    t2 = (jnp.pi - base_angle) - phi
    
    # 归一化到 0 ~ 2pi
    t1 = t1 % (2 * jnp.pi)
    t2 = t2 % (2 * jnp.pi)
    
    return t1, t2

@jax.jit
def select_solution(t1, t2, prev_angle):
    """连续性选解: 选离 prev_angle 最近的"""
    diff1 = jnp.abs(t1 - prev_angle)
    diff2 = jnp.abs(t2 - prev_angle)
    
    # 处理环绕
    diff1 = jnp.where(diff1 > jnp.pi, 2*jnp.pi - diff1, diff1)
    diff2 = jnp.where(diff2 > jnp.pi, 2*jnp.pi - diff2, diff2)
    
    return jnp.where(diff1 < diff2, t1, t2)

@jax.jit
def finger_ik_kernel(target_pos, prev_phys_q, params):
    """
    整手 IK Kernel (系统级)
    """
    R_out, R_in, L_ext = params[0], params[1], params[2]
    psi_rad = params[9]
    
    # 1. 距离钳制 (Reachability)
    dist = jnp.linalg.norm(target_pos)
    max_reach = R_out + L_ext
    min_reach = jnp.abs(R_out - L_ext)
    
    scale = jnp.clip(dist, min_reach, max_reach) / (dist + 1e-9)
    P = target_pos * scale
    dist_clamped = dist * scale
    
    # 2. 余弦定理求 Phi
    num = R_out**2 + dist_clamped**2 - L_ext**2
    den = 2 * R_out * dist_clamped
    cos_phi = jnp.clip(num / (den + 1e-9), -1.0, 1.0)
    
    phi = jnp.arccos(cos_phi)
    sin_phi = jnp.sin(phi) # Elbow Up
    
    # 3. 构建最小扭曲平面 (Psi固定)
    u = P / (dist_clamped + 1e-9)
    x_axis = jnp.array([1.0, 0.0, 0.0])
    
    w = jnp.cross(P, x_axis)
    w_norm = jnp.linalg.norm(w)
    # 处理奇异
    w = jnp.where(w_norm < 1e-6, jnp.array([0., 1., 0.]), w / (w_norm + 1e-9))
    
    v = jnp.cross(w, u)
    
    # 旋转 Psi 角
    v_psi = v * jnp.cos(psi_rad) + w * jnp.sin(psi_rad)
    
    # 4. 计算 B 点和 A 点
    vec_B = R_out * (cos_phi * u + sin_phi * v_psi)
    
    vec_BP = P - vec_B
    norm_BP = jnp.linalg.norm(vec_BP)
    vec_A = (vec_BP / (norm_BP + 1e-9)) * R_in
    
    # 5. CSPM 反解
    out_sin, out_proj, out_cos = params[3], params[4], params[5]
    in_sin,  in_proj,  in_cos  = params[6], params[7], params[8]
    
    dir_B = vec_B / R_out
    dir_A = vec_A / R_in
    
    # --- 解算 ---
    # Motor 1 (Outer Right)
    out_r_1, out_r_2 = solve_cspm_layer(dir_B, out_sin, out_proj, out_cos, 1.0)
    theta_1 = select_solution(out_r_1, out_r_2, prev_phys_q[0])
    
    # Motor 2 (Inner Right)
    in_r_1, in_r_2 = solve_cspm_layer(dir_A, in_sin, in_proj, in_cos, 1.0)
    theta_2 = select_solution(in_r_1, in_r_2, prev_phys_q[1])
    
    # Motor 3 (Inner Left)
    in_l_1, in_l_2 = solve_cspm_layer(dir_A, in_sin, in_proj, in_cos, -1.0)
    theta_3 = select_solution(in_l_1, in_l_2, prev_phys_q[2])
    
    # Motor 4 (Outer Left)
    out_l_1, out_l_2 = solve_cspm_layer(dir_B, out_sin, out_proj, out_cos, -1.0)
    theta_4 = select_solution(out_l_1, out_l_2, prev_phys_q[3])
    
    return jnp.stack([theta_1, theta_2, theta_3, theta_4])