# 文件路径: mujoco_playground/_src/manipulation/trico/trico.py

"""
    __init__ 环境和物理引擎初始化设置
    reset 






"""







import os
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env
from ml_collections import config_dict


# Keep debug logging off by default: this env is traced/jitted by JAX during
# training, and NumPy-based formatting inside reset/step will break tracing.
DEBUG_TRICO_OBS = False


def _fmt_arr(arr) -> str:
    arr = np.asarray(arr)
    return np.array2string(
        arr,
        formatter={"float_kind": lambda x: f"{x:.1f}"},
        separator=", ",
    )


def _fmt_range(arr) -> str:
    arr = np.asarray(arr)
    return f"[{arr.min():.1f}, {arr.max():.1f}]"


def _fmt_ctrlrange_deg(ctrlrange_rad) -> str:
    """Format control range for each actuator (from XML ctrlrange)."""
    lines = []
    names = [
        "act_OA_x_right", "act_OA_y_right", "act_OB_x_right", "act_OB_y_right",
        "act_OA_x_left", "act_OA_y_left", "act_OB_x_left", "act_OB_y_left"
    ]
    for i, name in enumerate(names):
        low_deg = ctrlrange_rad[i, 0] * 180.0 / np.pi
        high_deg = ctrlrange_rad[i, 1] * 180.0 / np.pi
        lines.append(f"  [{i}] {name:20s}: [{low_deg:6.1f}, {high_deg:5.1f}]°")
    return "\n".join(lines)


def _rad_to_deg(arr):
    return jnp.asarray(arr) * 180.0 / jnp.pi


def _m_to_mm(arr):
    return jnp.asarray(arr) * 1000.0



class TricoEnv(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = None, **kwargs):
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        self._xml_path = os.path.join(curr_dir, 'xmls', 'trico_hand_for_rl.xml')

        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        
        # --- 求解器设置 (防穿透优化) ---
        self._mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
        self._mj_model.opt.iterations = 50   # 优化: 从100降至50 (加速+稳定性均衡)
        self._mj_model.opt.ls_iterations = 6 # 优化: 从10降至6
        self._mj_model.opt.tolerance = 1e-4

        self._mjx_model = mjx.put_model(self._mj_model)


        if config is None:
            config = config_dict.ConfigDict()
        
        
        config.ctrl_dt = 0.02  # 50Hz 控制
        config.sim_dt = 0.001  # 1ms 物理步长 (关键)
        
        
        if 'episode_length' not in config:
            config.episode_length = 500  # 优化: 5秒 (增加学习时间)
        if 'action_repeat' not in config:
            config.action_repeat = 1
            
        super().__init__(config=config)

        # 获取 Site ID
        self._tip_right_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "tip_right")
        self._tip_left_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "tip_left")
        
        # ★★★ 获取所有电机对应的关节初始位置 ★★★
        actuator_joint_ids = self._mj_model.actuator_trnid[:, 0]
        actuator_qpos_adr = self._mj_model.jnt_qposadr[actuator_joint_ids]
        self._actuator_qpos_adr = jnp.asarray(actuator_qpos_adr, dtype=jnp.int32)
        self._ctrlrange = jnp.asarray(self._mj_model.actuator_ctrlrange)
        self._init_ctrl_qpos = jnp.asarray(self._mj_model.qpos0[actuator_qpos_adr])
        
        # ★★★ 获取力传感器ID ★★★
        self._sensor_F_right_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "sensor_F_right")
        self._sensor_F_left_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "sensor_F_left")

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key = jax.random.split(rng)
        
        
        
        # 1. 物体复位 (XML前7位)
        obj_qpos = self.mjx_model.qpos0[:7]
        rng, key = jax.random.split(rng)
        
        # 随机平移
        
        # obj_pos_noise = jax.random.uniform(key, (2,), minval=-0.01, maxval=0.01)
        # obj_qpos = obj_qpos.at[:2].add(obj_pos_noise)
        
        
        # 固定高度 (防止开局穿模或掉落) - 根据实测地表高度 0.0289m，设为 0.0290m
        obj_qpos = obj_qpos.at[2].set(0.0290) 
        
        # 2. 手指复位 (XML后16位)
        hand_qpos = self.mjx_model.qpos0[7:]
        
        
        # 加微小随机扰动
        # hand_noise = jax.random.uniform(key, (hand_qpos.shape[0],), minval=-0.05, maxval=0.05)
        # hand_qpos = hand_qpos + hand_noise
        
        
        # 拼接
        qpos = jnp.concatenate([obj_qpos, hand_qpos])
        qvel = jnp.zeros(self.mjx_model.nv)
        
        data = mjx.make_data(self.mjx_model)
        data = data.replace(qpos=qpos, qvel=qvel)
        data = mjx.kinematics(self.mjx_model, data)
        
        obs = self._get_obs(data, jnp.zeros(6))
        if DEBUG_TRICO_OBS:
            print("[reset] obs shape:", obs.shape)
        reward, done = jnp.zeros(2)
        
        # 初始化与 step() 中完全一致的 metrics 字典结构
        metrics = {
            "reward_reach": 0.0,
            "reward_smooth": 0.0,
            "total_reward": 0.0,
            "reward": 0.0,
        }
        
        # 初始化滤波状态 (EMA prev values): [Fx_r, Fy_r, Fz_r, Fx_l, Fy_l, Fz_l]
        filtered_force = jnp.zeros(6)
        
        info = {
            "rng": rng, 
            "init_obj_height": obj_qpos[2],
            "filtered_force": filtered_force
        }
        
        obs = self._get_obs(data, filtered_force)
        if DEBUG_TRICO_OBS:
            print("[reset] obs (final) shape:", obs.shape)
        
        return mjx_env.State(data, obs, reward, done, metrics, info)


    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        
        
        '''
        1. 动作映射
        
        '''
        ctrl_center = (self._ctrlrange[:, 0] + self._ctrlrange[:, 1]) * 0.5
        ctrl_half = (self._ctrlrange[:, 1] - self._ctrlrange[:, 0]) * 0.5
        motor_targets = ctrl_center + action * ctrl_half
        motor_targets = jnp.clip(
            motor_targets, self._ctrlrange[:, 0], self._ctrlrange[:, 1]
        )





        if DEBUG_TRICO_OBS:
            print("[step] action (无量纲):", _fmt_arr(action))
            motor_targets_deg = _rad_to_deg(motor_targets)
            print("\n=== 8个舵机输出 ===")
            print("motor_targets (度):", _fmt_arr(motor_targets_deg))
            print("范围:", _fmt_range(motor_targets_deg))
        
        data = state.data.replace(ctrl=motor_targets)
        data = mjx.step(self.mjx_model, data)
        
        # --- 获取状态 ---
        obj_pos = data.qpos[:3]
        
        tip_right = data.site_xpos[self._tip_right_id]
        tip_left = data.site_xpos[self._tip_left_id]
        
        # --- 奖励计算 ---
        # 任务目标：指尖稳定接近物体中心 (Reach + Stability)
        
        dist_r = jnp.linalg.norm(tip_right - obj_pos)
        dist_l = jnp.linalg.norm(tip_left - obj_pos)
        
        # 30mm (0.03m) 半径区域内得分一样
        threshold = 0.03
        reach_dist_r = jnp.maximum(0.0, dist_r - threshold)
        reach_dist_l = jnp.maximum(0.0, dist_l - threshold)
        
        # ★ 改进奖励函数：鼓励长期稳定性而非仅接近
        # 在30mm内的奖励衰减更缓，鼓励维持接近
        reward_r = jnp.exp(-8.0 * reach_dist_r)
        reward_l = jnp.exp(-8.0 * reach_dist_l)
        reach_reward = (reward_r + reward_l) / 2.0

        # ★ 降低动作成本，避免过度平滑策略
        ctrl_cost = jnp.sum(jnp.square(action)) * 0.0005

        # ★ 添加速度平滑性鼓励 (防止抖动)
        hand_qvel = data.qvel[7:]  # 跳过物体的速度，只看手指
        vel_smooth_reward = -0.001 * jnp.sum(jnp.square(hand_qvel))

        # --- 总奖励组合 ---
        total_reward = reach_reward - ctrl_cost + vel_smooth_reward
        
        obs = self._get_obs(data, jnp.zeros(6))
        done = 0.0
        
        # 填充指标
        metrics = {
            "reward_reach": reach_reward,
            "reward_smooth": vel_smooth_reward,
            "total_reward": total_reward,
            "reward": total_reward,
        }
        
        # 更新 info
        new_info = state.info.copy()
        
        return state.replace(
            data=data, obs=obs, reward=total_reward, done=done, metrics=metrics, info=new_info
        )

    def _get_obs(self, data: mjx.Data, filtered_force: jax.Array) -> jax.Array:
        """
        观测空间 (27维):
        - 8维: 8个主动关节的当前角度值
        - 6维: 滤波后的力传感器读数 (Fx, Fy, Fz for right and left)
        - 7维: 物体位姿 (pos xyz + quat wxyz)
        - 6维: 指尖-物体相对位置 (right xyz + left xyz)
        """
        qpos_act = data.qpos[self._actuator_qpos_adr]
        obj_pose = data.qpos[:7]
        obj_pos = obj_pose[:3]
        tip_right = data.site_xpos[self._tip_right_id]
        tip_left = data.site_xpos[self._tip_left_id]
        rel_pos_right = obj_pos - tip_right
        rel_pos_left = obj_pos - tip_left
        rel_pos = jnp.concatenate([rel_pos_right, rel_pos_left])
        obs = jnp.concatenate([qpos_act, filtered_force, obj_pose, rel_pos])
        if DEBUG_TRICO_OBS:
            qpos_act_deg = _rad_to_deg(qpos_act)
            obj_pos_mm = _m_to_mm(obj_pos)
            obj_quat = np.asarray(obj_pose)[3:7]
            rel_pos_mm = _m_to_mm(rel_pos)
            obs_print = np.concatenate(
                [qpos_act_deg, np.asarray(filtered_force), obj_pos_mm, obj_quat, rel_pos_mm]
            )
            print("\n")
            print("[obs] qpos_act (8, 度):", _fmt_arr(qpos_act_deg))          
            print("[obs] qpos_act 当前值范围(度):", _fmt_range(qpos_act_deg))

            print("\n")  
            
            print("[obs] 物体位置 (3, mm):", _fmt_arr(obj_pos_mm))
            print("\n")


        return obs

    def render(self, trajectory, height, width, scene_option=None, camera=None):
        return super().render(
            trajectory=trajectory,
            height=height,
            width=width,
            scene_option=scene_option,
            camera="closeup"
        )


def default_config() -> config_dict.ConfigDict:
  """Default configuration for Trico environment."""
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.001,
      episode_length=500,
      action_repeat=1,
  )
